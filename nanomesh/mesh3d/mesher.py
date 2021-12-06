from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Union

import matplotlib.pyplot as plt
import meshio
import numpy as np
from skimage import measure, morphology

from nanomesh._mesh_shared import BaseMesher
from nanomesh.mesh2d import simple_triangulate
from nanomesh.volume import Volume

from ..region_markers import RegionMarker, RegionMarkerLike
from .bounding_box import BoundingBox

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nanomesh.mesh import TriangleMesh


def get_point_in_prop(
        prop: measure._regionprops.RegionProperties) -> np.ndarray:
    """Uses `skeletonize` to find a point in the center of the regionprop.

    Parameters
    ----------
    prop : RegionProperties
        RegionProp from skimage.measure.regionproperties.

    Returns
    -------
    point : (3,) np.array
        Returns 3 indices describing a pixel in the labeled region.
    """
    skeleton = morphology.skeletonize(prop.image)
    coords = np.argwhere(skeleton)
    middle = len(coords) // 2
    try:
        point = coords[middle]
        point += np.array(prop.bbox[0:3])  # Add prop offset
    except IndexError:
        point = np.array(prop.centroid)
    return point


def get_region_markers(vol: Union[Volume, np.ndarray]) -> List[RegionMarker]:
    """Get region markers describing the featuers in the volume.

    The array will be labeled, and points inside the labeled region
    will be obtained using the `skeletonize` function. The region
    markers can be used to flood the connected regions in the
    tetrahedralization step.

    Parameters
    ----------
    vol : Union[Volume, np.array]
        Segmented integer volume.

    Returns
    -------
    region_markers : List[tuple]
        List of tuples. The first element is the label in the source image,
        and the second the pixel coordinates somewhere in the center of the
        corresponding region.
    """
    region_markers = []

    if isinstance(vol, Volume):
        image = vol.image
    else:
        image = vol

    labels = measure.label(image, background=-1, connectivity=1)

    props = measure.regionprops(labels, intensity_image=image)

    for prop in props:
        point = get_point_in_prop(prop)
        i, j, k = point.astype(int)
        label = image[i, j, k]

        region_marker = RegionMarker(label=label, coordinates=point)
        region_markers.append(region_marker)

    return region_markers


def add_corner_points(mesh: TriangleMesh, bbox: BoundingBox) -> None:
    """Add corner points from bounding box to mesh points.

    Parameters
    ----------
    mesh : TriangleMesh
        Mesh to add corner points to.
    bbox : BoundingBox
        Container for the bounding box coordinates.
    """
    corners = bbox.to_points()
    mesh.points = np.vstack([mesh.points, corners])


def close_side(mesh: TriangleMesh,
               *,
               side: str,
               bbox: BoundingBox,
               ax: plt.Axes = None):
    """Fill a side of the bounding box with triangles.

    Parameters
    ----------
    mesh : TriangleMesh
        Input contour mesh.
    side : str
        Side of the volume to close. Must be one of
        `left`, `right`, `top`, `bottom`, `front`, `back`.
    bbox : BoundingBox
        Coordinates of the bounding box.
    ax : plt.Axes, optional
        Plot the generated side on a matplotlib axis.

    Returns
    -------
    mesh : TriangleMesh
        Triangle mesh with the given side closed.

    Raises
    ------
    ValueError
        When the value of `side` is invalid.
    """
    from nanomesh.mesh import TriangleMesh
    all_points = mesh.points

    if side == 'top':
        edge_col = 2
        edge_value = bbox.zmin
    elif side == 'bottom':
        edge_col = 2
        edge_value = bbox.zmax
    elif side == 'left':
        edge_col = 1
        edge_value = bbox.ymin
    elif side == 'right':
        edge_col = 1
        edge_value = bbox.ymax
    elif side == 'front':
        edge_col = 0
        edge_value = bbox.xmin
    elif side == 'back':
        edge_col = 0
        edge_value = bbox.xmax
    else:
        raise ValueError('Side must be one of `right`, `left`, `bottom`'
                         f'`top`, `front`, `back`. Got {side=}')

    keep_cols = [col for col in (0, 1, 2) if col != edge_col]
    is_edge = all_points[:, edge_col] == edge_value

    coords = all_points[is_edge][:, keep_cols]

    edge_mesh = simple_triangulate(points=coords, opts='')
    cells = edge_mesh.cells_dict['triangle'].copy()

    shape = cells.shape
    new_cells = cells.ravel()

    mesh_edge_index = np.argwhere(is_edge).flatten()
    new_edge_index = np.arange(len(mesh_edge_index))
    mapping = np.vstack([new_edge_index, mesh_edge_index])

    mask = np.in1d(new_cells, mapping[0, :])
    new_cells[mask] = mapping[1,
                              np.searchsorted(mapping[0, :], new_cells[mask])]
    new_cells = new_cells.reshape(shape)

    new_labels = np.ones(len(new_cells))

    points = all_points
    cells = np.vstack([mesh.cells, new_cells])
    labels = np.hstack([mesh.labels, new_labels])

    mesh = TriangleMesh(points=points, cells=cells, labels=labels)

    if ax:
        edge_mesh.plot(ax=ax)
        ax.set_title(side)

    return mesh


def generate_envelope(mesh: TriangleMesh,
                      *,
                      bbox: BoundingBox,
                      plot: bool = False) -> TriangleMesh:
    """Wrap the surface mesh and close any open contours along the bounding
    box.

    Parameters
    ----------
    mesh : TriangleMesh
        Input mesh.
    bbox : BoundingBox
        Coordinates of the bounding box.

    Returns
    -------
    TriangleMesh
        Ouput mesh.
    """
    add_corner_points(mesh, bbox=bbox)

    sides = 'top', 'bottom', 'left', 'right', 'front', 'back'

    for i, side in enumerate(sides):
        mesh = close_side(mesh, side=side, bbox=bbox)

    return mesh


class Mesher3D(BaseMesher):
    def __init__(self, image: np.ndarray):
        super().__init__(image)
        self.contour: TriangleMesh
        self.pad_width = 0

    def generate_contour(
        self,
        level: float = None,
    ):
        """Generate contours using marching cubes algorithm.

        Also generates an envelope around the entire data volume
        corresponding to the bounding box.

        The bounding box equals the dimensions of the data volume.

        Parameters
        ----------
        level : float, optional
            Contour value to search for isosurfaces (i.e. the threshold value).
            By default takes the average of the min and max value. Can be
            ignored if a binary image is passed to `Mesher3D`.
        """
        from nanomesh.mesh import TriangleMesh

        points, cells, *_ = measure.marching_cubes(
            self.image,
            level=level,
            allow_degenerate=False,
        )

        mesh = TriangleMesh(points=points, cells=cells)

        bbox = BoundingBox.from_shape(self.image.shape)
        mesh = generate_envelope(mesh, bbox=bbox)

        logger.info(f'Generated contour with {len(mesh.cells)} cells')

        self.contour = mesh

    def pad_contour(self, **kwargs):
        """Pad the contour. See `nanomesh.TriangleMesh.pad3d` for info.

        Parameters
        ----------
        **kwargs
            Keyword arguments for `nanomesh.TriangleMesh.pad3d`.
        """
        self.contour = self.contour.pad3d(**kwargs)

    def show_contour(self, **kwargs):
        """Pad the contour. See `nanomesh.BaseMesh.plot_pyvista` for info.

        Parameters
        ----------
        **kwargs
            Keyword arguments for `nanomesh.BaseMesh.plot_pyvista`.
        """
        self.contour.plot_pyvista(**kwargs)

    def set_region_markers(self, region_markers: List[RegionMarkerLike]):
        """Sets custom region markers for tetrahedralization.

        Parameters
        ----------
        region_markers : List[RegionMarkerLike]
            List of `RegionMarker` objects or `(int, np.ndarray)` tuples.
        """
        self.contour.region_markers.clear()

        for region_marker in region_markers:
            self.contour.add_region_marker(region_marker)

    def tetrahedralize(self, generate_region_markers: bool = False, **kwargs):
        """Tetrahedralize a surface contour mesh.

        Parameters
        ----------
        generate_region_markers : bool, optional
            Attempt to automatically generate region markers.
            Overwrites existing region_markers.
        **kwargs
            Keyword arguments passed to
            `nanomesh.mesh_container.TriangleMesh.tetrahedralize`

        Returns
        -------
        TetraMesh

        Raises
        ------
        ValueError
            Description
        """
        if not self.contour:
            raise ValueError('No contour mesh available.'
                             'Run `Mesher3D.generate_contour()` first.')

        if generate_region_markers:
            region_markers = get_region_markers(self.image)
            self.contour.region_markers = region_markers

        contour = self.contour
        volume_mesh = contour.tetrahedralize(**kwargs)
        return volume_mesh


def generate_3d_mesh(
    image: np.ndarray,
    *,
    level: float = None,
    **kwargs,
) -> 'meshio.Mesh':
    """Generate mesh from binary (segmented) image.

    Parameters
    ----------
    image : 3D np.ndarray
        Input image to mesh.
    level : float, optional
        Contour value to search for isosurfaces (i.e. the threshold value).
        By default takes the average of the min and max value. Can be
        ignored if a binary image is passed as `image`.
    **kwargs
        Optional keyword arguments passed to
        `nanomesh.mesh_container.TriangleMesh.tetrahedralize`

    Returns
    -------
    volume_mesh : TetraMesh
        Description of the mesh.
    """
    mesher = Mesher3D(image)
    mesher.generate_contour(level=level)

    volume_mesh = mesher.tetrahedralize(**kwargs)
    return volume_mesh
