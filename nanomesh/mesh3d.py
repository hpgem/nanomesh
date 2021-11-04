import logging
from dataclasses import dataclass
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import meshio
import numpy as np
from skimage import measure, morphology

from nanomesh import Volume
from nanomesh.mesh_utils import simple_triangulate

from ._mesh_shared import BaseMesher
from .mesh_container import TriangleMesh

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Container for bounding box coordinates."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @classmethod
    def from_shape(cls, shape: tuple):
        """Generate bounding box from data shape."""
        xmin, ymin, zmin = 0, 0, 0
        xmax, ymax, zmax = np.array(shape) - 1
        return cls(xmin=xmin,
                   ymin=ymin,
                   zmin=zmin,
                   xmax=xmax,
                   ymax=ymax,
                   zmax=zmax)

    @classmethod
    def from_points(cls, points: np.array):
        """Generate bounding box from set of points or coordinates."""
        xmax, ymax, zmax = points.max(axis=0)
        xmin, ymin, zmin = points.min(axis=0)

        return cls(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
        )

    def to_points(self):
        """Return (m,3) array with corner points."""
        return np.array([
            [self.xmin, self.ymin, self.zmin],
            [self.xmin, self.ymin, self.zmax],
            [self.xmin, self.ymax, self.zmin],
            [self.xmin, self.ymax, self.zmax],
            [self.xmax, self.ymin, self.zmin],
            [self.xmax, self.ymin, self.zmax],
            [self.xmax, self.ymax, self.zmin],
            [self.xmax, self.ymax, self.zmax],
        ])

    @property
    def center(self) -> np.ndarray:
        """Return center of the bounding box."""
        return np.array(
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
            (self.zmin + self.ymax) / 2,
        )


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


def get_region_markers(
        vol: Union[Volume, np.ndarray]) -> List[Tuple[int, np.ndarray]]:
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
        region_markers.append((label, point))

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


def close_side(mesh, *, side: str, bbox: BoundingBox, ax: plt.Axes = None):
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

    mesh_edge_index = np.argwhere(is_edge).flatten()
    new_edge_index = np.arange(len(mesh_edge_index))
    mapping = np.vstack([new_edge_index, mesh_edge_index])

    shape = edge_mesh.cells.shape
    new_cells = edge_mesh.cells.copy().ravel()

    mask = np.in1d(new_cells, mapping[0, :])
    new_cells[mask] = mapping[1,
                              np.searchsorted(mapping[0, :], new_cells[mask])]
    new_cells = new_cells.reshape(shape)

    new_labels = np.ones(len(new_cells)) * 123

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

    def tetrahedralize(self, generate_region_markers: bool = False, **kwargs):
        """Tetrahedralize a surface contour mesh.

        Parameters
        ----------
        generate_region_markers : bool, optional
            Attempt to automatically generate region markers.
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
            kwargs['region_markers'] = region_markers

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
