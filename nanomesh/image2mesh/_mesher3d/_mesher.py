from __future__ import annotations

from typing import TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology

from nanomesh import Volume, simple_triangulate, tetrahedralize
from nanomesh._constants import BACKGROUND, FEATURE
from nanomesh._doc import doc
from nanomesh.mesh import TriangleMesh
from nanomesh.region_markers import RegionMarker, RegionMarkerList

from .._mesher import Mesher
from ._bounding_box import BoundingBox
from ._helpers import pad

if TYPE_CHECKING:
    from nanomesh import MeshContainer


def get_point_in_prop(
        prop: measure._regionprops.RegionProperties) -> np.ndarray:
    """Uses `skeletonize` to find a point in the center of the regionprop.

    Parameters
    ----------
    prop : RegionProperties
        RegionProp from :mod:`skimage.measure.regionproperties`.

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


def get_region_markers(vol: Union[Volume, np.ndarray]) -> RegionMarkerList:
    """Get region markers describing the featuers in the volume.

    The array will be labeled, and points inside the labeled region
    will be obtained using the `skeletonize` function. The region
    markers can be used to flood the connected regions in the
    tetrahedralization step.

    Parameters
    ----------
    vol : Union[Volume, np.ndarray]
        Segmented integer volume.

    Returns
    -------
    region_markers : List[RegionMArker]
        List of tuples. The first element is the label in the source image,
        and the second the pixel coordinates somewhere in the center of the
        corresponding region.
    """
    region_markers = RegionMarkerList()

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

        if label == 0:
            name = 'background'
            label = BACKGROUND
        else:
            name = 'X'
            label = FEATURE

        region_markers.append(RegionMarker(label=label, point=point,
                                           name=name))

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
    ax : matplotlib.axes.Axes, optional
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


@doc(Mesher, prefix='tetrahedral mesh from 3D (volumetric) image data')
class Mesher3D(Mesher, ndim=3):

    def __init__(self, image: np.ndarray):
        super().__init__(image)
        self.contour: TriangleMesh

    def generate_contour(
        self,
        level: float = None,
        group_regions: bool = True,
    ):
        """Generate contours using marching cubes algorithm
        (:func:`skimage.measure.marching_cubes`).

        Also generates an envelope around the entire data volume
        corresponding to the bounding box. The bounding box equals
        the dimensions of the data volume.

        By default, the 0-value after segmentation will map to
        the 'background', and the 1-value to `feature`.

        Parameters
        ----------
        level : float, optional
            Contour value to search for isosurfaces (i.e. the threshold value).
            By default takes the average of the min and max value. Can be
            ignored if a binary image is passed to :class:`Mesher3D`.
        group_regions : bool, optional
            If True, assign the same label to all features
            If False, label regions sequentially
        """
        from nanomesh.mesh import TriangleMesh

        points, cells, *_ = measure.marching_cubes(
            self.image,
            level=level,
            allow_degenerate=False,
        )

        contour = TriangleMesh(points=points, cells=cells)

        bbox = BoundingBox.from_shape(self.image.shape)
        contour = generate_envelope(contour, bbox=bbox)

        if level:
            segmented = self.image.binary_digitize(threshold=level)
        else:
            segmented = self.image

        regions = get_region_markers(segmented)

        if not group_regions:
            regions = regions.label_sequentially(FEATURE, fmt_name='X{}')

        contour.region_markers = regions

        self.contour = contour

    @doc(pad, prefix='Pad the contour using :func:`image2mesh._mesher3d.pad`')
    def pad_contour(self, **kwargs):
        self.contour = pad(self.contour, **kwargs)

    @doc(TriangleMesh.plot_pyvista,
         prefix='Shortcut for :meth:`TriangleMesh.plot_pyvista`.')
    def plot_contour(self, **kwargs):
        self.contour.plot_pyvista(**kwargs)

    def set_region_markers(self, region_markers: RegionMarkerList):
        """Sets custom region markers for tetrahedralization.

        This overwrites any contours generated by `.generate_contour()`.

        Parameters
        ----------
        region_markers : RegionMarkerList
            List of `RegionMarker` objects.
        """
        self.contour.region_markers.clear()
        self.contour.region_markers.extend(region_markers)

    @doc(tetrahedralize,
         prefix=('Tetrahedralize surface contour mesh using'
                 ':func:`tetrahedralize`'))
    def tetrahedralize(self, **kwargs) -> MeshContainer:
        if not self.contour:
            raise ValueError('No contour mesh available.'
                             'Run `Mesher3D.generate_contour()` first.')

        contour = self.contour
        mesh = contour.tetrahedralize(**kwargs)

        mesh.set_field_data('tetra', {0: 'background', 1: 'X'})
        fields = {
            m.label: m.name
            for m in self.contour.region_markers if m.name
        }
        mesh.set_field_data('tetra', fields)

        return mesh


def volume2mesh(
    image: np.ndarray | Volume,
    *,
    level: float = None,
    **kwargs,
) -> 'MeshContainer':
    """Generate a tetrahedral mesh from a 3D segmented image.

    Parameters
    ----------
    image : (i,j,k) numpy.ndarray or Volume
        Input image to mesh.
    level : float, optional
        Contour value to search for isosurfaces (i.e. the threshold value).
        By default takes the average of the min and max value. Can be
        ignored if a binary image is passed as `image`.
    **kwargs
        Optional keyword arguments passed to
        :func:`tetrahedralize`

    Returns
    -------
    volume_mesh : MeshContainer
        Instance of :class:`MeshContainer`
    """
    mesher = Mesher3D(image)
    mesher.generate_contour(level=level)

    volume_mesh = mesher.tetrahedralize(**kwargs)
    return volume_mesh
