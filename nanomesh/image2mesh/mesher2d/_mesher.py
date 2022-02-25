from __future__ import annotations

from typing import TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure

from nanomesh._doc import doc
from nanomesh.region_markers import RegionMarker, RegionMarkerList

from .._base import AbstractMesher
from ._helpers import append_to_segment_markers, generate_segment_markers, pad
from ._polygon import Polygon

if TYPE_CHECKING:
    from ..image import Plane
    from ..mesh import LineMesh
    from ..mesh_container import MeshContainer


def _polygons_to_line_mesh(polygons: List[Polygon],
                           bbox: np.ndarray) -> LineMesh:
    """Generate line-mesh from polygons and surrounding bbox. The polygons are
    stacked and missing corners are obtained from the bounding box coordinates.

    Parameters
    ----------
    polygons : List[Polygon]
        List of polygons.
    bbox : (n, 2) numpy.ndarray
        Coordinates for the bounding box. These define the convex hull
        of the meshing area.

    Returns
    -------
    points : (m,2) numpy.ndarray
        List of points.
    segments : (n,2) numpy.ndarray
        List of segments.
    """
    from nanomesh import LineMesh

    segments = _generate_segments(polygons)

    all_points = np.vstack([polygon.points for polygon in polygons])

    corner_idx = np.argwhere(cdist(bbox, all_points) == 0)

    if len(corner_idx) < len(bbox):
        # Add missing corners and add them where necessary
        missing_corners = np.delete(bbox, corner_idx[:, 0], axis=0)
        all_points = np.vstack([all_points, missing_corners])
        corner_idx = np.argwhere(cdist(bbox, all_points) == 0)

    R = corner_idx[:, 1].tolist()
    additional_segments = list(zip(R, R[1:] + R[:1]))
    segments = np.vstack([segments, additional_segments])

    segment_markers = generate_segment_markers(polygons)
    segment_markers = append_to_segment_markers(segment_markers,
                                                additional_segments)

    mesh = LineMesh(points=all_points,
                    cells=segments,
                    segment_markers=segment_markers)

    return mesh


def _generate_background_region(polygons: List[Polygon],
                                bbox: np.ndarray) -> RegionMarker:
    """Generate marker for background. This point is inside the bbox, but
    outside the given polygons.

    Parameters
    ----------
    polygons : List[Polygon]
        List of polygons.
    bbox : (n, 2) numpy.ndarray
        Coordinates for the bounding box. These define the convex hull
        of the meshing area.

    Returns
    -------
    region : RegionMarker
        Region marker to describe the background feature
    """
    point = bbox.mean(axis=0)

    xmin, ymin = bbox.min(axis=0)
    xmax, ymax = bbox.max(axis=0)

    while any(polygon.contains_point(point) for polygon in polygons):
        point = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)

    return RegionMarker(label=0, point=point, name='background')


def _generate_regions(polygons: List[Polygon],
                      same_label: bool = True) -> RegionMarkerList:
    """Generate regions for triangle.

    Parameters
    ----------
    polygons : List[Polygon]
        List of polygons.
    same_label : bool, optional
        If True, all labels equal 1.
        If False, label regions sequentially from 1

    Returns
    -------
    regions : RegionMarkerList
        List of region markers describing each feature
    """
    regions = RegionMarkerList()

    for i, polygon in enumerate(polygons):
        point = polygon.find_point()

        # in LineMesh format
        if same_label:
            regions.append(RegionMarker(label=1, point=point, name='feature'))
        else:
            label = i + 1
            regions.append(
                RegionMarker(label=label, point=point, name=f'feature{label}'))

    return regions


def _generate_segments(polygons: List[Polygon]) -> np.ndarray:
    """Generate segments for triangle.

    Parameters
    ----------
    polygons : List[Polygon]
        List of polygons

    Returns
    -------
    segments : numpy.ndarray
        Segment connectivity array
    """
    i = 0
    segments = []

    for polygon in polygons:
        n_points = len(polygon)
        rng = np.arange(i, i + n_points)

        # generate segment connectivity matrix
        segment = np.vstack([rng, np.roll(rng, shift=-1)]).T
        segments.append(segment)

        i += n_points

    return np.vstack(segments)


@doc(AbstractMesher, prefix='triangular mesh from 2D image data')
class Mesher2D(AbstractMesher):

    def __init__(self, image: np.ndarray | Plane):
        super().__init__(image)
        self.contour: LineMesh

    def generate_contour(
        self,
        level: float = None,
        contour_precision: int = 1,
        max_contour_dist: int = 5,
    ):
        """Generate contours using marching cubes algorithm.

        Contours are approximated by a polygon, where the maximum distance
        between points is decided by `max_contour_dist`.

        Parameters
        ----------
        level : float, optional
            Contour value to search for isosurfaces (i.e. the threshold value).
            By default takes the average of the min and max value. Can be
            ignored if a binary image is passed to :class:`Mesher2D`.
        contour_precision : int, optional
            Maximum distance from original points in polygon approximation
            routine.
        max_contour_dist : int, optional
            Divide long edges so that maximum distance between points does not
            exceed this value.
        """
        polygons = [
            Polygon(points)
            for points in measure.find_contours(self.image, level=level)
        ]
        polygons = [
            polygon.approximate(contour_precision) for polygon in polygons
        ]
        polygons = [
            polygon.subdivide(max_dist=max_contour_dist)
            for polygon in polygons
        ]
        polygons = [
            polygon.close_corner(self.image.shape) for polygon in polygons
        ]
        polygons = [polygon.remove_duplicate_points() for polygon in polygons]

        regions = _generate_regions(polygons, same_label=True)
        regions.append(_generate_background_region(polygons, self.image_bbox))

        contour = _polygons_to_line_mesh(polygons, self.image_bbox)
        contour.region_markers.extend(regions)

        self.polygons = polygons
        self.contour = contour

    @property
    def image_bbox(self) -> np.ndarray:
        """Return bbox from image shape.

        Returns
        -------
        bbox : (4,2) np.array
            Coordinates of bounding box contour.
        """
        x, y = self.image.shape
        return np.array((
            (0, 0),
            (x - 1, 0),
            (x - 1, y - 1),
            (0, y - 1),
        ))

    def triangulate(self, opts='pAq30a100', **kwargs) -> MeshContainer:
        """Triangulate contours.

        Mandatory switches for `opts`:
            - `e`: ensure edges get returned
            - `p`: planar straight line graph
            - `A`: assign regional attribute to each triangle

        If missing, these will be added.

        Parameters
        ----------
        opts : str, optional
            Options passed to :func:`triangulate`. For more info,
            see: https://rufat.be/triangle/API.html#triangle.triangulate

        Returns
        -------
        mesh : MeshContainer
            Triangulated 2D mesh with domain labels
        """
        for var in 'pAe':
            if var not in opts:
                opts = f'{opts}{var}'
        kwargs['opts'] = opts

        mesh = self.contour.triangulate(**kwargs)

        return mesh

    @doc(pad, prefix='Pad the contour using :func:`image2mesh.mesher2d.pad`')
    def pad_contour(self, **kwargs):
        self.contour = pad(self.contour, **kwargs)

    def plot_contour(self, ax: plt.Axes = None, cmap: str = None, **kwargs):
        """Plot contours on image.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to use for plotting.
        cmap : str
            Matplotlib color map for :func:`matplotlib.pyplot.imshow`
        **kwargs
            These parameters are passed to :func:`plotting.linemeshplot`

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if not ax:
            fig, ax = plt.subplots()

        ax.set_title('Contours')
        self.contour.plot_mpl(ax=ax, **kwargs)

        ax.imshow(self.image, cmap=cmap)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])

        return ax


def plane2mesh(image: np.ndarray | Plane,
               *,
               level: float = None,
               max_contour_dist: int = 5,
               opts: str = 'q30a100',
               plot: bool = False) -> 'MeshContainer':
    """Generate a triangular mesh from a 2D segmented image.

    Parameters
    ----------
    image : (i,j) numpy.ndarray or Plane
        Input image to mesh.
    level : float, optional
        Level to generate contours at from image
    max_contour_dist : int, optional
        Maximum distance between neighbouring pixels in contours.
    opts : str, optional
        Options passed to :func:`triangulate`. For more info,
        see: https://rufat.be/triangle/API.html#triangle.triangulate

    Returns
    -------
    mesh : MeshContainer
        Triangulated 2D mesh with domain labels.
    """
    mesher = Mesher2D(image)
    mesher.generate_contour(max_contour_dist=5, level=level)
    return mesher.triangulate(opts=opts)
