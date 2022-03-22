from __future__ import annotations

from typing import TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure

from nanomesh._constants import BACKGROUND, FEATURE
from nanomesh._doc import doc
from nanomesh.region_markers import RegionMarker, RegionMarkerList

from .._mesher import Mesher
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

    fields = {}
    for i in np.unique(segment_markers):
        fields[f'L{i}'] = i

    segment_markers = append_to_segment_markers(segment_markers,
                                                additional_segments)

    mesh = LineMesh(points=all_points,
                    cells=segments,
                    segment_markers=segment_markers,
                    fields=fields)

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

    return RegionMarker(label=BACKGROUND, point=point, name='background')


def _generate_regions(polygons: List[Polygon]) -> RegionMarkerList:
    """Generate regions for triangle.

    Parameters
    ----------
    polygons : List[Polygon]
        List of polygons.

    Returns
    -------
    regions : RegionMarkerList
        List of region markers describing each feature
    """
    regions = RegionMarkerList()

    for i, polygon in enumerate(polygons):
        point = polygon.find_point()

        regions.append(RegionMarker(label=FEATURE, point=point, name='X'))

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


@doc(Mesher, prefix='triangular mesh from 2D image data')
class Mesher2D(Mesher, ndim=2):

    def __init__(self, image: np.ndarray | Plane):
        super().__init__(image)
        self.contour: LineMesh
        self._bbox = None

    def generate_contour(
        self,
        level: float = None,
        precision: int = 1,
        max_edge_dist: int = 5,
        group_regions: bool = True,
    ):
        """Generate contours using marching cubes algorithm.

        Contours are approximated by a polygon, where the maximum distance
        between points is decided by `max_edge_dist`.

        Parameters
        ----------
        level : float, optional
            Contour value to search for isosurfaces (i.e. the threshold value).
            By default takes the average of the min and max value. Can be
            ignored if a binary image is passed to :class:`Mesher2D`.
        precision : int, optional
            Maximum distance from original points in polygon approximation
            routine.
        max_edge_dist : int, optional
            Divide long edges so that maximum distance between points does not
            exceed this value.
        group_regions : bool, optional
            If True, assign the same label to all features
            If False, label regions sequentially
        """
        polygons = [
            Polygon(points)
            for points in measure.find_contours(self.image, level=level)
        ]
        polygons = [polygon.approximate(precision) for polygon in polygons]
        polygons = [
            polygon.subdivide(max_dist=max_edge_dist) for polygon in polygons
        ]
        polygons = [
            polygon.close_corner(self.image.shape) for polygon in polygons
        ]
        polygons = [polygon.remove_duplicate_points() for polygon in polygons]

        regions = _generate_regions(polygons)
        regions.append(_generate_background_region(polygons, self.bbox))

        if not group_regions:
            regions = regions.label_sequentially(FEATURE, fmt_name='X{}')

        contour = _polygons_to_line_mesh(polygons, self.bbox)
        contour.region_markers = regions

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

    @property
    def bbox(self) -> np.ndarray:
        """Return bbox attribute.

        If not explicity set, returns :attr:`Mesher2D.image_bbox`.

        Sequence:
            x0, y0
            x1, y0
            x1, y1
            x0, y0

        Returns
        -------
        bbox : np.ndarray
            Bounding box set for output mesh.
        """
        if self._bbox is None:
            return self.image_bbox
        else:
            return self._bbox

    @bbox.setter
    def bbox(self, bbox: np.ndarray):
        """Set bounding box attribute.

        Parameters
        ----------
        bbox : np.ndarray
            List of coordinates for bounding box corners:
                x0, y0
                x1, y0
                x1, y1
                x0, y0

        Raises
        ------
        ValueError
            Raised if `bbox` has the wrong shape.
        """
        bbox = np.array(bbox)
        if bbox.shape != (4, 2):
            raise ValueError('Bounding box must be an array with shape (4,2).')
        self._bbox = bbox

    def triangulate(self, opts='pAq30a10', **kwargs) -> MeshContainer:
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
        default_opts = {'p': True, 'A': True, 'e': True}
        mesh = self.contour.triangulate(opts=opts,
                                        default_opts=default_opts,
                                        **kwargs)

        return mesh

    @doc(pad, prefix='Pad the contour using :func:`image2mesh._mesher2d.pad`')
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
               max_edge_dist: int = 5,
               opts: str = 'q30a10',
               plot: bool = False) -> 'MeshContainer':
    """Generate a triangular mesh from a 2D segmented image.

    Parameters
    ----------
    image : (i,j) numpy.ndarray or Plane
        Input image to mesh.
    level : float, optional
        Level to generate contours at from image
    max_edge_dist : int, optional
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
    mesher.generate_contour(max_edge_dist=5, level=level)
    return mesher.triangulate(opts=opts)
