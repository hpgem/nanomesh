from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure

from .._mesh_shared import BaseMesher
from .helpers import pad
from .polygon import Polygon

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..mesh import LineMesh
    from ..mesh_container import MeshContainer


def generate_mesh(polygons: List[Polygon], bbox: np.ndarray) -> LineMesh:
    """Generate line-mesh from polygons and surrounding bbox. The polygons are
    stacked and missing corners are obtained from the bounding box coordinates.

    Parameters
    ----------
    polygons : List[Polygon]
        List of polygons.
    bbox : (n, 2) np.ndarray
        Coordinates for the bounding box. These define the convex hull
        of the meshing area.

    Returns
    -------
    points : (m,2) np.ndarray
        List of points.
    segments : (n,2) np.ndarray
        List of segments.
    """
    from nanomesh import LineMesh

    segments = generate_segments(polygons)

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

    mesh = LineMesh(points=all_points, cells=segments)
    return mesh


def generate_regions(polygons: List[Polygon]) -> List[Tuple[int, np.ndarray]]:
    """Generate regions for triangle.

    Parameters
    ----------
    polygons : List[Polygon]
        List of polygons.

    Returns
    -------
    regions : (n,5) np.ndarray
        Array with regions with each row: (x, y, z, index, 0)
    """
    regions = []

    for label, polygon in enumerate(polygons):
        point = polygon.find_point()

        # in LineMesh format
        regions.append((1, point))

    return regions


def generate_segments(polygons: List[Polygon]) -> np.ndarray:
    """Generate segments for triangle.

    Parameters
    ----------
    polygons : List[Polygon]
        List of polygons

    Returns
    -------
    segments : np.ndarray
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


class Mesher2D(BaseMesher):
    def __init__(self, image: np.ndarray):
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
            ignored if a binary image is passed to `Mesher2D`.
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

        regions = generate_regions(polygons)

        contour = generate_mesh(polygons, self.image_bbox)
        contour.add_region_markers(regions)

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

    def triangulate(self,
                    opts='pAq30a100',
                    clip_line_data: bool = True,
                    **kwargs) -> MeshContainer:
        """Triangulate contours.

        Mandatory switches for `opts`:
            - e: ensure edges get returned
            - p: planar straight line graph
            - A: assign regional attribute to each triangle

        If missing, these will be added.

        Parameters
        ----------
        opts : str, optional
            Options passed to `triangle.triangulate`
        clip_line_data: bool
            If set, clips the line data to 0: body,
            1: external boundary, 2: internal boundary
            instead of individual numbers for each segment
        **kwargs
            Keyword arguments passed to `triangle.triangulate`

        Returns
        -------
        mesh : MeshContainer
            Output 2D mesh with domain labels
        """
        for var in 'pAe':
            if var not in opts:
                opts = f'{opts}{var}'
        kwargs['opts'] = opts

        mesh = self.contour.triangulate(**kwargs)

        mesh.set_field_data('triangle', {0: 'background', 1: 'feature'})
        mesh.set_field_data('line', {0: 'body', 1: 'external', 2: 'internal'})

        fields = {
            m.label: m.name
            for m in self.contour.region_markers if m.name
        }
        mesh.set_field_data('triangle', fields)

        return mesh

    def generate_domain_mask_from_contours(
        self,
        mesh: MeshContainer,
    ) -> np.ndarray:
        """Generate domain mask from contour.

        Parameters
        ----------
        mesh : MeshContainer
            Input mesh

        Returns
        -------
        labels : (n,) np.array
            Array cell labels.
        """
        centers = mesh.get('triangle').cell_centers

        labels = np.zeros(len(centers), dtype=int)

        for polygon in self.polygons:
            mask = polygon.contains_points(centers)
            labels[mask] = 1

        return labels

    def pad_contour(self, **kwargs):
        """Pad the contour. See `.helpers.pad` for info.

        Parameters
        ----------
        **kwargs
            Keyword arguments for `.helpers.pad`.
        """
        self.contour = pad(self.contour, **kwargs)

    def show_contour(self, ax: plt.Axes = None):
        """Plot contours on image.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes to use for plotting.

        Returns
        -------
        ax : matplotlib.Axes
        """
        if not ax:
            fig, ax = plt.subplots()

        ax.set_title('Contours')
        self.contour.plot_mpl(ax=ax)

        ax.imshow(self.image)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])

        return ax


def generate_2d_mesh(image: np.ndarray,
                     *,
                     level: float = None,
                     max_contour_dist: int = 5,
                     opts: str = 'q30a100',
                     plot: bool = False) -> 'meshio.Mesh':
    """Generate mesh from binary (segmented) image.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image to mesh.
    level : float, optional
        Level to generate contours at from image
    max_contour_dist : int, optional
        Maximum distance between neighbouring pixels in contours.
    opts : str, optional
        Options passed to `triangle.triangulate`

    Returns
    -------
    meshio.Mesh
        Description of the mesh.
    """
    mesher = Mesher2D(image)
    mesher.generate_contour(max_contour_dist=5, level=level)
    return mesher.triangulate(opts=opts)
