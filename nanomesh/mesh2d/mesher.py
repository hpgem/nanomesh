import logging
from typing import Any, List

import matplotlib.pyplot as plt
import meshio
import numpy as np
from skimage import measure

from nanomesh._mesh_shared import BaseMesher
from nanomesh.mesh_container import TriangleMesh

from .helpers import simple_triangulate

logger = logging.getLogger(__name__)


def _legend_with_triplot_fix(ax: plt.Axes):
    """Add legend for triplot with fix that avoids duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    # reverse to avoid blank line color
    by_label = dict(zip(reversed(labels), reversed(handles)))
    ax.legend(by_label.values(), by_label.keys())


def find_point_in_contour(contour: np.ndarray) -> np.ndarray:
    """Use rejection sampling to find point in contour.

    Parameters
    ----------
    contour : (n,2) np.ndarray
        List of coordinates describing a contour.

    Returns
    -------
    point : np.array
        Coordinate of point in the contour
    """
    # start with guess in center of contour
    point = contour.mean(axis=0)

    while not measure.points_in_poly([point], contour):
        xmin, ymin = contour.min(axis=0)
        xmax, ymax = contour.max(axis=0)
        point = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)

    return point


def generate_regions(contours: List[np.ndarray]) -> np.ndarray:
    """Generate regions for triangle.

    Parameters
    ----------
    contours : List[np.ndarray]
        List of contours.

    Returns
    -------
    regions : (n,5) np.ndarray
        Array with regions with each row: (x, y, z, index, 0)
    """
    regions = []

    for j, contour in enumerate(contours):
        point = find_point_in_contour(contour)

        # in triangle format
        regions.append([*point, j, 0])

    return np.array(regions)


def generate_segments(contours: List[np.ndarray]) -> np.ndarray:
    """Generate segments for triangle.

    Parameters
    ----------
    contours : List[np.ndarray]
        List of contours

    Returns
    -------
    segments : np.ndarray
        Segment connectivity array
    """
    i = 0
    segments = []

    for contour in contours:
        n_points = len(contour)
        rng = np.arange(i, i + n_points)

        # generate segment connectivity matrix
        segment = np.vstack([rng, np.roll(rng, shift=-1)]).T
        segments.append(segment)

        i += n_points

    return np.vstack(segments)


def close_corner_contour(contour: np.ndarray, shape: tuple) -> np.ndarray:
    """Check if contours are in the corner, and close them if needed.

    Contours which cover a corner cannot be closed by joining the first
    and last element, because some of the area is missed. This algorithm
    adds the corner point to close the contours.

    Parameters
    ----------
    contour : (n,2) np.ndarray
        List of coordinates describing a contour.
    shape : tuple
        Shape of the source image. Used to check which corners the
        contour touches.

    Returns
    -------
    contour : (n+1,2) or (n,2) np.array
        Return a contour with a corner point added if needed,
        otherwise return the input contour
    """
    xmin, ymin = contour.min(axis=0)
    xmax, ymax = contour.max(axis=0)

    xdim, ydim = np.array(shape) - 1

    left = (xmin == 0)
    right = (xmax == xdim)
    bottom = (ymin == 0)
    top = (ymax == ydim)

    if bottom and left:
        extra_point = (0, 0)
    elif top and left:
        extra_point = (0, ydim)
    elif top and right:
        extra_point = (xdim, ydim)
    elif bottom and right:
        extra_point = (xdim, 0)
    else:
        # all good
        return contour

    contour = np.vstack([contour, extra_point])
    return contour


def subdivide_contour(contour, max_dist: int = 10, plot: bool = False):
    """This algorithm looks for long edges in the contour and subdivides them
    so they are no longer than `max_dist`

    Parameters
    ----------
    contour : (n,2) np.ndarray
        List of coordinates describing a contour.
    max_dist : int, optional
        Maximum distance between neighbouring coordinates.
    plot : bool, optional
        Show plot of the generated points.

    Returns
    -------
    (m,2) np.ndarray
        Updated coordinate array.
    """
    new_contour: Any = []
    rolled = np.roll(contour, shift=-1, axis=0)
    diffs = rolled - contour
    # ignore last point, do not wrap around
    dist = np.linalg.norm(diffs[:-1], axis=1)

    last_i = 0

    for i in np.argwhere(dist > max_dist).reshape(-1, ):
        new_contour.append(contour[last_i:i])
        start = contour[i]
        stop = rolled[i]
        to_add = int(dist[i] // max_dist)
        new_points = np.linspace(start, stop, to_add, endpoint=False)
        new_contour.append(new_points)

        last_i = i + 1

    new_contour.append(contour[last_i:])
    new_contour = np.vstack(new_contour)

    if plot:
        plt.scatter(*contour.T[::-1], color='red', s=100, marker='x')
        plt.plot(*contour.T[::-1], color='red')
        plt.scatter(*new_contour.T[::-1], color='green', s=100, marker='+')
        plt.plot(*new_contour.T[::-1], color='green')
        plt.axis('equal')
        plt.show()

    return new_contour


class Mesher2D(BaseMesher):
    def __init__(self, image: np.ndarray):
        super().__init__(image)
        self.contours: List[np.ndarray] = []

    def generate_contours(
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
        contours = measure.find_contours(self.image, level=level)
        contours = [
            measure.approximate_polygon(contour, contour_precision)
            for contour in contours
        ]
        contours = [
            subdivide_contour(contour, max_dist=max_contour_dist)
            for contour in contours
        ]
        contours = [
            close_corner_contour(contour, self.image.shape)
            for contour in contours
        ]
        self.contours = contours

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

    def triangulate(self, **kwargs):
        """Triangulate contours.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `triangle.triangulate`

        Returns
        -------
        mesh : TriangleMesh
            Output 2D mesh with domain labels
        """
        bbox = self.image_bbox

        contours = [bbox, *self.contours]

        regions = generate_regions(contours)
        segments = generate_segments(contours)
        points = np.vstack(contours)

        mesh = simple_triangulate(points=points,
                                  segments=segments,
                                  regions=regions,
                                  **kwargs)

        labels = self.generate_domain_mask_from_contours(mesh)
        mesh.labels = labels
        return mesh

    def generate_domain_mask_from_contours(
        self,
        mesh: TriangleMesh,
    ) -> np.ndarray:
        """Generate domain mask from contour.

        Parameters
        ----------
        mesh : TriangleMesh
            Input mesh

        Returns
        -------
        labels : (n,) np.array
            Array cell labels.
        """
        centers = mesh.cell_centers

        labels = np.zeros(len(centers), dtype=int)

        for contour in self.contours:
            mask = measure.points_in_poly(centers, contour)
            labels[mask] = 1

        return labels

    def plot_contour(self, ax: plt.Axes = None):
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
        for contour in self.contours:
            cont_x, cont_y = contour.T
            ax.plot(cont_y, cont_x)

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
    mesher.generate_contours(max_contour_dist=5, level=level)
    return mesher.triangulate(opts=opts)
