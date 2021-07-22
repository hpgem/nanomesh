import logging
from collections import defaultdict
from itertools import chain, tee
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import meshio
import numpy as np
from skimage import measure

from ._mesh_shared import BaseMesher
from .mesh_utils import TwoDMeshContainer

logger = logging.getLogger(__name__)


def pairwise_circle(iterable):
    """s -> (s0,s1), (s1,s2), ..., (sn,s0)"""
    a, b = tee(iterable)
    first = next(b, None)
    return zip(a, chain(b, (first, )))


def find_point_in_contour(contour):
    """Use rejection sampling to find point in contour."""
    # start with guess in center of contour
    point = contour.mean(axis=0)

    while not measure.points_in_poly([point], contour):
        xmin, ymin = contour.min(axis=0)
        xmax, ymax = contour.max(axis=0)
        point = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)

    return point


def get_edge_coords(shape: tuple) -> np.ndarray:
    """Get sorted list of edge coordinates around an image of given shape.

    Parameters
    ----------
    shape : tuple
        Shape of the image.

    Returns
    -------
    edge_coords : (n,2) np.ndarray
        Coordinate array going in clockwise orientation from (0, 0)
    """
    shape_x, shape_y = shape

    x_grid = np.arange(0, shape_x, dtype=float)
    y_grid = np.arange(0, shape_y, dtype=float)[1:-1]

    max_y = shape_y - 1
    max_x = shape_x - 1

    ones = np.ones_like
    zeros = np.zeros_like

    top_edge = np.vstack((x_grid, zeros(x_grid))).T
    bottom_edge = np.vstack((x_grid, max_y * ones(x_grid))).T
    left_edge = np.vstack((zeros(y_grid), y_grid)).T
    right_edge = np.vstack((max_x * ones(y_grid), y_grid)).T

    edge_coords = np.vstack((
        top_edge,
        right_edge,
        bottom_edge[::-1],
        left_edge[::-1],
    ))

    return edge_coords


def generate_edge_contours(shape: tuple, contours: list) -> list:
    """Generate edge contours for given shape. The edge contour is split by any
    objects defined by the contours parameter.

    Parameters
    ----------
    shape : tuple
        Shape of the image.
    contours : list
        List of object contours used to split the edge contour.

    Returns
    -------
    edge_contours : list
        List of coordinate arrays with the edge contours.
    """
    edge_coords = get_edge_coords(shape)

    in_contour = []

    for contour in contours:
        index = measure.points_in_poly(edge_coords, contour)
        in_contour.append(index)

    in_contour = np.any(in_contour, axis=0)

    grouped = measure.label(in_contour + 1)

    # generate edge coordinates
    beginnings = np.argwhere(grouped - np.roll(grouped, shift=1))
    ends = np.argwhere(grouped - np.roll(grouped, shift=-1))

    edge_splits = np.hstack((beginnings, ends))

    if len(edge_splits) == 0:
        edge_contours = [edge_coords]
    else:
        edge_contours = []

        for i, j in edge_splits:
            contour = edge_coords[i:j + 1]
            edge_contours.append(contour)

        connect_first_and_last_contour = (in_contour[0] == in_contour[-1])
        if connect_first_and_last_contour:
            logger.info('Connecting first and last contour')
            loop_around_contour = np.vstack(
                (edge_contours.pop(-1), edge_contours.pop(0)))
            edge_contours.append(loop_around_contour)

        logger.info('Updating boundary values')
        for contour_1, contour_2 in pairwise_circle(edge_contours):
            boundary = (contour_1[-1] + contour_2[0]) / 2
            contour_1[-1] = boundary
            contour_2[0] = boundary

    # use low tolerance for floating point errors
    edge_contours = [
        measure.approximate_polygon(contour, tolerance=1e-3)
        for contour in edge_contours
    ]

    return edge_contours


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


def plot_mesh_steps(*, image: np.ndarray, contours: list, points: np.ndarray,
                    triangles: np.ndarray, labels: np.ndarray):
    """Plot meshing steps.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image.
    contours : list of (n,2) np.ndarrays
        Each contour is an ndarray of shape `(n, 2)`,
        consisting of n ``(row, column)`` coordinates along the contour.
    points : (i,2) np.ndarray
        Coordinates of input points
    triangles : (j,3) np.ndarray
        Indices of points forming a triangle.
    labels : (j,) np.ndarray of int
        Array of integers corresponding to triangle labels
    """
    import matplotlib.pyplot as plt

    x, y = points.T[::-1]

    fig, axes = plt.subplots(nrows=3, figsize=(8, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].set_title(f'Contours ({len(contours)})')
    ax[0].imshow(image, cmap='gray')
    for contour in contours:
        contour_x, contour_y = contour.T[::-1]
        ax[0].plot(contour_x, contour_y, color='red')

    ax[1].set_title(f'Vertices ({len(points)} points)')
    ax[1].imshow(image, cmap='gray')
    ax[1].scatter(*points.T[::-1], s=2, color='red')

    ax[2].set_title(f'Labeled mesh ({len(triangles)} triangles)')
    ax[2].imshow(image, cmap='gray')
    for label in (0, 1):
        mask = (labels == label)
        lines, *_ = ax[2].triplot(x, y, triangles=triangles, mask=mask)
        lines.set_label(f'label = {label} ({np.count_nonzero(~mask)})')

    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')


class Mesher2D(BaseMesher):
    def __init__(self, image: np.ndarray):
        super().__init__(image)
        self.contours: Dict[int, List[list]] = defaultdict(list)

    def generate_contours(
        self,
        level: float = None,
        contour_precision: int = 1,
        max_contour_dist: int = 5,
        label: int = 1,
    ):
        """Generate contours using marching cubes algorithm.

        Contours are approximated by a polygon, where the maximum distance
        between points is decided by `max_contour_dist`.

        Parameters
        ----------
        contour_precision : int, optional
            Maximum distance from original points in polygon approximation
            routine.
        max_contour_dist : int, optional
            Divide long edges so that maximum distance between points does not
            exceed this value.
        label : int, optional
            Label to assign to contour.
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
        self.contours[label] = contours

    @property
    def flattened_contours(self) -> list:
        """Return flattened list of contours."""
        flat_list = [
            contour for contour_subset in self.contours.values()
            for contour in contour_subset
        ]
        return flat_list

    @property
    def image_bbox(self) -> np.array:
        """Return bbox from image shape."""
        x, y = self.image.shape
        return np.array((
            (0, 0),
            (x - 1, 0),
            (x - 1, y - 1),
            (0, y - 1),
        ))

    def triangulate(self, label: int = 1, plot: bool = False, **kwargs):
        """Triangulate contours.

        Parameters
        ----------
        label : int, optional
            Label of the contour set
        plot : bool, optional
            If True, plot a comparison of the input/output
        **kwargs
            Keyword arguments passed to `triangle.triangulate`

        Returns
        -------
        TwoDMeshContainer
        """
        import triangle as tr
        bbox = self.image_bbox

        regions = []
        vertices = [bbox, *self.contours[label]]
        segments = []
        i = 0

        for j, contour in enumerate(vertices):

            point = find_point_in_contour(contour)

            # in triangle
            regions.append([*point, j, 0])
            n_points = len(contour)
            rng = np.arange(i, i + n_points)

            # generate segment connectivity matrix
            segment = np.vstack([rng, np.roll(rng, shift=-1)]).T
            segments.append(segment)

            i += n_points

        segments = np.vstack(segments)
        regions = np.array(regions)
        vertices = np.vstack(vertices)

        triangle_dict_in = {
            'vertices': vertices,
            'segments': segments,
            'regions': regions,
        }

        triangle_dict_out = tr.triangulate(triangle_dict_in, **kwargs)

        if plot:
            tr.compare(plt, triangle_dict_in, triangle_dict_out)

        mesh = TwoDMeshContainer.from_triangle_dict(triangle_dict_out)
        labels = self.generate_domain_mask_from_contours(mesh, label=label)
        mesh.labels = labels
        return mesh

    def generate_domain_mask_from_contours(self, mesh, *, label: int = 1):
        """Generate domain mask from contour."""
        centers = mesh.face_centers

        labels = np.zeros(len(centers), dtype=int)

        for contour in self.contours[label]:
            mask = measure.points_in_poly(centers, contour)
            labels[mask] = label

        return labels

    def plot_steps(self):
        """Plot meshing steps."""
        raise NotImplementedError


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
    mesher.generate_contours(label=1, max_contour_dist=5, level=level)
    return mesher.triangulate(label=1, opts=opts)
