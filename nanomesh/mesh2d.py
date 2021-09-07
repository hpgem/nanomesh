import logging
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import meshio
import numpy as np
from skimage import measure

from ._mesh_shared import BaseMesher
from .mesh_container import TriangleMesh

logger = logging.getLogger(__name__)


def find_point_in_contour(contour: np.array) -> np.array:
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


def close_corner_contour(contour: np.array, shape: tuple) -> np.array:
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
        extra_point = (ydim, 0)
    elif top and right:
        extra_point = (ydim, xdim)
    elif bottom and right:
        extra_point = (0, xdim)
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
        contours = [
            close_corner_contour(contour, self.image.shape)
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
        mesh : TriangleMesh
            Output 2D mesh with domain labels
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

        mesh = TriangleMesh.from_triangle_dict(triangle_dict_out)
        labels = self.generate_domain_mask_from_contours(mesh, label=label)
        mesh.metadata['labels'] = labels
        return mesh

    def generate_domain_mask_from_contours(self,
                                           mesh: TriangleMesh,
                                           *,
                                           label: int = 1) -> np.array:
        """Generate domain mask from contour.

        Parameters
        ----------
        mesh : TriangleMesh
            Input mesh
        label : int, optional
            Label of the contour set

        Returns
        -------
        labels : (n,) np.array
            Array face labels.
        """
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
