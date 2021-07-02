import logging
from collections import defaultdict
from itertools import chain, tee
from typing import Any, Dict

import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.spatial import Delaunay
from skimage import measure
from sklearn import cluster, mixture

from .mesh_utils import TwoDMeshContainer

logger = logging.getLogger(__name__)


def pairwise_circle(iterable):
    """s -> (s0,s1), (s1,s2), ..., (sn,s0)"""
    a, b = tee(iterable)
    first = next(b, None)
    return zip(a, chain(b, (first, )))


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


def add_points_kmeans(image: np.ndarray,
                      iters: int = 10,
                      n_points: int = 100,
                      label: int = 1,
                      **kwargs):
    """Add evenly distributed points to the image using K-Means clustering.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image
    iters : int, optional
        Number of iterations for the algorithm.
    n_points : int, optional
        Total number of points to add
    label : int, optional
        Domain to select coordinates from
    **kwargs
        Extra keyword arguments to pass to `sklearn.cluster.KMeans`

    Returns
    -------
    (n,2) np.ndarray
        Array with the generated points.
    """
    coordinates = np.argwhere(image == label)

    kmeans = cluster.KMeans(n_clusters=n_points,
                            n_init=1,
                            init='random',
                            max_iter=iters,
                            algorithm='full',
                            **kwargs)
    ret = kmeans.fit(coordinates)

    return ret.cluster_centers_


def add_points_gaussian_mixture(image: np.ndarray,
                                iters: int = 10,
                                n_points: int = 100,
                                label: int = 1,
                                **kwargs):
    """Add evenly distributed points to the image using a Gaussian Mixture
    model.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image
    iters : int, optional
        Number of iterations for the algorithm.
    n_points : int, optional
        Total number of points to add
    label : int, optional
        Domain to select coordinates from
    **kwargs
        Extra keyword arguments to pass to `sklearn.mixture.GaussianMixture`

    Returns
    -------
    (n,2) np.ndarray
        Array with the generated points.
    """
    coordinates = np.argwhere(image == label)

    gmm = mixture.GaussianMixture(n_components=n_points, max_iter=10, **kwargs)
    ret = gmm.fit(coordinates)

    return ret.means_


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


class Mesher2D:
    def __init__(self, image: np.ndarray):
        self.image_orig = image
        self.image = image
        self.points: Dict[int, list] = defaultdict(list)
        self.contours: Dict[int, list] = defaultdict(list)

    def add_points(
        self,
        point_density: float = 1 / 100,
        label: int = 1,
        method: str = 'kmeans',
        **kwargs,
    ):
        """Generate evenly distributed points using K-Means in the domain body
        for generating tetrahedra. Alternative implementation using a Gaussian
        Mixture model available via `method='gmm'`.

        Parameters
        ----------
        point_density : float, optional
            Density of points (points per pixels) to distribute over the
            domain for triangle generation.
        label : int, optional
            Label of the domain to add points to.
        method : str
            Clustering algorithm to use,
                `kmeans` : `sklearn.cluster.KMeans`
                `gmm` : `sklearn.mixture.GaussianMixture`
        **kwargs :
            Keywords arguments for the clustering algorithm.
        """
        n_points = int(np.sum(self.image == label) * point_density)

        if method == 'kmeans':
            add_points_func = add_points_kmeans
        elif method == 'gmm':
            add_points_func = add_points_gaussian_mixture
        else:
            raise ValueError(f'Unknown method: {method!r}')
        grid_points = add_points_func(self.image,
                                      iters=10,
                                      n_points=n_points,
                                      label=label,
                                      **kwargs)
        self.points[label].append(grid_points)
        logger.info(f'added {len(grid_points)} points ({label=}), '
                    f'{point_density=}, {method=}')

    def generate_contours(
        self,
        contour_precision: int = 1,
        max_contour_dist: int = 10,
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
        label : int
            Label to assign to contour.
        """

        contours = measure.find_contours(self.image)
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
    def flattened_points(self) -> list:
        """Return flattened list of pointss."""
        flat_list = [
            points for points_subset in self.points.values()
            for points in points_subset
        ]
        return flat_list

    def generate_edge_contours(self, max_contour_dist: int = 10):
        """Generate contours around the edge of the image.

        If the edge contour is intersected by an existing contour
        (`self.contours`), the contour is split at that point.

        Contours are given as a polygon, where the maximum distance
        between points is decided by `max_contour_dist`.

        Parameters
        ----------
        max_contour_dist : int, optional
            Divide long edges so that maximum distance between points does not
            exceed this value.
        """
        contours = generate_edge_contours(self.image.shape,
                                          self.flattened_contours)
        contours = [
            subdivide_contour(contour, max_dist=max_contour_dist)
            for contour in contours
        ]
        self.edge_contours = contours

    def generate_mesh(self):
        """Generate 2D triangle mesh using Delauny triangulation with vertices
        from contours and k-means point generation."""
        verts = np.vstack([
            *self.flattened_points, *self.flattened_contours,
            *self.edge_contours
        ])

        # TODO: merge close vertices

        faces = Delaunay(verts, incremental=False).simplices

        self.surface_mesh = TwoDMeshContainer(vertices=verts, faces=faces)

    def generate_domain_mask(self, label: int = 1):
        """Generate domain mask.

        Parameters
        ----------
        label : int, optional
            Domain to generate mask for. Not implemented yet.
        """
        logger.info('generating mask')
        vertices = self.surface_mesh.vertices

        centers = vertices[self.surface_mesh.faces].mean(1)

        # cannot use `trimesh.Trimesh.contains` which relies on watertight
        # meshes, 2d meshes are per definition not watertight
        labels = self.generate_domain_mask_from_contours(centers, label=label)

        self.labels = labels

    def generate_domain_mask_from_contours(self, centers, *, label):
        """Alternative implementation to generate a domain mask for surface
        meshes that are not closed, i.e. not watertight.

        Returns
        -------
        mask : (n,1) np.ndarray
            1-dimensional mask for the faces
        """
        labels = np.zeros(len(centers), dtype=int)

        for label, contours in self.contours.items():
            for contour in contours:
                mask = measure.points_in_poly(centers, contour)
                labels[mask] = label

        return labels

    def plot_steps(self):
        """Plot meshing steps."""
        plot_mesh_steps(
            image=self.image,
            contours=self.flattened_contours,
            points=self.surface_mesh.vertices,
            triangles=self.surface_mesh.faces,
            labels=self.labels,
        )

    def to_meshio(self, label: int = None) -> 'meshio.Mesh':
        """Retrieve volume mesh as `meshio.Mesh` object."""
        verts = self.surface_mesh.vertices
        faces = self.surface_mesh.faces
        labels = self.labels

        if label is not None:
            mask = (labels != label)
            faces = faces[mask]
            labels = labels[mask]

        mesh = TwoDMeshContainer(vertices=verts, faces=faces).to_meshio()
        mesh.remove_orphaned_nodes()

        mesh.cell_data['labels'] = [labels]

        return mesh


def generate_2d_mesh(image: np.ndarray,
                     *,
                     point_density: float = 1 / 100,
                     contour_precision: int = 1,
                     max_contour_dist: int = 5,
                     max_edge_contour_dist: int = 10,
                     plot: bool = False) -> 'meshio.Mesh':
    """Generate mesh from binary (segmented) image.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image to mesh.
    point_density : float, optional
        Density of points to distribute over the domains for triangle
        generation. Expressed as a fraction of the number of pixels.
    contour_precision : int, optional
        Maximum distance from original contour to approximate polygon.
    max_contour_dist : int, optional
        Maximum distance between neighbouring pixels in contours.
    max_edge_contour_dist : int, optional
        Maximum distance between neighbouring pixels in edge contours.
    plot : bool, optional
        Plot the meshing steps using matplotlib.

    Returns
    -------
    meshio.Mesh
        Description of the mesh.
    """
    mesher = Mesher2D(image)

    if point_density > 0:
        mesher.add_points(label=1,
                          point_density=point_density,
                          method='kmeans')
        mesher.add_points(label=0, point_density=point_density, method='gmm')
    mesher.generate_contours(contour_precision=contour_precision,
                             max_contour_dist=max_contour_dist)
    mesher.generate_edge_contours(max_contour_dist=max_edge_contour_dist)
    mesher.generate_mesh()
    mesher.generate_domain_mask()

    if plot:
        mesher.plot_steps()

    mesh = mesher.to_meshio()
    return mesh
