import logging
from typing import Any, List

import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.spatial import Delaunay
from skimage.measure import approximate_polygon, find_contours
from sklearn import cluster

from .mesh_utils import TwoDMeshContainer

logger = logging.getLogger(__name__)


def add_edge_points(image: np.ndarray, n_points: tuple = (10, 10), plot=False):
    """Add points around the edge of the image.

    The points are masked by the image.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image.
    n_points : tuple, optional
        Number of points to add along the edges in the x and y direction.
    plot : bool, optional
        Show plot of the added points.

    Returns
    -------
    (n,2) np.ndarray
        Array with the generated points.
    """
    shape_x, shape_y = image.shape
    n_points_x, n_points_y = n_points

    x_grid = np.linspace(0, shape_x - 1, n_points_x).astype(int)
    y_grid = np.linspace(0, shape_y - 1, n_points_y).astype(int)[1:-1]

    top_edge = np.vstack((x_grid, np.zeros_like(x_grid))).T
    bottom_edge = np.vstack((x_grid, (shape_y - 1) * np.ones_like(x_grid))).T
    left_edge = np.vstack((np.zeros_like(y_grid), y_grid)).T
    right_edge = np.vstack(((shape_x - 1) * np.ones_like(y_grid), y_grid)).T

    edge_points = np.vstack((top_edge, bottom_edge, left_edge, right_edge))
    mask = image[tuple(edge_points.T)] == True
    edge_points = edge_points[mask]

    if plot:
        plt.imshow(image)
        plt.scatter(*edge_points.T[::-1])

    return edge_points


def add_points_kmeans(image: np.ndarray,
                      iters: int = 10,
                      n_points: int = 100,
                      label: int = 1,
                      plot: bool = False):
    """Add evenly distributed points to the image.

    Parameters
    ----------
    image : 3D np.ndarray
        Input image
    iters : int, optional
        Number of iterations for the kmeans algorithm.
    n_points : int, optional
        Total number of points to add

    Returns
    -------
    (n,3) np.ndarray
        Array with the generated points.
    """
    coordinates = np.argwhere(image == label)

    kmeans = cluster.KMeans(n_clusters=n_points,
                            n_init=1,
                            init='random',
                            max_iter=iters,
                            algorithm='full')
    ret = kmeans.fit(coordinates)

    return ret.cluster_centers_


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
                    triangles: np.ndarray, mask: np.ndarray):
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
    mask : (j,) np.ndarray of booleans
        Array of booleans for which triangles are masked out
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))

    x, y = points.T[::-1]

    ax = plt.subplot(221)
    ax.set_title(f'Contours ({len(contours)})')
    ax.imshow(image, cmap='gray')
    for contour in contours:
        contour_x, contour_y = contour.T[::-1]
        ax.plot(contour_x, contour_y, color='red')

    ax = plt.subplot(222)
    ax.set_title(f'Vertices ({len(points)} points)')
    ax.imshow(image, cmap='gray')
    ax.scatter(*points.T[::-1], s=2, color='red')

    ax = plt.subplot(223)
    ax.set_title(f'Inside mesh ({(mask==0).sum()} triangles)')
    ax.imshow(image, cmap='gray')
    ax.triplot(x, y, triangles=triangles, mask=mask)

    ax = plt.subplot(224)
    ax.set_title(f'Outside mesh ({(mask==1).sum()} triangles)')
    ax.imshow(image, cmap='gray')
    ax.triplot(x, y, triangles, mask=~mask)


class Mesher2D:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.points: List[np.ndarray] = []
        self.pad_width = 0
        self.mask = None

    def pad(self, pad_width: int, mode: str = 'constant', **kwargs):
        """Pad the image so that the triangles will extend beyond the boundary.
        Uses `np.pad`.

        Parameters
        ----------
        pad_width : int
            Number of voxels to pad the image with on each side.
        mode : str, optional
            Set the padding mode. For more info see `np.pad`.
        **kwargs :
            Keyword arguments passed to `np.pad`
        """
        logger.info(f'padding image, {pad_width=}')
        self.image = np.pad(self.image, pad_width, mode=mode, **kwargs)
        self.pad_width = pad_width

    def add_points(
        self,
        point_density: float = 1 / 100,
        label: int = 1,
    ):
        """Generate evenly distributed points using K-Means in the domain body
        for generating tetrahedra.

        Parameters
        ----------
        point_density : float, optional
            Density of points (points per pixels) to distribute over the
            domain for triangle generation.
        label : int, optional
            Label of the domain to add points to.
        """
        n_points = int(np.sum(self.image == label) * point_density)
        grid_points = add_points_kmeans(self.image,
                                        iters=10,
                                        n_points=n_points,
                                        label=1)
        self.points.append(grid_points)
        logger.info(f'added {len(grid_points)} points ({label=})')

    def generate_contours(
        self,
        contour_precision: int = 1,
        max_contour_dist: int = 10,
    ):
        """Generate surface mesh using marching cubes algorithm.

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
        """

        contours = find_contours(self.image)
        contours = [
            approximate_polygon(contour, contour_precision)
            for contour in contours
        ]
        contours = [
            subdivide_contour(contour, max_dist=max_contour_dist)
            for contour in contours
        ]
        self.contours = contours

    def generate_mesh(self):
        """Generate 2D triangle mesh using Delauny triangulation with vertices
        from contours and k-means point generation."""
        verts = np.vstack([*self.points, *self.contours])

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

        # cannot use `trimesh.Trimesh.contains` which relies on watertight meshes
        # 2d meshes are per definition not watertight
        if self.contours:
            mask = self.generate_domain_mask_from_contours(centers,
                                                           label=label)
        else:
            mask = self.generate_domain_mask_from_image(centers, label=label)

        self.mask = mask

    def generate_domain_mask_from_contours(self, centers, *, label):
        """Alternative implementation to generate a domain mask for surface
        meshes that are not closed, i.e. not watertight.

        Returns
        -------
        mask : (n,1) np.ndarray
            1-dimensional mask for the faces
        """
        from skimage.measure import points_in_poly
        masks = []

        for contour in self.contours:
            mask = points_in_poly(centers, contour)
            masks.append(~mask)

        if self.pad_width:
            # if padded, invert mask for first contour to avoid
            # masking entire image
            masks[0] = ~masks[0]

        mask = np.product(masks, axis=0).astype(bool)

        return mask

    def generate_domain_mask_from_image(self, centers, *, label):
        """Alternative implementation to generate a domain mask for surface
        meshes that are not closed, i.e. not watertight.

        Returns
        -------
        mask : (n,1) np.ndarray
            1-dimensional mask for the faces
        """

        pore_mask_center = self.image[tuple(
            np.round(centers).astype(int).T)] == label

        masks = [pore_mask_center]

        if self.pad_width:
            for i, dim in enumerate(reversed(self.image.shape)):
                bound_min = self.pad_width
                bound_max = dim - self.pad_width
                mask = (centers[:, i] > bound_min) & (centers[:, i] <
                                                      bound_max)
                masks.append(mask)

        mask = np.product(masks, axis=0).astype(bool)

        return mask

    def plot_steps(self):
        """Plot meshing steps."""
        plot_mesh_steps(
            image=self.image,
            contours=self.contours,
            points=self.surface_mesh.vertices,
            triangles=self.surface_mesh.faces,
            mask=self.mask,
        )

    def to_meshio(self) -> 'meshio.Mesh':
        """Retrieve volume mesh as `meshio.Mesh` object."""
        verts = self.surface_mesh.vertices - self.pad_width
        faces = self.surface_mesh.faces[self.mask]
        mesh = TwoDMeshContainer(vertices=verts, faces=faces).to_meshio()
        mesh.remove_orphaned_nodes()
        return mesh


def generate_2d_mesh(image: np.ndarray,
                     *,
                     pad_width: int = 1,
                     point_density: float = 1 / 100,
                     contour_precision: int = 1,
                     max_contour_dist: int = 10,
                     plot: bool = False) -> 'meshio.Mesh':
    """Generate mesh from binary (segmented) image.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image to mesh.
    pad_width : int, optional
        Number of pixel to pad the images with on each side. Triangles will
        be generated to the boundary.
    point_density : float, optional
        Density of points to distribute over the domains for triangle
        generation. Expressed as a fraction of the number of pixels.
    contour_precision : int, optional
        Maximum distance from original contour to approximate polygon.
    max_contour_dist : int, optional
        Maximum distance between neighbouring pixels in contours.
    plot : bool, optional
        Plot the meshing steps using matplotlib.

    Returns
    -------
    meshio.Mesh
        Description of the mesh.
    """
    mesher = Mesher2D(image)

    mesher.pad(pad_width)
    if point_density > 0:
        mesher.add_points(label=1, point_density=point_density)
    mesher.generate_contours(contour_precision=contour_precision,
                             max_contour_dist=max_contour_dist)
    mesher.generate_mesh()
    mesher.generate_domain_mask()

    if plot:
        mesher.plot_steps()

    mesh = mesher.to_meshio()
    return mesh
