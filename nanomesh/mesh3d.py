from typing import Any

import meshio
import numpy as np
from scipy._lib._util import _asarray_validated
from scipy.cluster.vq import kmeans2
from scipy.spatial import Delaunay
from skimage import morphology
from skimage.measure import approximate_polygon, find_contours


def generate_3d_mesh(image: np.ndarray,
                     *,
                     pad: bool = True,
                     point_density: float = 1 / 100,
                     contour_precision: int = 1,
                     max_contour_dist: int = 10,
                     plot: bool = False) -> 'meshio.Mesh':
    """Generate mesh from binary (segmented) image.

    Code derived from https://forum.image.sc/t/create-3d-volume-mesh/34052/9

    Parameters
    ----------
    image : 3D np.ndarray
        Input image to mesh.
    pad : bool, optional
        Pad the image using zeros to ensure the contours are closed at the
        image edge. The number of points are derived from `points_density`.
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
    points: Any = []

    if point_density:
        # grid_points = add_points_grid(image, border=5)
        n_points1 = int(np.sum(image == 1) * point_density)
        grid_points = add_points_kmeans(image, iters=20, n_points=n_points1)
        points.append(grid_points)

        # adding points to holes helps to get a cleaner result
        n_points0 = int(np.sum(image == 0) * point_density)
        grid_points = add_points_kmeans(1 - image,
                                        iters=20,
                                        n_points=n_points0)
        points.append(grid_points)

    if pad:
        n_points_edge = (np.array(image.shape) *
                         (point_density**0.5)).astype(int)
        pad_points = add_edge_points(image, n_points=n_points_edge)
        points.append(pad_points)
        # image = np.pad(image, 1, constant_values=0)

    contours = find_contours(image)
    contours = [
        approximate_polygon(contour, contour_precision) for contour in contours
    ]
    contours = [
        subdivide_contour(contour, max_dist=max_contour_dist)
        for contour in contours
    ]

    points.extend(contours)
    points = np.vstack(points)

    triangles = Delaunay(points, incremental=False).simplices
    tri_x, tri_y, tri_z = triangles.T
    centers = (points[tri_x] + points[tri_y] + points[tri_z]) / 3
    mask = image[tuple(centers.astype(int).T)] == 0

    if plot:
        plot_mesh_steps(
            image=image,
            contours=contours,
            points=points,
            triangles=triangles,
            mask=mask,
        )

    cells = [
        ('triangle', triangles[~mask]),
    ]

    mesh = meshio.Mesh(points, cells)
    mesh.remove_orphaned_nodes()

    return mesh
