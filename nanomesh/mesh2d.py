import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy._lib._util import _asarray_validated
from scipy.cluster.vq import kmeans2
from scipy.spatial import Delaunay
from skimage import morphology
from skimage.measure import approximate_polygon, find_contours


def add_edge_points(image, n_points=20, plot=False):
    """Add points around the edge of the image."""
    shape_x, shape_y = image.shape

    x_grid = np.linspace(0, shape_x - 1, n_points).astype(int)
    y_grid = np.linspace(0, shape_y - 1, n_points).astype(int)[1:-1]

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


def add_speckle_grid(image, border=5, n_points=40):
    """Add dummy points to aid triangulation."""
    disk = morphology.disk(border)
    image_dilated = morphology.binary_erosion(image, selem=disk)

    shape_x, shape_y = image.shape

    x, y = np.meshgrid(np.linspace(0, shape_x, 40),
                       np.linspace(0, shape_y, 40))
    grid_points = np.vstack((np.hstack(x), np.hstack(y))).T.astype(int)

    mask = image_dilated[tuple(grid_points.T)] == True
    grid_points = grid_points[mask]

    return grid_points


def whiten(obs, check_finite=False):
    """Adapted from c:/python27/lib/site-
    packages/skimage/filters/thresholding.py to return array and std_dev."""
    obs = _asarray_validated(obs, check_finite=check_finite)
    std_dev = np.std(obs, axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
        raise RuntimeWarning('Some columns have standard deviation zero. '
                             'The values of these columns will not change.')
    return obs / std_dev, std_dev


def add_speckle_kmeans(image, iters=20, n_points=100, plot=False):
    """Add evenly distributed points to the image."""
    coordinates = np.argwhere(image)

    # kmeans needs normalized data (w),
    # store std to calculate coordinates after
    w, std = whiten(coordinates)

    # nclust must be an integer for some reason
    cluster_centroids, closest_centroids = kmeans2(w,
                                                   n_points,
                                                   iter=iters,
                                                   minit='points')

    # convert to image coordinates
    grid_points = (cluster_centroids * std)

    if plot:
        plt.imshow(image)
        plt.scatter(*grid_points.T[::-1])

    return grid_points


def subdivide_contour(contour, max_dist=10, plot=False):
    """This algorithm looks for long edges in the contour and subdivides them
    so they are no longer than `max_dist`"""
    new_contour = []
    rolled = np.roll(contour, shift=-1, axis=0)
    diffs = rolled - contour
    # ignore last point, do not wrap around
    dist = np.linalg.norm(diffs[:-1], axis=1)

    last_i = 0

    for i in np.argwhere(dist > max_dist).reshape(-1, ):
        new_contour.append(contour[last_i:i])
        start = contour[i]
        stop = rolled[i]
        to_add = int(dist[i] // max_dist) + 1
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

    plt.figure(figsize=(12, 12))

    x, y = points.T[::-1]

    ax = plt.subplot(221)
    ax.set_title(f'Contours ({len(contours)})')
    ax.imshow(image, cmap='gray')
    for contour in contours:
        contour_x, contour_y = contour.T[::-1]
        ax.plot(contour_x, contour_y, color='red')

    ax = plt.subplot(222)
    ax.set_title('Points for triangulation')
    ax.imshow(image, cmap='gray')
    ax.scatter(*points.T[::-1], s=2, color='red')

    ax = plt.subplot(223)
    ax.set_title(f'Mesh ({len(triangles)} triangles)')
    ax.imshow(image, cmap='gray')
    ax.triplot(x, y, triangles)

    ax = plt.subplot(224)
    ax.set_title(f'Filtered mesh ({(mask==0).sum()} triangles)')
    ax.imshow(image, cmap='gray')
    ax.triplot(x, y, triangles=triangles, mask=mask)


def generate_2d_mesh(image: np.ndarray,
                     *,
                     pad: bool = True,
                     speckle: bool = True,
                     plot: bool = False) -> 'meshio.Mesh':
    """Generate mesh from binary (segmented) image.

    Code derived from https://forum.image.sc/t/create-3d-volume-mesh/34052/9

    Parameters
    ----------
    image : 2D np.ndarray
        Input image to mesh.
    pad : bool, optional
        Pad the image using zeros to ensure the contours are closed at the
        image edge.
    plot : bool, optional
        Plot the meshing steps using matplotlib.

    Returns
    -------
    meshio.Mesh
        Description of the mesh.
    """
    points = []

    if speckle:
        # grid_points = add_speckle_grid(image, border=5)
        grid_points = add_speckle_kmeans(image, iters=20, n_points=500)
        points.append(grid_points)

        # adding points to holes helps to get a cleaner result
        grid_points = add_speckle_kmeans(1 - image, iters=20, n_points=100)
        points.append(grid_points)

    if pad:
        pad_points = add_edge_points(image, n_points=20)
        points.append(pad_points)
        # image = np.pad(image, 1, constant_values=0)

    contours = find_contours(image)
    contours = [approximate_polygon(contour, 1) for contour in contours]
    contours = [subdivide_contour(contour, max_dist=5) for contour in contours]

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
