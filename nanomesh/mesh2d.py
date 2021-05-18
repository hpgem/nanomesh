import meshio
import numpy as np
from scipy.spatial import Delaunay
from skimage.measure import approximate_polygon, find_contours


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
    ax.set_title('Segmented image')
    ax.imshow(image, cmap='gray')

    ax = plt.subplot(222)
    ax.set_title(f'Contours ({len(contours)})')
    ax.imshow(image, cmap='gray')
    for contour in contours:
        contour_x, contour_y = contour.T[::-1]
        ax.plot(contour_x, contour_y, color='red')

    ax = plt.subplot(223)
    ax.set_title(f'All triangles ({len(triangles)})')
    ax.imshow(image, cmap='gray')
    ax.triplot(x, y, triangles)

    ax = plt.subplot(224)
    ax.set_title(f'Filtered triangles ({mask.sum()})')
    ax.imshow(image, cmap='gray')
    ax.triplot(x, y, triangles=triangles, mask=mask)


def generate_2d_mesh(image: np.ndarray,
                     *,
                     pad: bool = True,
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
    if pad:
        image = np.pad(image, 1, constant_values=0)

    contours = find_contours(image)
    contours = [approximate_polygon(contour, 1) for contour in contours]
    points = np.vstack(contours)
    triangles = Delaunay(points).simplices
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
