from typing import Any, List

import meshio
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import Delaunay
from skimage import measure, transform
from sklearn import cluster

from .mesh_utils import meshio_to_polydata, tetrahedra_to_mesh


def show_submesh(*meshes: List[meshio.Mesh],
                 index: int = 100,
                 along: str = 'x',
                 plotter=pv.PlotterITK):
    """Slow a slice of the mesh.

    Parameters
    ----------
    *meshes : List[meshio.Mesh]
        List of meshes to show
    index : int, optional
        Index of where to cut the mesh.
    along : str, optional
        Direction along which to cut.
    plotter : TYPE, optional
        Plotting instance (`pv.PlotterITK` or `pv.Plotter`)

    Returns
    -------
    plotter : `pyvista.Plotter`
        Instance of the plotter
    """
    plotter = plotter()

    for mesh in meshes:
        grid = meshio_to_polydata(mesh)

        # get cell centroids
        cells = grid.cells.reshape(-1, 5)[:, 1:]
        cell_center = grid.points[cells].mean(1)

        # extract cells below index
        axis = 'zyx'.index(along)

        mask = cell_center[:, axis] < index
        cell_ind = mask.nonzero()[0]
        subgrid = grid.extract_cells(cell_ind)

        plotter.add_mesh(subgrid)

    plotter.show()

    return plotter


def simplify_mesh_trimesh(vertices, faces, n_faces):
    """Simplify mesh using trimesh."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    decimated = mesh.simplify_quadratic_decimation(n_faces)
    return decimated


def simplify_mesh_open3d(vertices: np.ndarray, faces: np.ndarray, n_faces):
    """Simplify mesh using open3d."""
    import open3d
    o3d_verts = open3d.utility.Vector3dVector(vertices)
    o3d_faces = open3d.utility.Vector3iVector(faces)
    o3d_mesh = open3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

    o3d_new_mesh = o3d_mesh.simplify_quadric_decimation(n_faces)

    new_verts = np.array(o3d_new_mesh.vertices)
    new_faces = np.array(o3d_new_mesh.triangles)

    decimated = meshio.Mesh(points=new_verts, cells=[('triangle', new_faces)])
    return decimated


def add_points_kmeans_sklearn(image: np.ndarray,
                              iters: int = 10,
                              n_points: int = 100,
                              scale=1.0,
                              plot: bool = False):
    """Add evenly distributed points to the image.

    Parameters
    ----------
    image : 3D np.ndarray
        Input image
    iters : int, optional
        Number of iterations for the kmeans algorithm.
    scale : float
        Reduce resolution of image to improve performance.
    n_points : int, optional
        Total number of points to add

    Returns
    -------
    (n,3) np.ndarray
        Array with the generated points.
    """
    if scale != 1.0:
        image = transform.rescale(image.astype(float), scale) > 0.5

    coordinates = np.argwhere(image)

    kmeans = cluster.KMeans(n_clusters=n_points,
                            n_init=1,
                            init='random',
                            max_iter=iters,
                            algorithm='full')
    ret = kmeans.fit(coordinates)

    return ret.cluster_centers_ / scale


def generate_3d_mesh(
    image: np.ndarray,
    *,
    step_size: int = 2,
    pad_width: int = 20,
    point_density: float = 1 / 10000,
    res_kmeans: float = 1.0,
    n_faces: int = 1000,
) -> 'meshio.Mesh':
    """Generate mesh from binary (segmented) image.

    Code derived from https://forum.image.sc/t/create-3d-volume-mesh/34052/9

    Parameters
    ----------
    image : 3D np.ndarray
        Input image to mesh.
    pad_width : int, optional
        Number of voxels to pad the images with on each side. Tetrahedra will
        extend beyond the boundary. The image is padded with the edge values
        of the array `mode='edge'` in `np.pad`).
        Pad the image by this number of pixels by reflecting the
    point_density : float, optional
        Density of points to distribute over the domains for triangle
        generation. Expressed as a fraction of the number of pixels.
    step_size : int
        Step size in voxels for the marching cubes algorithms. Larger steps
        yield faster but coarser results.
    res_kmeans : float
        Resolution for the point generation using k-means. Lower resolution of
        the image by this factor. Larger number yiels faster but coarser
        results.
    n_faces : int
        Target number of faces for the mesh decimation step.

    Returns
    -------
    meshio.Mesh
        Description of the mesh.
    """
    points: Any = []

    if pad_width:
        print(f'padding image, {pad_width=}')
        image = np.pad(image, pad_width, mode='reflect')

    if point_density:
        scale = 1 / res_kmeans
        print(f'adding points, {res_kmeans=}->{scale=}')

        # grid_points = add_points_grid(image, border=5)
        n_points1 = int(np.sum(image == 1) * point_density)
        grid_points = add_points_kmeans_sklearn(image,
                                                iters=10,
                                                n_points=n_points1,
                                                scale=scale)
        points.append(grid_points)
        print(f'added {len(grid_points)} points (1)')

        # adding points to holes helps to get a cleaner result
        n_points0 = int(np.sum(image == 0) * point_density)
        grid_points = add_points_kmeans_sklearn(1 - image,
                                                iters=10,
                                                n_points=n_points0,
                                                scale=scale)
        points.append(grid_points)
        print(f'added {len(grid_points)} points (0)')

    print(f'generating vertices, {step_size=}')
    verts, faces, normals, values = measure.marching_cubes(
        image,
        allow_degenerate=False,
        step_size=step_size,
    )
    print(f'generated {len(verts)} verts and {len(faces)} faces')

    print(f'simplifying mesh, {n_faces=}')
    mesh = simplify_mesh_trimesh(vertices=verts, faces=faces, n_faces=n_faces)
    print(f'reduced to {len(mesh.vertices)} verts and {len(mesh.faces)} faces')

    print('smoothing mesh')
    trimesh.smoothing.filter_taubin(mesh, iterations=50)

    points.append(mesh.vertices)
    points = np.vstack(points)

    print('triangulating')
    tetrahedra = Delaunay(points, incremental=False).simplices
    print(f'generated {len(tetrahedra)} tetrahedra')

    centers = points[tetrahedra].mean(1)

    print('masking')
    pore_mask = image[tuple(centers.astype(int).T)] == 1

    masks = [pore_mask]

    if pad_width:
        for i, dim in enumerate(reversed(image.shape)):
            bound_min = pad_width
            bound_max = dim - pad_width
            mask = (centers[:, i] >= bound_min) & (centers[:, i] <= bound_max)
            masks.append(mask)
        # move back to origin
        points -= pad_width

    mask = np.product(masks, axis=0).astype(bool)

    mesh = tetrahedra_to_mesh(points, tetrahedra, mask)
    mesh.remove_orphaned_nodes()

    return mesh
