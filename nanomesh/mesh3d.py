import logging
from dataclasses import dataclass
from typing import List

import meshio
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import Delaunay
from skimage import measure, transform
from sklearn import cluster

from .mesh_utils import (meshio_to_polydata, tetrahedra_to_mesh,
                         triangles_to_mesh)

logger = logging.getLogger(__name__)


@dataclass
class SurfaceMeshContainer:
    vertices: np.ndarray
    faces: np.ndarray

    def to_trimesh(self) -> 'trimesh.Trimesh':
        """Return instance of `trimesh.Trimesh`."""
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        return triangles_to_mesh(self.vertices, self.faces)


@dataclass
class VolumeMeshContainer:
    vertices: np.ndarray
    faces: np.ndarray

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        return tetrahedra_to_mesh(self.vertices, self.faces)


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


def simplify_mesh_trimesh(vertices: np.ndarray, faces: np.ndarray,
                          n_faces: int):
    """Simplify mesh using trimesh."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    decimated = mesh.simplify_quadratic_decimation(n_faces)
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


class Mesher3D:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.points: List[np.ndarray] = []
        self.pad_width = 0

    def pad(self, pad_width: int, mode: str = 'reflect'):
        """Pad the image so that the tetrahedra will extend beyond the
        boundary. Uses `np.pad`.

        Parameters
        ----------
        pad_width : int
            Number of voxels to pad the image with on each side.
        mode : str, optional
            Set the padding mode. For more info see `np.pad`.
        """
        logger.info(f'padding image, {pad_width=}')
        self.image = np.pad(self.image, pad_width, mode=mode)
        self.pad_width = pad_width

    def add_points(self,
                   point_density: float = 1 / 10000,
                   label: int = 1,
                   step_size=2):
        """Generate evenly distributed points using K-Means in the domain body
        for generating tetrahedra.

        Parameters
        ----------
        point_density : float, optional
            Density of points (points per pixels) to distribute over the
            domain for triangle generation.
        label : int, optional
            Label of the domain to add points to.
        step_size : int, optional
            If specified, downsample the image for point generation. This
            speeds up the kmeans algorithm at the cost of lowered precision.
        """
        scale = 1 / step_size
        logger.info(f'adding points, {step_size=}->{scale=}')

        n_points = int(np.sum(self.image == label) * point_density)
        grid_points = add_points_kmeans_sklearn(self.image,
                                                iters=10,
                                                n_points=n_points,
                                                scale=scale)
        self.points.append(grid_points)
        logger.info(f'added {len(grid_points)} points ({label=})')

    def generate_surface_mesh(self, step_size: int = 1):
        """Generate surface mesh using marchin cubes algorithm.

        Parameters
        ----------
        step_size : int, optional
            Step size in voxels. Larger number means better performance at
            the cost of lowered precision.
        """
        logger.info(f'generating vertices, {step_size=}')
        verts, faces, normals, values = measure.marching_cubes(
            self.image,
            allow_degenerate=False,
            step_size=step_size,
        )
        self.surface_mesh = SurfaceMeshContainer(vertices=verts, faces=faces)
        logger.info(f'generated {len(verts)} verts and {len(faces)} faces')

    def simplify_mesh(self, n_faces: int):
        """Reduce number of faces in surface mesh to `n_faces`.

        Parameters
        ----------
        n_faces : int
            The mesh is simplified until this number of faces is reached.
        """
        logger.info(f'simplifying mesh, {n_faces=}')

        mesh = self.surface_mesh.to_trimesh()
        decimated = mesh.simplify_quadratic_decimation(n_faces)
        self.surface_mesh = decimated  # trimesh.Trimesh

        logger.info(f'reduced to {len(self.surface_mesh.vertices)} verts '
                    f'and {len(self.surface_mesh.faces)} faces')

    def smooth_mesh(self):
        """Smooth surface mesh using 'Taubin' algorithm."""
        logger.info('smoothing mesh')

        mesh = trimesh.smoothing.filter_taubin(self.surface_mesh,
                                               iterations=50)
        self.surface_mesh = mesh  # trimesh.Trimesh

    def generate_volume_mesh(self):
        """Generate volume mesh using Delauny triangulation with vertices from
        surface mesh and k-means point generation."""
        logger.info('triangulating')
        verts = np.vstack([*self.points, self.surface_mesh.vertices])
        tetrahedra = Delaunay(verts, incremental=False).simplices
        logger.info(f'generated {len(tetrahedra)} tetrahedra')

        self.volume_mesh = VolumeMeshContainer(vertices=verts,
                                               faces=tetrahedra)

    def generate_domain_mask(self, label: int = 1):
        """Generate domain mask.

        Parameters
        ----------
        label : int, optional
            Domain to generate mask for.
        """
        # align points with voxel centers, and remove pad_width
        logger.info('generating mask')
        points_shifted = self.volume_mesh.vertices + 0.5

        centers = points_shifted[self.volume_mesh.faces].mean(1)

        pore_mask_center = self.image[tuple(centers.astype(int).T)] == 1

        index = np.clip(points_shifted,
                        a_min=0,
                        a_max=np.array(self.image.shape) - 1).astype(int)
        selection = self.image[tuple(index.T)] == label
        selection = np.argwhere(selection)
        pore_mask_vert = np.any(np.isin(self.volume_mesh.faces, selection),
                                axis=1)

        masks = [pore_mask_vert, pore_mask_center]

        if self.pad_width:
            for i, dim in enumerate(reversed(self.image.shape)):
                bound_min = self.pad_width
                bound_max = dim - self.pad_width
                mask = (centers[:, i] >= bound_min) & (centers[:, i] <=
                                                       bound_max)
                masks.append(mask)
            # move back to origin

        mask = np.product(masks, axis=0).astype(bool)

        self.mask = mask

    def to_meshio(self) -> 'meshio.Mesh':
        """Retrieve volume mesh as meshio object."""
        verts = self.volume_mesh.vertices + 0.5 - self.pad_width
        faces = self.volume_mesh.faces
        mesh = tetrahedra_to_mesh(verts, faces, self.mask)
        mesh.remove_orphaned_nodes()
        return mesh


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
    mesher = Mesher3D(image)
    mesher.pad(pad_width=pad_width)
    mesher.add_points(point_density=point_density,
                      step_size=res_kmeans,
                      label=1)
    mesher.add_points(point_density=point_density / 10,
                      step_size=res_kmeans,
                      label=0)
    mesher.generate_surface_mesh(step_size=step_size)
    mesher.simplify_mesh(n_faces=n_faces)
    mesher.smooth_mesh()
    mesher.generate_volume_mesh()
    mesher.generate_domain_mask()
    mesh = mesher.to_meshio()

    return mesh
