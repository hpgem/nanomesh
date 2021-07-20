import logging
from typing import Dict, List

import meshio
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import Delaunay
from skimage import measure

from ._mesh_shared import BaseMesher
from .mesh_utils import (SurfaceMeshContainer, VolumeMeshContainer,
                         meshio_to_polydata)

logger = logging.getLogger(__name__)


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
    plotter : pyvista.Plotter, optional
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


class Mesher3D(BaseMesher):
    def __init__(self, image: np.ndarray):
        super().__init__(image)
        self.contours: Dict[int, SurfaceMeshContainer] = {}
        self.pad_width = 0

    def pad(self, pad_width: int, mode: str = 'constant', **kwargs):
        """Pad the image so that the tetrahedra will extend beyond the
        boundary. Uses `np.pad`.

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

    def generate_contour(self, label: int = 1):
        """Generate contours using marching cubes algorithm."""
        padded = np.pad(self.image, 5, mode='edge')
        padded = np.pad(padded, 1, mode='constant', constant_values=label + 1)
        obj = (padded == label).astype(int)

        verts, faces, *_ = measure.marching_cubes(
            obj,
            allow_degenerate=False,
        )
        verts -= np.array([6] * 3)
        mesh = SurfaceMeshContainer(vertices=verts, faces=faces)

        # mesh = mesh.smooth(iterations=50)
        mesh = mesh.simplify(n_faces=10000)
        mesh = mesh.simplify_by_vertex_clustering(voxel_size=2)

        logger.info(
            f'Generated contour with {len(mesh.faces)} faces ({label=})')

        self.contours[label] = mesh

    def generate_surface_mesh(self, step_size: int = 1):
        """Generate surface mesh using marching cubes algorithm.

        Parameters
        ----------
        step_size : int, optional
            Step size in voxels. Larger number means better performance at
            the cost of lowered precision. Equivalent to
            `self.image[::2,::2,::2]` for `step_size==2`.
        """
        logger.info(f'generating vertices, {step_size=}')
        verts, faces, *_ = measure.marching_cubes(
            self.image,
            allow_degenerate=False,
            step_size=step_size,
        )
        mesh = SurfaceMeshContainer(vertices=verts, faces=faces)

        logger.info(f'generated {len(verts)} verts and {len(faces)} faces')

        self.surface_mesh = mesh

    def simplify_mesh(self, n_faces: int):
        """Reduce number of faces in surface mesh to `n_faces`.

        Parameters
        ----------
        n_faces : int
            The mesh is simplified until this number of faces is reached.
        """
        logger.info(f'simplifying mesh, {n_faces=}')

        self.surface_mesh = self.surface_mesh.simplify(n_faces=n_faces)

        logger.info(f'reduced to {len(self.surface_mesh.vertices)} verts '
                    f'and {len(self.surface_mesh.faces)} faces')

    def simplify_mesh_by_vertex_clustering(self, voxel_size: float = 1.0):
        """Simplify mesh geometry using vertex clustering.

        Parameters
        ----------
        voxel_size : float, optional
            Size of the target voxel within which vertices are grouped.
        """
        self.surface_mesh = self.surface_mesh.simplify_by_vertex_clustering(
            voxel_size=voxel_size)

    def smooth_mesh(self):
        """Smooth surface mesh using 'Taubin' algorithm."""
        logger.info('smoothing mesh')
        self.surface_mesh = self.surface_mesh.smooth()

    def optimize_mesh(self, **kwargs):
        """Optimize mesh using `optimesh`.

        Parameters
        ----------
        **kwargs
            Arguments to pass to `optimesh.optimize_points_cells`
        """
        logger.info('optimizing mesh')
        self.surface_mesh = self.surface_mesh.optimize(**kwargs)

    def subdivide_mesh(self, max_edge: int = 10, iters: int = 10):
        """Subdivide triangles until the maximum edge size is reached.

        Parameters
        ----------
        max_edge : int, optional
            Max triangle edge distance.
        iter : int, optional
            Maximum number of iterations of iterations.
        """
        self.surface_mesh = self.surface_mesh.subdivide(max_edge=max_edge,
                                                        iters=iters)

    def generate_volume_mesh(self):
        """Generate volume mesh using Delauny triangulation with vertices from
        surface mesh and k-means point generation."""
        logger.info('triangulating')
        verts = np.vstack([*self.flattened_points, self.surface_mesh.vertices])

        tetrahedra = Delaunay(verts, incremental=False).simplices
        logger.info(f'generated {len(tetrahedra)} tetrahedra')

        self.volume_mesh = VolumeMeshContainer(vertices=verts,
                                               faces=tetrahedra)

    def generate_domain_mask(self, label: int = 1):
        """Generate domain mask.

        Parameters
        ----------
        label : int, optional
            Domain to generate mask for. Not implemented yet.
        """
        vertices = self.volume_mesh.vertices

        centers = vertices[self.volume_mesh.faces].mean(1)

        contour_mesh = self.contours[label].to_trimesh()

        logger.info(f'generating mask, {contour_mesh.is_watertight=}')
        if contour_mesh.is_watertight:
            mask = contour_mesh.contains(centers)
        else:
            mask = self.generate_domain_mask_from_image(centers, label=label)

        self.mask = mask

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

    def to_meshio(self) -> 'meshio.Mesh':
        """Retrieve volume mesh as meshio object."""
        verts = self.volume_mesh.vertices - self.pad_width
        faces = self.volume_mesh.faces[self.mask]
        mesh = VolumeMeshContainer(vertices=verts, faces=faces).to_meshio()
        mesh.remove_orphaned_nodes()
        return mesh


def generate_3d_mesh(
    image: np.ndarray,
    *,
    step_size: int = 2,
    pad_width: int = 2,
    point_density: float = 1 / 10000,
    n_faces: int = 1000,
) -> 'meshio.Mesh':
    """Generate mesh from binary (segmented) image.

    Parameters
    ----------
    image : 3D np.ndarray
        Input image to mesh.
    pad_width : int, optional
        Number of voxels to pad the images with on each side. Tetrahedra will
        extend beyond the boundary. The image is padded with the edge values
        of the array `mode='edge'` in `np.pad`).
    point_density : float, optional
        Density of points to distribute over the domains for triangle
        generation. Expressed as a fraction of the number of voxels.
    step_size : int
        Step size in voxels for the marching cubes algorithms. Larger steps
        yield faster but coarser results.
    n_faces : int
        Target number of faces for the mesh decimation step.

    Returns
    -------
    meshio.Mesh
        Description of the mesh.
    """
    mesher = Mesher3D(image)
    mesher.pad(pad_width=pad_width)
    mesher.add_points(point_density=point_density, label=1)
    mesher.generate_surface_mesh(step_size=step_size)
    mesher.simplify_mesh(n_faces=n_faces)
    mesher.smooth_mesh()
    mesher.generate_volume_mesh()
    mesher.generate_domain_mask()
    mesh = mesher.to_meshio()

    return mesh
