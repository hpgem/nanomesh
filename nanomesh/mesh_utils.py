from dataclasses import dataclass

import meshio
import numpy as np
import open3d
import pyvista as pv
import trimesh
from trimesh import remesh


class BaseMeshContainer:
    pass


@dataclass
class TwoDMeshContainer(BaseMeshContainer):
    vertices: np.ndarray
    faces: np.ndarray

    def to_trimesh(self) -> 'trimesh.Trimesh':
        """Return instance of `trimesh.Trimesh`."""
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            ('triangle', self.faces),
        ]

        mesh = meshio.Mesh(self.vertices, cells)
        return mesh

    def to_open3d(self) -> 'open3d.geometry.TriangleMesh':
        """Return instance of `open3d.geometry.TriangleMesh`."""
        import open3d
        return open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            triangles=open3d.utility.Vector3iVector(self.faces))

    @classmethod
    def from_open3d(cls, mesh: 'open3d.geometry.TriangleMesh'):
        """Return instance of `TwoDMeshContainer` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh'):
        """Return instance of `TwoDMeshContainer` from open3d."""
        return cls(vertices=mesh.vertices, faces=mesh.faces)


@dataclass
class SurfaceMeshContainer(BaseMeshContainer):
    vertices: np.ndarray
    faces: np.ndarray

    def to_trimesh(self) -> 'trimesh.Trimesh':
        """Return instance of `trimesh.Trimesh`."""
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            ('triangle', self.faces),
        ]

        mesh = meshio.Mesh(self.vertices, cells)
        return mesh

    def to_open3d(self) -> 'open3d.geometry.TriangleMesh':
        """Return instance of `open3d.geometry.TriangleMesh`."""
        import open3d
        return open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            triangles=open3d.utility.Vector3iVector(self.faces))

    @classmethod
    def from_open3d(
            cls,
            mesh: 'open3d.geometry.TriangleMesh') -> 'SurfaceMeshContainer':
        """Return instance of `SurfaceMeshContainer` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh') -> 'SurfaceMeshContainer':
        """Return instance of `SurfaceMeshContainer` from open3d."""
        return cls(vertices=mesh.vertices, faces=mesh.faces)

    def simplify(self, n_faces: int) -> 'SurfaceMeshContainer':
        """Simplify triangular mesh using `open3d`.

        Parameters
        ----------
        n_faces : int
            Simplify mesh until this number of faces is reached.

        Returns
        -------
        SurfaceMeshContainer
        """
        mesh_o3d = self.to_open3d()
        simplified_o3d = mesh_o3d.simplify_quadric_decimation(int(n_faces))
        return SurfaceMeshContainer.from_open3d(simplified_o3d)

    def simplify_by_vertex_clustering(self,
                                      voxel_size: float = 1.0
                                      ) -> 'SurfaceMeshContainer':
        """Simplify mesh geometry using vertex clustering.

        Parameters
        ----------
        voxel_size : float, optional
            Size of the target voxel within which vertices are grouped.

        Returns
        -------
        SurfaceMeshContainer
        """
        mesh_in = self.to_open3d()
        mesh_smp = mesh_in.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=open3d.geometry.SimplificationContraction.Average)

        return SurfaceMeshContainer.from_open3d(mesh_smp)

    def smooth(self, iterations: int = 50) -> 'SurfaceMeshContainer':
        """Smooth mesh using the Taubin filter in `trimesh`.

        The advantage of the Taubin algorithm is that it avoids
        shrinkage of the object.

        Parameters
        ----------
        iterations : int, optional
            Number of smoothing operations to apply

        Returns
        -------
        SurfaceMeshContainer
        """
        mesh_tri = self.to_trimesh()
        smoothed_tri = trimesh.smoothing.filter_taubin(mesh_tri,
                                                       iterations=iterations)
        return SurfaceMeshContainer.from_trimesh(smoothed_tri)

    def optimize(self,
                 *,
                 method='CVT (block-diagonal)',
                 tol: float = 1.0e-3,
                 max_num_steps: int = 10,
                 **kwargs) -> 'SurfaceMeshContainer':
        """Optimize mesh using `optimesh`.

        Parameters
        ----------
        method : str, optional
            Method name
        tol : float, optional
            Tolerance
        max_num_steps : int, optional
            Maximum number of optimization steps.
        **kwargs
            Arguments to pass to `optimesh.optimize_points_cells`

        Returns
        -------
        SurfaceMeshContainer
        """
        import optimesh
        verts, faces = optimesh.optimize_points_cells(
            points=self.vertices,
            cells=self.faces,
            method=method,
            tol=tol,
            max_num_steps=max_num_steps,
            **kwargs,
        )
        return SurfaceMeshContainer(vertices=verts, faces=faces)

    def subdivide(self,
                  max_edge: int = 10,
                  iters: int = 10) -> 'SurfaceMeshContainer':
        """Subdivide triangles until the maximum edge size is reached.

        Parameters
        ----------
        max_edge : int, optional
            Max triangle edge distance.
        iters : int, optional
            Maximum number of iterations of iterations.
        """
        verts, faces = remesh.subdivide_to_size(self.vertices,
                                                self.faces,
                                                max_edge=max_edge,
                                                max_iter=10)
        return SurfaceMeshContainer(vertices=verts, faces=faces)


@dataclass
class VolumeMeshContainer(BaseMeshContainer):
    vertices: np.ndarray
    faces: np.ndarray

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            ('tetra', self.faces),
        ]

        mesh = meshio.Mesh(self.vertices, cells)
        return mesh

    def to_open3d(self):
        """Return instance of `open3d.geometry.TetraMesh`."""
        import open3d
        return open3d.geometry.TetraMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            tetras=open3d.utility.Vector4iVector(self.faces))

    @classmethod
    def from_open3d(cls, mesh):
        """Return instance of `VolumeMeshContainer` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.tetras)
        return cls(vertices=vertices, faces=faces)


def meshio_to_polydata(mesh):
    """Convert instance of `meshio.Mesh` to `pyvista.PolyData`."""
    return pv.from_meshio(mesh)
