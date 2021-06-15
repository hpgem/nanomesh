from dataclasses import dataclass

import meshio
import numpy as np
import open3d
import pyvista as pv
import trimesh


@dataclass
class TwoDMeshContainer:
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
        """Return instance of `SurfaceMeshContainer` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh'):
        """Return instance of `SurfaceMeshContainer` from open3d."""
        return cls(vertices=mesh.vertices, faces=mesh.faces)


@dataclass
class SurfaceMeshContainer:
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
        """Return instance of `SurfaceMeshContainer` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh'):
        """Return instance of `SurfaceMeshContainer` from open3d."""
        return cls(vertices=mesh.vertices, faces=mesh.faces)


@dataclass
class VolumeMeshContainer:
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
