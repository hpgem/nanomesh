import meshio

from nanomesh.mesh import TetraMesh, TriangleMesh
from nanomesh.mesh_container import MeshContainer


def test_mesh_container_triangle_2d(triangle_mesh_2d):
    mesh = MeshContainer.from_mesh(triangle_mesh_2d)
    assert isinstance(mesh, MeshContainer)
    assert isinstance(mesh.get('triangle'), TriangleMesh)
    assert isinstance(mesh.get(), TriangleMesh)
    assert isinstance(mesh, meshio.Mesh)


def test_mesh_container_triangle_3d(triangle_mesh_2d):
    mesh = MeshContainer.from_mesh(triangle_mesh_2d)
    assert isinstance(mesh, MeshContainer)
    assert isinstance(mesh.get('triangle'), TriangleMesh)
    assert isinstance(mesh.get(), TriangleMesh)
    assert isinstance(mesh, meshio.Mesh)


def test_mesh_container_tetra(tetra_mesh):
    mesh = MeshContainer.from_mesh(tetra_mesh)
    assert isinstance(mesh, MeshContainer)
    assert isinstance(mesh.get('tetra'), TetraMesh)
    assert isinstance(mesh.get(), TetraMesh)
    assert isinstance(mesh, meshio.Mesh)
