import numpy as np
import pytest
import pyvista as pv

from nanomesh.mesh_utils import MeshContainer, TetraMesh, TriangleMesh


@pytest.mark.parametrize('n_points,n_faces,expected', (
    (2, 3, 'triangle'),
    (3, 3, 'triangle'),
    (3, 4, 'tetra'),
))
def test_create(n_points, n_faces, expected):
    vertices = np.arange(5 * n_points).reshape(5, n_points)
    faces = np.zeros((5, n_faces))

    mesh = MeshContainer.create(vertices=vertices, faces=faces)

    assert mesh._element_type == expected


@pytest.fixture
def triangle_mesh_2d():
    vertices = np.arange(10).reshape(5, 2)
    faces = np.zeros((5, 3), dtype=int)
    metadata = {'labels': np.arange(5)}

    mesh = TriangleMesh.create(faces=faces, vertices=vertices, **metadata)
    assert isinstance(mesh, TriangleMesh)
    return mesh


@pytest.fixture
def triangle_mesh_3d():
    vertices = np.arange(15).reshape(5, 3)
    faces = np.zeros((5, 3), dtype=int)
    metadata = {'labels': np.arange(5)}

    mesh = TriangleMesh.create(faces=faces, vertices=vertices, **metadata)
    assert isinstance(mesh, TriangleMesh)
    return mesh


@pytest.fixture
def tetra_mesh():
    vertices = np.arange(15).reshape(5, 3)
    faces = np.zeros((5, 4), dtype=int)
    metadata = {'labels': np.arange(5)}

    mesh = TetraMesh.create(faces=faces, vertices=vertices, **metadata)
    assert isinstance(mesh, TetraMesh)
    return mesh


def test_meshio_interface(triangle_mesh_2d):
    meshio_mesh = triangle_mesh_2d.to_meshio()
    new_mesh = TriangleMesh.from_meshio(meshio_mesh)

    np.testing.assert_allclose(new_mesh.vertices, triangle_mesh_2d.vertices)
    np.testing.assert_allclose(new_mesh.faces, triangle_mesh_2d.faces)
    np.testing.assert_allclose(new_mesh.metadata['labels'],
                               triangle_mesh_2d.metadata['labels'])


def test_open3d_interface_triangle(triangle_mesh_3d):
    mesh_o3d = triangle_mesh_3d.to_open3d()
    new_mesh = TriangleMesh.from_open3d(mesh_o3d)

    np.testing.assert_allclose(new_mesh.vertices, triangle_mesh_3d.vertices)
    np.testing.assert_allclose(new_mesh.faces, triangle_mesh_3d.faces)


def test_open3d_interface_tetra(tetra_mesh):
    mesh_o3d = tetra_mesh.to_open3d()
    new_mesh = TetraMesh.from_open3d(mesh_o3d)

    np.testing.assert_allclose(new_mesh.vertices, tetra_mesh.vertices)
    np.testing.assert_allclose(new_mesh.faces, tetra_mesh.faces)


def test_simplify(triangle_mesh_3d):
    n_faces = 10
    new = triangle_mesh_3d.simplify(n_faces=n_faces)
    assert len(new.faces) <= n_faces
    assert isinstance(new, TriangleMesh)


def test_simplify_by_vertex_clustering(triangle_mesh_3d):
    new = triangle_mesh_3d.simplify_by_vertex_clustering(voxel_size=1.0)
    assert isinstance(new, TriangleMesh)


def test_smooth(triangle_mesh_3d):
    new = triangle_mesh_3d.smooth(iterations=1)
    assert isinstance(new, TriangleMesh)


def test_subdivide(triangle_mesh_3d):
    new = triangle_mesh_3d.subdivide()
    assert isinstance(new, TriangleMesh)


def test_plot_submesh(tetra_mesh):
    plotter = tetra_mesh.plot_submesh(show=False)
    assert isinstance(plotter, pv.Plotter)
    plotter.close()
