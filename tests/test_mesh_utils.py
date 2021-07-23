import numpy as np
import pytest

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
    faces = np.zeros((5, 3))
    metadata = {'labels': np.arange(5)}

    return TriangleMesh.create(faces=faces, vertices=vertices, **metadata)


@pytest.fixture
def triangle_mesh_3d():
    vertices = np.arange(15).reshape(5, 3)
    faces = np.zeros((5, 4))
    metadata = {'labels': np.arange(5)}

    return TriangleMesh.create(faces=faces, vertices=vertices, **metadata)


@pytest.fixture
def tetra_mesh():
    vertices = np.arange(15).reshape(5, 3)
    faces = np.zeros((5, 4))
    metadata = {'labels': np.arange(5)}

    return TetraMesh.create(faces=faces, vertices=vertices, **metadata)


def test_meshio_interface(triangle_mesh_2d):
    meshio_mesh = triangle_mesh_2d.to_meshio()
    new_mesh = TriangleMesh.from_meshio(meshio_mesh)

    np.testing.assert_allclose(new_mesh.vertices, triangle_mesh_2d.vertices)
    np.testing.assert_allclose(new_mesh.faces, triangle_mesh_2d.faces)
    np.testing.assert_allclose(new_mesh.metadata['labels'],
                               triangle_mesh_2d.metadata['labels'])
