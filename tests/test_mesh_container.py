from types import MappingProxyType

import meshio
import numpy as np

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


def test_mesh_container_field_to_number(mesh_square2d):
    assert mesh_square2d.field_to_number == {
        'triangle': {
            'Triangle A': 1,
            'Triangle B': 2
        },
        'line': {
            'Line A': 0,
            'Line B': 1
        }
    }
    assert isinstance(mesh_square2d.field_to_number, MappingProxyType)


def test_mesh_container_number_to_field(mesh_square2d):
    assert mesh_square2d.number_to_field == {
        'triangle': {
            1: 'Triangle A',
            2: 'Triangle B'
        },
        'line': {
            0: 'Line A',
            1: 'Line B'
        }
    }
    assert isinstance(mesh_square2d.number_to_field, MappingProxyType)


def test_mesh_container_cell_types(mesh_square2d):
    assert set(mesh_square2d.cell_types) == {'line', 'triangle'}
    assert isinstance(mesh_square2d.cell_types, tuple)


def test_mesh_container_get_default_types(mesh_square2d):
    assert mesh_square2d.get_default_type() == 'triangle'


def test_mesh_container_set_cell_data(mesh_square2d):
    cell_type = 'triangle'
    new_key = 'new_data'
    new_data = np.array([5, 6])

    assert new_key not in mesh_square2d.cell_data
    mesh_square2d.set_cell_data(cell_type, new_key, new_data)

    assert new_key in mesh_square2d.cell_data
    assert isinstance(mesh_square2d.cell_data[new_key], list)

    data_dict = mesh_square2d.cell_data_dict
    np.testing.assert_equal(data_dict[new_key]['triangle'], new_data)
    np.testing.assert_equal(data_dict[new_key]['line'], 0)


def test_write_read(mesh_square2d, tmp_path):
    def asserts(mesh):
        assert mesh.points.shape[1] == 2
        assert 'physical' in mesh.cell_data
        assert 'gmsh:physical' not in mesh.cell_data
        assert 'gmsh:geometrical' not in mesh.cell_data

    filename = tmp_path / 'out.msh'

    asserts(mesh_square2d)

    mesh_square2d.write(filename, file_format='gmsh22')
    asserts(mesh_square2d)

    new_mesh = MeshContainer.read(filename)
    asserts(new_mesh)

    np.testing.assert_equal(mesh_square2d.points, new_mesh.points)

    assert len(mesh_square2d.cells) == len(new_mesh.cells)

    for (left, right) in zip(mesh_square2d.cells, new_mesh.cells):
        np.testing.assert_equal(left.data, right.data)

    for key, arrays in mesh_square2d.cell_data.items():
        new_arrays = new_mesh.cell_data[key]

        for left, right in zip(arrays, new_arrays):
            np.testing.assert_equal(left, right)
