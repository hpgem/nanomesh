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


def test_mesh_container_field_to_number(line_tri_mesh):
    assert line_tri_mesh.field_to_number == {
        'triangle': {
            'Triangle A': 1,
            'Triangle B': 2
        },
        'line': {
            'Line A': 0,
            'Line B': 1
        }
    }
    assert isinstance(line_tri_mesh.field_to_number, MappingProxyType)


def test_mesh_container_number_to_field(line_tri_mesh):
    assert line_tri_mesh.number_to_field == {
        'triangle': {
            1: 'Triangle A',
            2: 'Triangle B'
        },
        'line': {
            0: 'Line A',
            1: 'Line B'
        }
    }
    assert isinstance(line_tri_mesh.number_to_field, MappingProxyType)


def test_mesh_container_cell_types(line_tri_mesh):
    assert set(line_tri_mesh.cell_types) == {'line', 'triangle'}
    assert isinstance(line_tri_mesh.cell_types, tuple)


def test_mesh_container_get_default_types(line_tri_mesh):
    assert line_tri_mesh.get_default_type() == 'triangle'


def test_mesh_container_set_cell_data(line_tri_mesh):
    cell_type = 'triangle'
    new_key = 'new_data'
    new_data = np.array([5, 6])

    assert new_key not in line_tri_mesh.cell_data
    line_tri_mesh.set_cell_data(cell_type, new_key, new_data)

    assert new_key in line_tri_mesh.cell_data
    assert isinstance(line_tri_mesh.cell_data[new_key], list)

    data_dict = line_tri_mesh.cell_data_dict
    np.testing.assert_equal(data_dict[new_key]['triangle'], new_data)
    np.testing.assert_equal(data_dict[new_key]['line'], 0)


def test_write_read(line_tri_mesh, tmp_path):

    def asserts(mesh):
        assert mesh.points.shape[1] == 2
        assert 'physical' in mesh.cell_data
        assert 'gmsh:physical' not in mesh.cell_data
        assert 'gmsh:geometrical' not in mesh.cell_data

    filename = tmp_path / 'out.msh'

    asserts(line_tri_mesh)

    line_tri_mesh.write(filename, file_format='gmsh22')
    asserts(line_tri_mesh)

    new_mesh = MeshContainer.read(filename)
    asserts(new_mesh)

    np.testing.assert_equal(line_tri_mesh.points, new_mesh.points)

    assert len(line_tri_mesh.cells) == len(new_mesh.cells)

    for (left, right) in zip(line_tri_mesh.cells, new_mesh.cells):
        np.testing.assert_equal(left.data, right.data)

    for key, arrays in line_tri_mesh.cell_data.items():
        new_arrays = new_mesh.cell_data[key]

        for left, right in zip(arrays, new_arrays):
            np.testing.assert_equal(left, right)
