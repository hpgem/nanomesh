import numpy as np
import pytest
import pyvista as pv

from nanomesh.mesh import BaseMesh, TetraMesh, TriangleMesh


@pytest.mark.parametrize('n_points,n_cells,expected', (
    (2, 3, 'triangle'),
    (3, 3, 'triangle'),
    (3, 4, 'tetra'),
))
def test_create(n_points, n_cells, expected):
    points = np.arange(5 * n_points).reshape(5, n_points)
    cells = np.zeros((5, n_cells))

    mesh = BaseMesh.create(points=points, cells=cells)

    assert mesh._cell_type == expected


def test_meshio_interface(triangle_mesh_2d):
    meshio_mesh = triangle_mesh_2d.to_meshio()
    new_mesh = TriangleMesh.from_meshio(meshio_mesh)

    np.testing.assert_allclose(new_mesh.points, triangle_mesh_2d.points)
    np.testing.assert_allclose(new_mesh.cells, triangle_mesh_2d.cells)
    np.testing.assert_allclose(new_mesh.cell_data['labels'],
                               triangle_mesh_2d.cell_data['labels'])


def test_open3d_interface_triangle(triangle_mesh_3d):
    mesh_o3d = triangle_mesh_3d.to_open3d()
    new_mesh = TriangleMesh.from_open3d(mesh_o3d)

    np.testing.assert_allclose(new_mesh.points, triangle_mesh_3d.points)
    np.testing.assert_allclose(new_mesh.cells, triangle_mesh_3d.cells)


def test_open3d_interface_tetra(tetra_mesh):
    mesh_o3d = tetra_mesh.to_open3d()
    new_mesh = TetraMesh.from_open3d(mesh_o3d)

    np.testing.assert_allclose(new_mesh.points, tetra_mesh.points)
    np.testing.assert_allclose(new_mesh.cells, tetra_mesh.cells)


def test_simplify(triangle_mesh_3d):
    n_cells = 10
    new = triangle_mesh_3d.simplify(n_cells=n_cells)
    assert len(new.cells) <= n_cells
    assert isinstance(new, TriangleMesh)


def test_simplify_by_point_clustering(triangle_mesh_3d):
    new = triangle_mesh_3d.simplify_by_point_clustering(voxel_size=1.0)
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
