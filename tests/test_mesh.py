import numpy as np
import pytest
import pyvista as pv

from nanomesh.mesh import BaseMesh, TriangleMesh


@pytest.mark.parametrize('n_points,n_cells,expected', (
    (2, 2, 'line'),
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
    np.testing.assert_allclose(new_mesh.labels, triangle_mesh_2d.labels)


def test_plot_submesh(tetra_mesh):
    plotter = tetra_mesh.plot_submesh(show=False)
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


def test_prune_z_0_no_op(triangle_mesh_3d):
    assert triangle_mesh_3d.points.shape[1] == 3
    triangle_mesh_3d.prune_z_0()
    assert triangle_mesh_3d.points.shape[1] == 3


def test_prune_z_0(triangle_mesh_3d):
    assert triangle_mesh_3d.points.shape[1] == 3
    expected_points = triangle_mesh_3d.points[:, :2]

    triangle_mesh_3d.points[:, 2] = 0.0
    triangle_mesh_3d.prune_z_0()

    assert triangle_mesh_3d.points.shape[1] == 2
    np.testing.assert_equal(triangle_mesh_3d.points, expected_points)
