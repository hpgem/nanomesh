import numpy as np
import pytest
import pyvista as pv

from nanomesh.mesh import Mesh, TriangleMesh


@pytest.mark.parametrize('n_points,n_cells,expected', (
    (2, 2, 'line'),
    (2, 3, 'triangle'),
    (3, 3, 'triangle'),
    (3, 4, 'tetra'),
))
def test_create(n_points, n_cells, expected):
    points = np.arange(5 * n_points).reshape(5, n_points)
    cells = np.zeros((5, n_cells))

    mesh = Mesh(points=points, cells=cells)

    assert mesh.cell_type == expected


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


def test_line_mesh_label_boundary(line_tri_mesh):
    line_mesh = line_tri_mesh.get('line')

    key = line_mesh.default_key
    line_mesh.label_boundaries(left='left',
                               right=123,
                               top='moo',
                               bottom='bottom',
                               key=key)

    cell_data = line_mesh.cell_data[key]

    np.testing.assert_allclose(cell_data, (125, 123, 124, 2, 1))
    assert line_mesh.fields == {
        'Line A': 0,
        'Line B': 1,
        'left': 2,
        'moo': 124,
        'bottom': 125
    }


@pytest.mark.parametrize('cell_type', ('line', 'triangle'))
def test_reverse_cell_order(line_tri_mesh, cell_type):
    mesh = line_tri_mesh.get(cell_type)
    key = mesh.default_key

    points = mesh.points
    cells = mesh.cells
    data = mesh.cell_data[key]

    mesh.reverse_cell_order()

    np.testing.assert_allclose(mesh.points, points)
    np.testing.assert_allclose(mesh.cells, cells[::-1])
    np.testing.assert_allclose(mesh.cell_data[key], data[::-1])


@pytest.mark.parametrize('key', (None, 'labels'))
def test_remove_cells(triangle_mesh_2d, key):
    npoints = len(triangle_mesh_2d.points)
    np.testing.assert_equal(triangle_mesh_2d.labels, np.arange(5))
    triangle_mesh_2d.remove_cells(label=4, key=key)
    np.testing.assert_equal(triangle_mesh_2d.labels, np.arange(4))
    # ensure that orphaned points are be removed
    assert len(triangle_mesh_2d.points) == npoints - 1


def test_remove_loose_points(triangle_mesh_2d):
    i = 1
    unorphaned = triangle_mesh_2d.points[i:]
    triangle_mesh_2d.cells = triangle_mesh_2d.cells[i:]
    triangle_mesh_2d.remove_loose_points()
    np.testing.assert_equal(triangle_mesh_2d.points, unorphaned)


def test_mesh_crop_2d(triangle_mesh_2d):
    new_mesh = triangle_mesh_2d.crop(xmin=0, xmax=2)

    np.testing.assert_equal(new_mesh.cells, np.array([
        [0, 0, 0],
        [1, 1, 1],
    ]))
    np.testing.assert_equal(new_mesh.points, np.array([
        [0, 1],
        [2, 3],
    ]))
    np.testing.assert_equal(new_mesh.labels, np.array([0, 1]))


def test_mesh_crop_3d(triangle_mesh_3d):
    new_mesh = triangle_mesh_3d.crop(zmin=0, zmax=5)

    np.testing.assert_equal(new_mesh.cells, np.array([
        [0, 0, 0],
        [1, 1, 1],
    ]))
    np.testing.assert_equal(new_mesh.points, np.array([
        [0, 1, 2],
        [3, 4, 5],
    ]))
    np.testing.assert_equal(new_mesh.labels, np.array([0, 1]))
