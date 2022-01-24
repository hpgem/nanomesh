from pathlib import Path

import numpy as np
import pytest

from nanomesh.mesh2d import generate_2d_mesh
from nanomesh.mesh2d.polygon import Polygon
from nanomesh.mesh_container import MeshContainer


def block_image(shape=(10, 10)):
    """Generate test array with 4 block quadrants filled with 1 or 0."""
    i, j = (np.array(shape) / 2).astype(int)
    image = np.zeros(shape)
    image[:i, :j] = 1
    image[-i:, -j:] = 1
    return image


@pytest.mark.xfail(pytest.OS_MATCHES_DATA_GEN,
                   raises=AssertionError,
                   reason=('https://github.com/hpgem/nanomesh/issues/144'))
def test_generate_2d_mesh(segmented_image):
    """Test 2D mesh generation and plot."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_2d.msh'

    np.random.seed(1234)  # set seed for reproducible clustering
    mesh = generate_2d_mesh(segmented_image, max_contour_dist=4, plot=True)

    if expected_fn.exists():
        expected_mesh = MeshContainer.read(expected_fn)
    else:
        mesh.write(expected_fn, file_format='gmsh22', binary=False)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    assert mesh.points.shape[1] == 2
    assert mesh.points.shape == expected_mesh.points.shape
    np.testing.assert_allclose(mesh.points, expected_mesh.points)

    cell_types = mesh.cells_dict.keys()
    assert cell_types == expected_mesh.cells_dict.keys()

    for cell_type in cell_types:
        cells = mesh.cells_dict[cell_type]
        expected_cells = expected_mesh.cells_dict[cell_type]

        assert cells.shape == expected_cells.shape
        np.testing.assert_allclose(cells, expected_cells)

    data_keys = mesh.cell_data_dict.keys()
    for data_key in data_keys:
        for cell_type in cell_types:
            data = mesh.cell_data_dict[data_key][cell_type]
            expected_data = expected_mesh.cell_data_dict[data_key][cell_type]

            np.testing.assert_allclose(data, expected_data)


def test_subdivide_polygon():
    """Test polygon subdivision."""
    polygon = Polygon(np.array([[0, 0], [0, 6], [2, 6], [2, 0], [0, 0]]))

    ret = polygon.subdivide(max_dist=2)

    expected_points = np.array([[0., 0.], [0., 2.], [0., 4.], [0.,
                                                               6.], [2., 6.],
                                [2., 4.], [2., 2.], [2., 0.], [0., 0.]])

    assert np.all(ret.points == expected_points)


@pytest.mark.parametrize(
    'coords,expected_corner',
    (
        ([[0, 3], [5, 5], [0, 7]], None),
        ([[0, 3], [5, 5], [3, 0]], [0, 0]),  # bottom, left
        ([[3, 0], [5, 5], [0, 3]], [0, 0]),  # bottom, left
        ([[9, 17], [5, 15], [17, 19]], None),
        ([[9, 5], [7, 4], [4, 0]], [9, 0]),  # bottom, right
        ([[0, 17], [5, 15], [3, 19]], [0, 19]),  # top, left
        ([[9, 17], [5, 15], [3, 19]], [9, 19]),  # top, right
        ([[5, 5], [5, 7], [6, 6]], None),
    ))
def test_close_contour(coords, expected_corner):
    image_shape = 10, 20
    polygon = Polygon(np.array(coords))

    n_rows = polygon.points.shape[1]

    ret = polygon.close_corner(image_shape)

    is_corner = (expected_corner is not None)

    if is_corner:
        ret.points.shape[1] == n_rows + 1
        corner = ret.points[-1]
        np.testing.assert_equal(corner, expected_corner)
    else:
        ret.points.shape[1] == n_rows
