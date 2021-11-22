import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from nanomesh.mesh2d import Mesher2D, generate_2d_mesh
from nanomesh.mesh2d.mesher import close_corner_contour, subdivide_contour
from nanomesh.mesh_container import MeshContainer

# There is a small disparity between the data generated on Windows / posix
# platforms (mac/linux): https://github.com/hpgem/nanomesh/issues/144
# Update the variable below for the platform on which the testing data
# have been generated, windows: nt, linux/mac: posix
GENERATED_ON = 'nt'


def block_image(shape=(10, 10)):
    """Generate test array with 4 block quadrants filled with 1 or 0."""
    i, j = (np.array(shape) / 2).astype(int)
    image = np.zeros(shape)
    image[:i, :j] = 1
    image[-i:, -j:] = 1
    return image


@pytest.fixture
def segmented():
    """Segmented binary numpy array."""
    image_fn = Path(__file__).parent / 'segmented.npy'
    image = np.load(image_fn)
    return image


@pytest.mark.xfail(os.name != GENERATED_ON,
                   raises=AssertionError,
                   reason=('https://github.com/hpgem/nanomesh/issues/144'))
def test_generate_2d_mesh(segmented):
    """Test 2D mesh generation and plot."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_2d.msh'

    np.random.seed(1234)  # set seed for reproducible clustering
    mesh = generate_2d_mesh(segmented, max_contour_dist=4, plot=True)

    if expected_fn.exists():
        expected_mesh = MeshContainer.read(expected_fn)
        expected_mesh.prune_z_0()  # gmsh pads 0-column on save
    else:
        mesh.write(expected_fn, file_format='gmsh22', binary=False)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

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


def test_subdivide_contour():
    """Test contour subdivision."""
    contour = np.array([[0, 0], [0, 6], [2, 6], [2, 0], [0, 0]])

    ret = subdivide_contour(contour, max_dist=2)

    expected = np.array([[0., 0.], [0., 2.], [0., 4.], [0., 6.], [2., 6.],
                         [2., 4.], [2., 2.], [2., 0.], [0., 0.]])

    assert np.all(ret == expected)


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
    image_chape = 10, 20
    contour = np.array(coords)

    n_rows = contour.shape[1]

    ret = close_corner_contour(contour, image_chape)

    is_corner = (expected_corner is not None)

    if is_corner:
        ret.shape[1] == n_rows + 1
        corner = ret[-1]
        np.testing.assert_equal(corner, expected_corner)
    else:
        ret.shape[1] == n_rows


@pytest.mark.xfail(os.name != GENERATED_ON,
                   raises=AssertionError,
                   reason=('https://github.com/hpgem/nanomesh/issues/144'))
@image_comparison(
    baseline_images=['contour_plot'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_contour_plot(segmented):
    mesher = Mesher2D(segmented)
    mesher.generate_contours(max_contour_dist=5, level=0.5)
    mesher.plot_contour()
