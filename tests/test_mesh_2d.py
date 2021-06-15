import os
import pickle
from pathlib import Path

import helpers
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from nanomesh.mesh2d import (add_edge_points, add_points_kmeans,
                             generate_2d_mesh, subdivide_contour)

# There is a small disparity between the data generated on Windows / posix
# platforms (mac/linux). Allow some deviation if the platforms do not match.
# windows: nt, linux/mac: posix
generated_on = 'nt'
if os.name == generated_on:
    MPL_TOL = 0.0
    MESH_TOL = None
else:
    MPL_TOL = 2.0
    MESH_TOL = 0.005


@pytest.fixture
def segmented():
    """Segmented binary numpy array."""
    image_fn = Path(__file__).parent / 'segmented.npy'
    image = np.load(image_fn)
    return image


@image_comparison(
    baseline_images=['segment_2d'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
    tol=MPL_TOL,
)
def test_generate_2d_mesh(segmented):
    """Test 2D mesh generation and plot."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_2d.pickle'

    np.random.seed(1234)  # set seed for reproducible clustering
    mesh = generate_2d_mesh(segmented,
                            pad_width=1,
                            point_density=1 / 100,
                            max_contour_dist=4,
                            plot=True)

    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    helpers.assert_mesh_almost_equal(mesh, expected_mesh, tol=MESH_TOL)


def test_generate_2d_mesh_no_extra_points(segmented):
    """Test if 2D mesh generation works when no extra points are passed."""
    expected_fn = Path(
        __file__).parent / 'segmented_mesh_2d_no_extra_points.pickle'

    np.random.seed(1234)  # set seed for reproducible clustering
    mesh = generate_2d_mesh(segmented,
                            pad_width=0,
                            point_density=0,
                            max_contour_dist=4,
                            plot=False)

    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    helpers.assert_mesh_almost_equal(mesh, expected_mesh, tol=MESH_TOL)


def test_add_edge_points():
    """Test generation of edge points."""
    image = np.zeros((10, 10))
    image[:5, :5] = 1
    image[-5:, -5:] = 1

    ret = add_edge_points(image, n_points=(5, 5))

    expected = np.array([[0, 0], [2, 0], [4, 0], [6, 9], [9, 9], [0, 2],
                         [0, 4], [9, 6]])

    assert np.all(ret == expected)


def test_add_points_kmeans():
    """Test kmeans method for adding points."""
    image = np.zeros((10, 10))
    image[:5, :5] = 1
    image[-5:, -5:] = 1

    np.random.seed(9)
    ret = add_points_kmeans(image, iters=5, n_points=10)

    expected = np.array([[3.6, 1.8], [8., 6.], [0., 2.], [2., 3.5], [8., 8.5],
                         [1.5, 1.5], [2., 0.], [4., 0.], [5.5, 8.], [5.2,
                                                                     5.2]])

    np.testing.assert_almost_equal(ret, expected)


def test_subdivide_contour():
    """Test contour subdivision."""
    contour = np.array([[0, 0], [0, 6], [2, 6], [2, 0], [0, 0]])

    ret = subdivide_contour(contour, max_dist=2)

    expected = np.array([[0., 0.], [0., 2.], [0., 4.], [0., 6.], [2., 6.],
                         [2., 4.], [2., 2.], [2., 0.], [0., 0.]])

    assert np.all(ret == expected)
