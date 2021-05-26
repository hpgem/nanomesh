import pickle
from pathlib import Path

import helpers
import numpy as np
import pytest
import sys
from matplotlib.testing.decorators import image_comparison

from nanomesh.mesh2d import (add_edge_points, add_points_grid,
                             add_points_kmeans, generate_2d_mesh,
                             subdivide_contour)

# Reference points were generated on Linux. There is a minor difference 
# between the generated on Linux/Mac and Windows. Allow for some deviation.
windows = (sys.platform == 'win32')


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
    tol=2.0 if windows else 0,
)
def test_generate_2d_mesh(segmented):
    """Test 2D mesh generation and plot."""
    expected_fn = Path(__file__).parent / 'segmented_mesh.pickle'

    np.random.seed(1234)  # set seed for reproducible clustering
    mesh = generate_2d_mesh(segmented,
                            pad=True,
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

    tol = 0.0025 if windows else 0
    helpers.assert_mesh_almost_equal(mesh, expected_mesh, tol=tol)


def test_add_edge_points():
    """Test generation of edge points."""
    image = np.zeros((10, 10))
    image[:5, :5] = 1
    image[-5:, -5:] = 1

    ret = add_edge_points(image, n_points=(5, 5))

    expected = np.array([[0, 0], [2, 0], [4, 0], [6, 9], [9, 9], [0, 2],
                         [0, 4], [9, 6]])

    assert np.all(ret == expected)


def test_add_points_grid():
    """Test grid method for adding points."""
    image = np.zeros((10, 10))
    image[:5, :5] = 1
    image[-5:, -5:] = 1

    ret = add_points_grid(image, border=1, n_points=4)

    expected = np.array([[0, 0], [3, 0], [0, 3], [3, 3], [6, 6], [9, 6],
                         [6, 9], [9, 9]])

    assert np.all(ret == expected)


def test_add_points_kmeans():
    """Test kmeans method for adding points."""
    image = np.zeros((10, 10))
    image[:5, :5] = 1
    image[-5:, -5:] = 1

    np.random.seed(9)
    ret = add_points_kmeans(image, iters=5, n_points=10)

    expected = np.array([[3.6, 2.8], [8., 6.], [0., 2.], [2., 3.75], [8., 8.5],
                         [1.25, 2.], [2.2, 0.4], [4., 0.5], [5.5, 8.],
                         [5.5, 5.5]])

    np.testing.assert_almost_equal(ret, expected)


def test_subdivide_contour():
    """Test contour subdivision."""
    contour = np.array([[0, 0], [0, 6], [2, 6], [2, 0], [0, 0]])

    ret = subdivide_contour(contour, max_dist=2)

    expected = np.array([[0., 0.], [0., 2.], [0., 4.], [0., 6.], [2., 6.],
                         [2., 4.], [2., 2.], [2., 0.], [0., 0.]])

    assert np.all(ret == expected)
