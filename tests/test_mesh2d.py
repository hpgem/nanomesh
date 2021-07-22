import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from nanomesh._mesh_shared import (add_points_gaussian_mixture,
                                   add_points_kmeans)
from nanomesh.mesh2d import generate_2d_mesh, subdivide_contour

# There is a small disparity between the data generated on Windows / posix
# platforms (mac/linux). Allow some deviation if the platforms do not match.
# windows: nt, linux/mac: posix
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


@pytest.mark.xfail(
    os.name != GENERATED_ON,
    raises=AssertionError,
    reason=('No way of currently ensuring meshes on OSX / Linux / Windows '
            'are exactly the same.'))
def test_generate_2d_mesh(segmented):
    """Test 2D mesh generation and plot."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_2d.pickle'

    np.random.seed(1234)  # set seed for reproducible clustering
    mesh = generate_2d_mesh(segmented, max_contour_dist=4, plot=True)

    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    assert mesh.vertices.shape == expected_mesh.vertices.shape
    assert mesh.faces.shape == expected_mesh.faces.shape
    np.testing.assert_allclose(mesh.vertices, expected_mesh.vertices)
    np.testing.assert_allclose(mesh.faces, expected_mesh.faces)


def test_add_points_kmeans():
    """Test kmeans method for adding points."""
    image = block_image((10, 10))

    np.random.seed(9)
    ret = add_points_kmeans(image, iters=5, n_points=10)

    expected = np.array([[3.6, 1.8], [8., 6.], [0., 2.], [2., 3.5], [8., 8.5],
                         [1.5, 1.5], [2., 0.], [4., 0.], [5.5, 8.], [5.2,
                                                                     5.2]])

    np.testing.assert_almost_equal(ret, expected)


def test_add_points_gaussian_mixture():
    """Test GMM method for adding points."""
    image = block_image((10, 10))

    np.random.seed(9)
    ret = add_points_gaussian_mixture(image, iters=5, n_points=10)

    expected = np.array([[5., 7.00002112], [2.55881426, 1.26971411],
                         [6.52798169, 5.49043914], [8.48578877, 7.91372229],
                         [4., 1.99999747], [2.4822088, 3.53034368],
                         [6.51203175, 7.91383218], [0.50465566, 2.87061786],
                         [8.49298029, 5.48930714], [0.83423033, 0.41400083]])

    np.testing.assert_almost_equal(ret, expected)


def test_subdivide_contour():
    """Test contour subdivision."""
    contour = np.array([[0, 0], [0, 6], [2, 6], [2, 0], [0, 0]])

    ret = subdivide_contour(contour, max_dist=2)

    expected = np.array([[0., 0.], [0., 2.], [0., 4.], [0., 6.], [2., 6.],
                         [2., 4.], [2., 2.], [2., 0.], [0., 0.]])

    assert np.all(ret == expected)
