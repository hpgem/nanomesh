import os
import pickle
from pathlib import Path

import helpers
import numpy as np
import pytest

from nanomesh.mesh3d import generate_3d_mesh

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
    """Generate segmented binary numpy array."""
    image = np.ones((20, 20, 20))
    image[5:15, 5:15] = 0
    return image


def test_generate_2d_mesh(segmented):
    """Test 3D mesh generation."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_3d.pickle'

    np.random.seed(1234)  # set seed for reproducible clustering
    mesh = generate_3d_mesh(segmented, point_density=1 / 100, pad_width=5)

    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    helpers.assert_mesh_almost_equal(mesh, expected_mesh, tol=MESH_TOL)
