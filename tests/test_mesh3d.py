import os
import pickle
from pathlib import Path

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


def test_generate_3d_mesh(segmented):
    """Test 3D mesh generation."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_3d.pickle'

    np.random.seed(1234)  # set seed for reproducible clustering
    mesh = generate_3d_mesh(segmented)

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
