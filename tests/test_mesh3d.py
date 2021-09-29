import os
import pickle
from contextlib import nullcontext as do_not_raise
from pathlib import Path

import numpy as np
import pytest

from nanomesh.mesh3d import generate_3d_mesh

# There is a small disparity between the data generated on Windows / posix
# platforms (mac/linux) using tetgen and the randomizer cannot be controlled.
# This will cause the comparison to raise.
# windows: nt, linux/mac: posix
generated_on = 'nt'
if os.name == generated_on:
    expected_raises = do_not_raise()
else:
    expected_raises = pytest.raises(AssertionError)


@pytest.fixture
def segmented_image():
    """Generate segmented binary numpy array."""
    image = np.ones((20, 20, 20))
    image[5:15, 5:15] = 0
    return image


@pytest.mark.xfail(reason='https://github.com/hpgem/nanomesh/issues/106')
def test_generate_3d_mesh(segmented_image):
    """Test 3D mesh generation."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_3d.pickle'

    np.random.seed(1234)  # set seed for reproducible clustering
    tetra_mesh = generate_3d_mesh(segmented_image)

    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(tetra_mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    with expected_raises:
        assert tetra_mesh.vertices.shape == expected_mesh.vertices.shape
        assert tetra_mesh.faces.shape == expected_mesh.faces.shape
        np.testing.assert_allclose(tetra_mesh.vertices, expected_mesh.vertices)
        np.testing.assert_allclose(tetra_mesh.faces, expected_mesh.faces)

        np.testing.assert_allclose(tetra_mesh.metadata['regions'],
                                   expected_mesh.metadata['regions'])
