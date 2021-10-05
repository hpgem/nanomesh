import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from nanomesh.mesh3d import generate_3d_mesh, get_region_markers

TETGEN_NOT_AVAILABLE = shutil.which('tetgen') is None


@pytest.fixture
def segmented_image():
    """Generate segmented binary numpy array."""
    image = np.ones((20, 20, 20))
    image[5:12, 5:12, 0:10] = 0
    image[8:15, 8:15, 10:20] = 0
    return image


def compare_mesh_results(result_mesh, expected_fn):
    """`result_mesh` is an instance of TetraMesh, and `expected_fn` the
    filename of the mesh to compare to."""
    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(result_mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    assert result_mesh.vertices.shape == expected_mesh.vertices.shape
    assert result_mesh.faces.shape == expected_mesh.faces.shape
    np.testing.assert_allclose(result_mesh.vertices, expected_mesh.vertices)
    np.testing.assert_allclose(result_mesh.faces, expected_mesh.faces)

    np.testing.assert_allclose(result_mesh.metadata['tetgenRef'],
                               expected_mesh.metadata['tetgenRef'])


@pytest.mark.xfail(TETGEN_NOT_AVAILABLE,
                   reason='https://github.com/hpgem/nanomesh/issues/106')
def test_generate_3d_mesh(segmented_image):
    """Test 3D mesh generation."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_3d.pickle'

    tetra_mesh = generate_3d_mesh(segmented_image)
    compare_mesh_results(tetra_mesh, expected_fn)


@pytest.mark.xfail(TETGEN_NOT_AVAILABLE,
                   reason='https://github.com/hpgem/nanomesh/issues/106')
def test_generate_3d_mesh_region_markers(segmented_image):
    """Test 3D mesh generation."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_3d_markers.pickle'

    region_markers = get_region_markers(segmented_image)

    tetra_mesh = generate_3d_mesh(segmented_image,
                                  region_markers=region_markers)
    compare_mesh_results(tetra_mesh, expected_fn)
