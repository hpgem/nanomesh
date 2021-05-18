import pickle
from pathlib import Path

import helpers
import numpy as np
from matplotlib.testing.decorators import image_comparison

from nanomesh.mesh2d import generate_2d_mesh


@image_comparison(
    baseline_images=['segment_2d'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_generate_2d_mesh():
    """Test 2D mesh generation and plot."""
    expected_fn = Path(__file__).parent / 'segmented_mesh.pickle'
    image_fn = Path(__file__).parent / 'segmented.npy'
    image = np.load(image_fn)

    mesh = generate_2d_mesh(image, pad=True, plot=True)

    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    helpers.assert_mesh_almost_equal(mesh, expected_mesh)
