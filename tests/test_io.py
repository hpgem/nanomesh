"""Tests for the nanomesh module."""
from pathlib import Path

import numpy as np

from nanomesh.io import load_bin


def test_load_data():
    """Test loading of vol files."""
    fn = (Path(__file__).parents[1] / 'notebooks' / 'sample_data' /
          'sample_data.vol')

    expected_fn = Path(__file__).parent / 'sample_data.npy'

    data = load_bin(fn, input_dtype=np.uint8)

    if expected_fn.exists():
        expected_data = np.load(expected_fn)
    else:
        np.save(expected_fn, data)
        raise RuntimeError(f'Wrote expected data to {expected_fn.absolute()}')

    assert data.shape == (200, 200, 200)
    np.testing.assert_equal(data, expected_data)
