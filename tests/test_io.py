from pathlib import Path

import numpy as np

from nanomesh.io import load_vol


def test_load_data():
    """Test loading of vol files."""
    fn = Path(__file__).parent / 'sample_data.vol'

    data = load_vol(fn, dtype=np.uint8)

    shape = (2, 3, 4)
    expected_data = np.arange(np.product(shape), dtype=np.uint8).reshape(shape)

    assert data.shape == shape
    np.testing.assert_equal(data, expected_data)


def test_load_data_shape():
    """Test loading of vol files."""
    fn = Path(__file__).parent / 'sample_data.vol'

    shape = (6, 2)

    data = load_vol(fn, dtype=np.uint8, mmap_mode='r', shape=shape)

    assert data.shape == shape
