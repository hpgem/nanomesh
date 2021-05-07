"""Tests for the nanomesh module."""
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from nanomesh import NanoMesher


def test_load_data():
    """Test loading of vol files."""
    fn = Path(__file__).parents[1] / 'notebook' / 'sandbox' / 'sample_data.vol'
    expected_fn = Path(__file__).parent / fn.with_suffix('.npy').name

    mesh = NanoMesher()
    mesh.load_bin(fn,
                  size=[200, 200, 200],
                  input_dtype=np.uint8,
                  output_dtype=np.float32)
    volume = mesh.volume

    if expected_fn.exists():
        expected_data = np.load(expected_fn)
    else:
        np.save(expected_fn, volume.data)
        RuntimeError(f'Wrote expected data to {expected_fn.absolute()}')

    assert str(volume.name) == str(fn)
    assert volume.info is None
    assert volume.size == [200, 200, 200]
    assert volume.format is None
    np.testing.assert_equal(volume.data, expected_data)
    assert isinstance(volume.img, sitk.SimpleITK.Image)
