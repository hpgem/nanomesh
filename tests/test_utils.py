import matplotlib.pyplot as plt
import numpy as np
import pytest

from nanomesh.utils import SliceViewer


def test_SliceViewer_fails():
    """Test `utils.SliceViewer fails`."""
    data = np.arange(27).reshape(3, 3, 3)
    sv = SliceViewer(data, update_delay=0)

    with pytest.raises(TypeError):  # argument index missing
        sv.update(along='x')

    with pytest.raises(KeyError):
        sv.update(index=0, along='FAIL')

    plt.close()


@pytest.mark.parametrize('along,index,slice', (
    ('x', 0, np.s_[:, :, 0]),
    ('y', 1, np.s_[:, 1, :]),
    ('z', 2, np.s_[2, :, :]),
))
def test_SliceViewer(along, index, slice):
    """Test `utils.SliceViewer`."""
    data = np.arange(27).reshape(3, 3, 3)
    sv = SliceViewer(data, update_delay=0)

    sv.update(along=along, index=index)
    arr = sv.im.get_array()

    np.testing.assert_array_equal(arr, data[slice])

    plt.close()


def test_pairwise():
    """Test pairwise function."""
    from nanomesh.utils import pairwise
    inp = range(3)
    out = pairwise(inp)
    assert list(out) == [(0, 1), (1, 2)]
