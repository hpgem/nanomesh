import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from nanomesh.utils import SliceViewer, requires, show_image


@pytest.mark.parametrize('condition,expected', ((True, True), (False, None)))
def test_requires(condition, expected):
    """Test `requires` functionality."""
    @requires(condition=condition, message=f'condition: {condition}')
    def func():
        return True

    assert func() == expected


def test_SliceViewer_fails():
    """Test `utils.SliceViewer fails`."""
    data = np.arange(27).reshape(3, 3, 3)
    sv = SliceViewer(data, update_delay=0)

    with pytest.raises(ValueError):
        sv.update(along='FAIL')

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


@image_comparison(
    baseline_images=['show_image'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_show_image():
    """Test `utils.show_image`"""
    data = np.arange(25).reshape(5, 5)
    show_image(data, dpi=80, title='TESTING')
