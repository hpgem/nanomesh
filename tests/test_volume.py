import numpy as np
import pytest

from nanomesh.plane import Plane
from nanomesh.volume import Volume


@pytest.fixture
def volume():
    data = np.arange(125).reshape(5, 5, 5)
    return Volume(data)


def test_to_from_sitk(volume):
    """Test `SimpleITK` conversion."""
    pytest.importorskip('SimpleITK')

    sitk_image = volume.to_sitk_image()
    new_volume = Volume.from_sitk_image(sitk_image)

    assert new_volume == volume


def test_load_volume(volume, tmp_path):
    """Test `.load` classmethod."""
    filename = tmp_path / 'data_file.npy'

    np.save(filename, volume.image)
    new_volume = volume.load(filename)

    assert new_volume == volume


def test_load_volume_unknown_extension(volume, tmp_path):
    """Test `.load` classmethod fail."""
    filename = tmp_path / 'data_file.rawr'

    with pytest.raises(IOError):
        _ = volume.load(filename)


def test_apply(volume):
    """Test whether the apply method works with numpy functions."""
    def return_array(image):
        return image

    def return_scalar(image):
        return 123

    assert volume.apply(return_array) == volume
    assert volume.apply(return_scalar) == 123


def test_show_slice(volume):
    """Test slice viewer call."""
    from nanomesh.utils import SliceViewer
    sv = volume.show_slice()
    assert isinstance(sv, SliceViewer)


@pytest.mark.parametrize('kwargs,expected', (
    ({
        'x': 1
    }, np.s_[:, :, 1]),
    ({
        'y': 2
    }, np.s_[:, 2, :]),
    ({
        'z': 3
    }, np.s_[3, ...]),
))
def test_select_plane(volume, kwargs, expected):
    """Test plane selection method."""
    ret = volume.select_plane(**kwargs)
    assert isinstance(ret, Plane)
    np.testing.assert_array_equal(ret.image, volume.image[expected])


@pytest.mark.parametrize('kwargs,expected', (
    ({
        'xs': (1, 2)
    }, np.s_[:, :, 1:2]),
    ({
        'ys': (2, 3)
    }, np.s_[:, 2:3, :]),
    ({
        'zs': (3, 4)
    }, np.s_[3:4, :, :]),
    ({
        'xs': (1, 3),
        'ys': (2, 4),
        'zs': (3, 5)
    }, np.s_[3:5, 2:4, 1:3]),
))
def test_select_subvolume(volume, kwargs, expected):
    """Test plane selection method."""
    ret = volume.select_subvolume(**kwargs)
    assert isinstance(ret, Volume)
    np.testing.assert_array_equal(ret.image, volume.image[expected])


def test_gaussian(volume):
    out = volume.gaussian()
    assert isinstance(out, Volume)


def test_fft(volume):
    out = volume.fft()
    assert isinstance(out, Volume)


def test_digitize(volume):
    out = volume.digitize(bins=[25, 50, 75, 100])
    assert isinstance(out, Volume)
    assert np.all(np.unique(out.image) == np.array([0, 1, 2, 3, 4]))


@pytest.mark.parametrize('threshold', (None, 100, 'li'))
def test_binary_digitize(volume, threshold):
    out = volume.binary_digitize(threshold=threshold)
    assert isinstance(out, Volume)
    assert np.all(np.unique(out.image) == np.array([0, 1]))
