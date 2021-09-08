import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from nanomesh import Plane
from nanomesh.mesh_container import TriangleMesh


@pytest.fixture
def plane():
    data = np.arange(625).reshape(25, 25)
    return Plane(data)


def test_to_from_sitk(plane):
    """Test `SimpleITK` conversion."""
    pytest.importorskip('SimpleITK')

    sitk_image = plane.to_sitk_image()
    new_plane = Plane.from_sitk_image(sitk_image)

    assert new_plane == plane


def test_load_plane(plane, tmp_path):
    """Test `.load` classmethod."""
    filename = tmp_path / 'data_file.npy'

    np.save(filename, plane.image)
    new_plane = plane.load(filename)

    assert new_plane == plane


def test_apply(plane):
    """Test whether the apply method works with numpy functions."""
    def return_array(image):
        return image

    def return_scalar(image):
        return 123

    assert plane.apply(return_array) == plane
    assert plane.apply(return_scalar) == 123


@image_comparison(
    baseline_images=['plane_show'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_show(plane):
    """Test function to plot image."""
    plane.show(title='TESTING')


@pytest.mark.skip(reason='This test hangs')
def test_generate_mesh(plane):
    """Property test for mesh generation method."""
    seg = Plane(1.0 * (plane.image > 5))
    mesh = seg.generate_mesh(plot=False)
    assert isinstance(mesh, TriangleMesh)


def test_select_roi(plane):
    """Property test for roi selector."""
    roi = plane.select_roi()
    assert hasattr(roi, 'bbox')


def test_crop_to_roi(plane):
    """Test cropping method."""
    bbox = np.array([(1, 1), (1, 3), (3, 3), (3, 1)])
    cropped = plane.crop_to_roi(bbox=bbox)
    assert isinstance(cropped, Plane)
    assert cropped.image.shape == (2, 2)


def test_crop(plane):
    """Test cropping."""
    cropped = plane.crop(left=1, right=4, bottom=5, top=3)
    assert cropped.image.shape == (2, 3)


def test_equal(plane):
    """Test equivalency."""
    assert plane == plane
    assert plane == plane.image
    assert plane != 123


@image_comparison(
    baseline_images=['compare_with_digitized'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_compare_with_digitized(plane):
    segmented = Plane(plane.image > plane.image.mean())
    plane.compare_with_digitized(segmented)


@image_comparison(
    baseline_images=['compare_with_other'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_compare_with_other(plane):
    other = Plane(np.ones_like(plane.image))
    plane.compare_with_other(other)


def test_gaussian(plane):
    out = plane.gaussian()
    assert isinstance(out, Plane)


def test_digitize(plane):
    out = plane.digitize(bins=[125, 250, 375, 500])
    assert isinstance(out, Plane)
    assert np.all(np.unique(out.image) == np.array([0, 1, 2, 3, 4]))


def test_binary_digitize(plane):
    out = plane.binary_digitize()
    assert isinstance(out, Plane)
    assert np.all(np.unique(out.image) == np.array([0, 1]))


def test_clear_border():
    arr = np.array([
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 0, 0, 0],
    ])
    plane = Plane(arr)
    out = plane.clear_border(object_label=2, fill_val=5)
    assert isinstance(out, Plane)
    assert np.all(out.image[0, 0:2] == [5, 5])
    assert np.all(out.image[2, 1:3] == [2, 2])
