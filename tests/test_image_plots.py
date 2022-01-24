import numpy as np
from matplotlib.testing.decorators import image_comparison

from nanomesh import Plane
from nanomesh.utils import show_image


@image_comparison(
    baseline_images=['plane_show'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_show(plane):
    """Test function to plot image."""
    plane.show(title='TESTING')


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
