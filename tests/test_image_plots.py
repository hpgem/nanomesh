import numpy as np
from helpers import image_comparison2

from nanomesh import Plane
from nanomesh.image import show_image


@image_comparison2(baseline_images=['plane_show'])
def test_show(plane):
    """Test function to plot image."""
    plane.show(title='TESTING')


@image_comparison2(baseline_images=['compare_with_digitized'])
def test_compare_with_digitized(plane):
    segmented = Plane(plane.image > plane.image.mean())
    plane.compare_with_digitized(segmented)


@image_comparison2(baseline_images=['compare_with_other'])
def test_compare_with_other(plane):
    other = Plane(np.ones_like(plane.image))
    plane.compare_with_other(other)


@image_comparison2(baseline_images=['show_image'])
def test_show_image():
    """Test `utils.show_image`"""
    data = np.arange(25).reshape(5, 5)
    show_image(data, title='TESTING')
