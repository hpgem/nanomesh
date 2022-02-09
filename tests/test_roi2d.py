import numpy as np

from nanomesh.image import extract_rectangle, minimum_bounding_rectangle


def test_minimum_bounding_rectangle():
    """Test for `minimum_bounding_rectangle`."""
    verts = np.array([
        [60, 60],
        [110, 110],
        [60, 145],
        [10, 110],
    ])

    expected_bbox = np.array([[110., 110.], [60., 60.], [10., 110.],
                              [60., 160.]])

    bbox = minimum_bounding_rectangle(verts)

    assert bbox.shape == (4, 2)
    np.testing.assert_almost_equal(bbox, expected_bbox)


def test_extract_rectangle():
    """Test for `extract_rectangle`."""
    image = np.arange(25, dtype=float).reshape(5, 5)
    bbox = np.array([[1, 1], [1, 3], [3, 3], [3, 1]])

    expected_output = np.array([[6., 7.], [11., 12.]])

    output = extract_rectangle(image, bbox=bbox)

    np.testing.assert_almost_equal(output, expected_output)
