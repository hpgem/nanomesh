import numpy as np

from nanomesh.roi2d import extract_rectangle, minimum_bounding_rectangle


def test_minimum_bounding_rectangle():
    """Test for `minimum_bounding_rectangle`."""
    verts = np.array([[85.91548322, 103.78165584], [52.0951369, 136.92559524],
                      [92.67955248, 157.21780303], [149.4977343,
                                                    140.30762987]])

    expected_bbox = np.array([[149.4977343, 140.30762987],
                              [74.80265615, 97.39769136],
                              [52.0951369, 136.92559524],
                              [126.79021505, 179.83553375]])

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
