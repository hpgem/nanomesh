import numpy as np
import pytest

from nanomesh import LineMesh, MeshContainer, Mesher2D, TriangleMesh
from nanomesh.image2mesh._mesher2d._helpers import (append_to_segment_markers,
                                                    generate_segment_markers)


@pytest.fixture
def image_square():
    from nanomesh import Plane

    data = np.ones([10, 10], dtype=int)
    data[2:7, 2:7] = 0

    return Plane(data)


@pytest.mark.parametrize('side,shape', (
    ('left', (0, -1, 9, 9)),
    ('right', (0, 0, 9, 10)),
    ('top', (-1, 0, 9, 9)),
    ('bottom', (0, 0, 10, 9)),
))
def test_pad_side(image_square, side, shape):
    """Test `side` parameter for `pad`."""
    mesher = Mesher2D(image_square)
    mesher.generate_contour()

    mesher.pad_contour(side=side, width=1)
    contour = mesher.contour

    assert isinstance(contour, LineMesh)

    bbox = np.hstack([
        contour.points.min(axis=0),
        contour.points.max(axis=0),
    ])
    np.testing.assert_equal(bbox, shape)

    mesh = mesher.triangulate(opts='pq30a1')
    assert isinstance(mesh, MeshContainer)


@pytest.mark.parametrize('width', (0.0, 0.5, 1, np.pi, 100))
def test_pad_width(image_square, width):
    """Test `width` parameter for `pad`."""
    mesher = Mesher2D(image_square)
    mesher.generate_contour()

    mesher.pad_contour(side='left', width=width)
    contour = mesher.contour

    assert isinstance(contour, LineMesh)

    bbox = np.hstack([
        contour.points.min(axis=0),
        contour.points.max(axis=0),
    ])

    np.testing.assert_equal(bbox, (0, -width, 9, 9))

    mesh = mesher.triangulate(opts='pq30a1')
    assert isinstance(mesh, MeshContainer)


@pytest.mark.parametrize('side,label,name,expected_labels', (
    ('left', None, None, {
        1: 91,
        2: 32,
        3: 15
    }),
    ('top', 1, None, {
        1: 108,
        2: 34
    }),
    ('top', 2, None, {
        1: 93,
        2: 49
    }),
    ('left', 3, None, {
        1: 91,
        2: 32,
        3: 15
    }),
    ('bottom', np.pi, None, {
        1: 94,
        2: 34,
        np.pi: 15
    }),
    ('right', 3, None, {
        1: 96,
        2: 32,
        3: 14
    }),
    ('bottom', None, 'moo', {
        1: 94,
        2: 34,
        3: 15
    }),
    ('bottom', None, 'background', {
        1: 109,
        2: 34,
    }),
    ('bottom', None, 'X', {
        1: 94,
        2: 49,
    }),
))
def test_pad_label(image_square, side, label, name, expected_labels):
    """Test `label` parameter for `pad`."""
    mesher = Mesher2D(image_square)
    mesher.generate_contour()

    mesher.pad_contour(side=side, width=1, label=label, name=name)
    mesh = mesher.triangulate(opts='pq30a1')

    assert isinstance(mesh, MeshContainer)

    tri_mesh = mesh.get('triangle')

    assert isinstance(tri_mesh, TriangleMesh)

    unique, counts = np.unique(tri_mesh.labels, return_counts=True)
    labels = dict(zip(unique, counts))

    assert expected_labels == labels

    keys = set(tri_mesh.field_to_number.keys())
    default_keys = {'background', 'X'}

    if not name:
        assert keys == default_keys
    elif name in default_keys:
        assert keys == default_keys
    else:
        assert keys == default_keys | {
            name,
        }


@pytest.mark.parametrize('ones', (False, True))
def test_generate_segment_markers(ones):
    i, j, k = 2, 3, 1

    inp = [np.arange(i), np.arange(j), np.arange(k)]
    out = generate_segment_markers(inp, ones=ones)

    if ones:
        expected = [1, 1, 1, 1, 1, 1]
    else:
        expected = [1, 1, 2, 2, 2, 3]

    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize('same_label', (False, True))
def test_append_to_segment_markers(same_label):
    inp = np.array([1, 1, 1])
    extra = [(1, 2), (2, 3), (3, 4)]

    out = append_to_segment_markers(inp, extra, same_label=same_label)

    if same_label:
        expected = [1, 1, 1, 2, 2, 2]
    else:
        expected = [1, 1, 1, 2, 3, 4]

    np.testing.assert_allclose(out, expected)
