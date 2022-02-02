import numpy as np
import pytest

from nanomesh import LineMesh, MeshContainer, Mesher2D, TriangleMesh


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
        0: 91,
        1: 32,
        2: 15
    }),
    ('top', 0, None, {
        0: 108,
        1: 34
    }),
    ('top', 1, None, {
        0: 93,
        1: 49
    }),
    ('left', 2, None, {
        0: 91,
        1: 32,
        2: 15
    }),
    ('bottom', np.pi, None, {
        0: 94,
        1: 34,
        np.pi: 15
    }),
    ('right', 2, None, {
        0: 96,
        1: 32,
        2: 14
    }),
    ('bottom', None, 'moo', {
        0: 94,
        1: 34,
        2: 15
    }),
    ('bottom', None, 'background', {
        0: 109,
        1: 34,
    }),
    ('bottom', None, 'feature', {
        0: 94,
        1: 49,
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
    default_keys = {'background', 'feature'}

    if not name:
        assert keys == default_keys
    elif name in default_keys:
        assert keys == default_keys
    else:
        assert keys == default_keys | {
            name,
        }
