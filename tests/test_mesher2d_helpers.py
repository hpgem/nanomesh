import numpy as np
import pytest

from nanomesh import LineMesh, MeshContainer, Mesher2D, TriangleMesh


@pytest.fixture
def square():
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
def test_pad_side(square, side, shape):
    """Test `side` parameter for `pad`."""
    mesher = Mesher2D(square)
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
def test_pad_width(square, width):
    """Test `width` parameter for `pad`."""
    mesher = Mesher2D(square)
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


@pytest.mark.parametrize('label,expected_labels', (
    (None, {
        0: 91,
        1: 47
    }),
    (0, {
        0: 106,
        1: 32
    }),
    (2, {
        0: 91,
        1: 32,
        2: 15
    }),
    (np.pi, {
        0: 91,
        1: 32,
        np.pi: 15
    }),
))
def test_pad_label(square, label, expected_labels):
    """Test `label` parameter for `pad`."""
    mesher = Mesher2D(square)
    mesher.generate_contour()

    mesher.pad_contour(side='left', width=1, label=label)
    mesh = mesher.triangulate(opts='pq30a1')

    assert isinstance(mesh, MeshContainer)

    tri_mesh = mesh.get('triangle')

    assert isinstance(tri_mesh, TriangleMesh)

    unique, counts = np.unique(tri_mesh.labels, return_counts=True)
    labels = dict(zip(unique, counts))

    assert expected_labels == labels
