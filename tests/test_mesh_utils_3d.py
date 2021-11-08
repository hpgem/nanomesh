import numpy as np
import pytest

from nanomesh.mesh3d import BoundingBox
from nanomesh.mesh_container import TriangleMesh
from nanomesh.mesh_utils_3d import pad3d


@pytest.fixture
def mesh():
    """Box triangle mesh."""
    points = np.array([[0., 0., 0.], [10., 0., 0.], [0., 20., 0.],
                       [10., 20., 0.], [0., 0., 30.], [10., 0., 30.],
                       [0., 20., 30.], [10., 20., 30.]])
    cells = np.array([[0, 4, 6], [0, 6, 2], [5, 1, 3], [5, 3, 7], [0, 1, 5],
                      [0, 5, 4], [6, 7, 3], [6, 3, 2], [1, 0, 2], [1, 2, 3],
                      [4, 5, 7], [4, 7, 6]])

    return TriangleMesh(points=points, cells=cells)


@pytest.mark.parametrize('side,width, expected_bbox', (
    ('top', 5,
     BoundingBox(xmin=0.0, xmax=10.0, ymin=0.0, ymax=20.0, zmin=0.0,
                 zmax=35.0)),
    ('bottom', 5,
     BoundingBox(
         xmin=0.0, xmax=10.0, ymin=0.0, ymax=20.0, zmin=-5.0, zmax=30.0)),
    ('left', 10,
     BoundingBox(
         xmin=0.0, xmax=10.0, ymin=-10.0, ymax=20.0, zmin=0.0, zmax=30.0)),
    ('right', np.pi,
     BoundingBox(
         xmin=0.0, xmax=10.0, ymin=0.0, ymax=20 + np.pi, zmin=0.0, zmax=30.0)),
    ('front', 0.1,
     BoundingBox(
         xmin=-0.1, xmax=10.0, ymin=0.0, ymax=20.0, zmin=0.0, zmax=30.0)),
    ('back', 123,
     BoundingBox(
         xmin=0.0, xmax=133.0, ymin=0.0, ymax=20.0, zmin=0.0, zmax=30.0)),
))
def test_pad3d(mesh, side, width, expected_bbox):
    """Test pad3d function."""
    out = pad3d(mesh, side=side, width=width)

    assert len(out.points) == len(mesh.points) + 4
    assert len(out.cells) == len(mesh.cells) + 10

    bbox = BoundingBox.from_points(out.points)

    assert bbox == expected_bbox


def test_pad3d_no_width(mesh):
    """Test early return when width==0."""
    out = pad3d(mesh, side='top', width=0)

    assert out is mesh


def test_pad3d_invalid_side(mesh):
    """Test invalide keyword argument."""
    with pytest.raises(ValueError):
        _ = pad3d(mesh, side='FAIL', width=123)
