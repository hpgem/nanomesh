import numpy as np
import pytest

from nanomesh import (LineMesh, Mesher2D, Mesher3D, Plane, TetraMesh,
                      TriangleMesh, Volume)
from nanomesh.image import Image
from nanomesh.image2mesh._base import AbstractMesher as GenericMesher
from nanomesh.mesh import GenericMesh

im1d = np.arange(24)
im2d = np.arange(24).reshape(6, 4)
im3d = np.arange(24).reshape(3, 4, 2)

points = np.array((
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
))

lines = np.array((
    (0, 1),
    (1, 2),
    (2, 3),
))

triangles = np.array((
    (0, 1, 2),
    (1, 2, 3),
    (3, 0, 1),
))

tetras = np.array((
    (0, 1, 2, 3),
    (3, 2, 1, 0),
))

other = np.array((
    (0, ),
    (1, ),
    (2, ),
    (3, ),
))


@pytest.mark.parametrize('data,instance', (
    (im1d, Image),
    (im2d, Plane),
    (im2d, Plane),
    (im3d, Volume),
))
def test_image_subclassing(data, instance):
    image = Image(data)
    assert isinstance(image, instance)


@pytest.mark.parametrize('data,instance', (
    ((points, other), GenericMesh),
    ((points, lines), LineMesh),
    ((points, triangles), TriangleMesh),
    ((points, tetras), TetraMesh),
))
def test_mesh_subclassing(data, instance):
    mesh = GenericMesh(*data)
    assert isinstance(mesh, instance)


@pytest.mark.parametrize('data,instance', (
    (im2d, Mesher2D),
    (Plane(im2d), Mesher2D),
    (im3d, Mesher3D),
    (Volume(im3d), Mesher3D),
))
def test_mesher_subclassing(data, instance):
    image = GenericMesher(data)
    assert isinstance(image, instance)
