import numpy as np
import pytest

from nanomesh import (LineMesh, Mesher2D, Mesher3D, Plane, TetraMesh,
                      TriangleMesh, Volume)
from nanomesh.image import GenericImage
from nanomesh.image2mesh import GenericMesher
from nanomesh.mesh import GenericMesh


@pytest.mark.parametrize('data,instance', (
    (np.arange(24), GenericImage),
    (np.arange(24).reshape(4, 6), Plane),
    (np.arange(24).reshape(4, 3, 2), Volume),
))
def test_image_subclassing(data, instance):
    image = GenericImage(data)
    assert isinstance(image, instance)


@pytest.mark.parametrize('data,instance', (
    (1, GenericImage),
    (1, Plane),
    (1, Volume),
))
def test_mesh_subclassing(data, instance):
    image = GenericMesh(data)
    assert isinstance(image, instance)


@pytest.mark.parametrize('data,instance', (
    (1, Mesher2D),
    (1, Mesher3D),
))
def test_mesher_subclassing(data, instance):
    image = GenericMesher(data)
    assert isinstance(image, instance)
