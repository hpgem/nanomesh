# https://github.com/matplotlib/matplotlib/issues/18168#issuecomment-670211108
import numpy as np
import pytest
from matplotlib.testing.conftest import mpl_test_settings

from nanomesh import TetraMesh, TriangleMesh


@pytest.fixture
def triangle_mesh_2d():
    points = np.arange(10).reshape(5, 2)
    cells = np.zeros((5, 3), dtype=int)
    cell_data = {'labels': np.arange(5)}

    mesh = TriangleMesh.create(cells=cells, points=points, **cell_data)
    assert isinstance(mesh, TriangleMesh)
    return mesh


@pytest.fixture
def triangle_mesh_3d():
    points = np.arange(15).reshape(5, 3)
    cells = np.zeros((5, 3), dtype=int)
    cell_data = {'labels': np.arange(5)}

    mesh = TriangleMesh.create(cells=cells, points=points, **cell_data)
    assert isinstance(mesh, TriangleMesh)
    return mesh


@pytest.fixture
def tetra_mesh():
    points = np.arange(15).reshape(5, 3)
    cells = np.zeros((5, 4), dtype=int)
    cell_data = {'labels': np.arange(5)}

    mesh = TetraMesh.create(cells=cells, points=points, **cell_data)
    assert isinstance(mesh, TetraMesh)
    return mesh
