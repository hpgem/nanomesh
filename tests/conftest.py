import numpy as np
import pytest
# https://github.com/matplotlib/matplotlib/issues/18168#issuecomment-670211108
from matplotlib.testing.conftest import mpl_test_settings  # noqa

from nanomesh import LineMesh, MeshContainer, TetraMesh, TriangleMesh


@pytest.fixture
def line_mesh():
    points = np.arange(10).reshape(5, 2)
    cells = np.zeros((5, 3), dtype=int)
    cell_data = {'labels': np.arange(5)}

    mesh = LineMesh.create(cells=cells, points=points, **cell_data)
    assert isinstance(mesh, TriangleMesh)
    return mesh


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


@pytest.fixture
def mesh_square2d():
    points = np.array([
        [0., 0.],
        [0., 1.],
        [1., 1.],
        [1., 0.],
    ])

    lines = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2],
    ])

    triangles = np.array([[1, 0, 3], [3, 2, 1]])

    LINE = 1
    TRIANGLE = 2

    field_data = {
        'Triangle A': [1, TRIANGLE],
        'Triangle B': [2, TRIANGLE],
        'Line A': [0, LINE],
        'Line B': [1, LINE],
    }

    cell_data = {
        'data': [
            [0, 0, 0, 0, 1],
            [1, 2],
        ]
    }

    return MeshContainer(points=points,
                         cells={
                             'line': lines,
                             'triangle': triangles
                         },
                         field_data=field_data,
                         cell_data=cell_data)
