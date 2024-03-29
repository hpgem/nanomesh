import os
from pathlib import Path

import numpy as np
import pytest
# https://github.com/matplotlib/matplotlib/issues/18168#issuecomment-670211108
from matplotlib.testing.conftest import mpl_test_settings  # noqa

from nanomesh import (LineMesh, MeshContainer, Plane, TetraMesh, TriangleMesh,
                      Volume)

LABEL_KEY = 'labels'


def pytest_configure():
    # There is a small disparity between the data generated on Windows / posix
    # platforms (mac/linux): https://github.com/hpgem/nanomesh/issues/144
    # Update the variable below for the platform on which the testing data
    # have been generated, windows: nt, linux/mac: posix
    pytest.DATA_GENERATED_ON = 'nt'
    pytest.OS_MATCHES_DATA_GEN = (os.name == pytest.DATA_GENERATED_ON)
    pytest.OS_DOES_NOT_MATCH_DATA_GEN = not pytest.OS_MATCHES_DATA_GEN


@pytest.fixture
def line_mesh():
    points = np.arange(10).reshape(5, 2)
    cells = np.tile(np.arange(5, dtype=int).reshape(-1, 1), 2)
    cell_data = {LABEL_KEY: np.arange(5)}

    mesh = LineMesh(cells=cells, points=points, **cell_data)
    mesh.default_key = LABEL_KEY
    assert isinstance(mesh, LineMesh)
    return mesh


@pytest.fixture
def triangle_mesh_2d():
    points = np.arange(10).reshape(5, 2)
    cells = np.tile(np.arange(5, dtype=int).reshape(-1, 1), 3)
    cell_data = {LABEL_KEY: np.arange(5)}

    mesh = TriangleMesh(cells=cells, points=points, **cell_data)
    mesh.default_key = LABEL_KEY
    assert isinstance(mesh, TriangleMesh)
    return mesh


@pytest.fixture
def triangle_mesh_3d():
    points = np.arange(15).reshape(5, 3)
    cells = np.tile(np.arange(5, dtype=int).reshape(-1, 1), 3)
    cell_data = {LABEL_KEY: np.arange(5)}

    mesh = TriangleMesh(cells=cells, points=points, **cell_data)
    mesh.default_key = LABEL_KEY
    assert isinstance(mesh, TriangleMesh)
    return mesh


@pytest.fixture
def tetra_mesh():
    points = np.arange(15).reshape(5, 3)
    cells = np.tile(np.arange(5, dtype=int).reshape(-1, 1), 4)
    cell_data = {LABEL_KEY: np.arange(5)}

    mesh = TetraMesh(cells=cells, points=points, **cell_data)
    assert isinstance(mesh, TetraMesh)
    return mesh


@pytest.fixture
def line_tri_mesh():
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
        [1, 3],
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
        'physical': [
            [0, 0, 0, 0, 1],
            [1, 2],
        ]
    }

    point_data = {
        'physical': [1, 2, 3, 4],
    }

    return MeshContainer(
        points=points,
        cells={
            'line': lines,
            'triangle': triangles
        },
        field_data=field_data,
        cell_data=cell_data,
        point_data=point_data,
    )


@pytest.fixture
def segmented_image():
    """Segmented binary numpy array."""
    image_fn = Path(__file__).parent / 'segmented_image.npy'
    image = np.load(image_fn)
    return image


@pytest.fixture
def segmented_image_3d():
    """Generate segmented binary numpy array."""
    image = np.ones((20, 20, 20))
    image[5:12, 5:12, 0:10] = 0
    image[8:15, 8:15, 10:20] = 0
    return image


@pytest.fixture
def plane():
    data = np.arange(625).reshape(25, 25)
    return Plane(data)


@pytest.fixture
def volume():
    data = np.arange(125).reshape(5, 5, 5)
    return Volume(data)


@pytest.fixture
def sample_triangle_mesh():
    expected_fn = Path(__file__).parent / 'segmented_mesh_2d.msh'
    mesh = MeshContainer.read(expected_fn)

    tri_mesh = mesh.get('triangle')

    return tri_mesh
