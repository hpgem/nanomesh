import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from nanomesh.mesh3d import BoundingBox, generate_3d_mesh
from nanomesh.mesh3d.mesher import get_region_markers

# There is a small disparity between the data generated on Windows / posix
# platforms (mac/linux): https://github.com/hpgem/nanomesh/issues/144
# Update the variable below for the platform on which the testing data
# have been generated, windows: nt, linux/mac: posix
GENERATED_ON = 'nt'


@pytest.fixture
def segmented_image():
    """Generate segmented binary numpy array."""
    image = np.ones((20, 20, 20))
    image[5:12, 5:12, 0:10] = 0
    image[8:15, 8:15, 10:20] = 0
    return image


def compare_mesh_results(result_mesh, expected_fn):
    """`result_mesh` is an instance of TetraMesh, and `expected_fn` the
    filename of the mesh to compare to."""
    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(result_mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    assert result_mesh.points.shape == expected_mesh.points.shape
    assert result_mesh.cells.shape == expected_mesh.cells.shape
    np.testing.assert_allclose(result_mesh.points, expected_mesh.points)
    np.testing.assert_allclose(result_mesh.cells, expected_mesh.cells)

    np.testing.assert_allclose(result_mesh.cell_data['tetgenRef'],
                               expected_mesh.cell_data['tetgenRef'])


@pytest.mark.xfail(
    os.name != GENERATED_ON,
    raises=AssertionError,
    reason=('No way of currently ensuring meshes on OSX / Linux / Windows '
            'are exactly the same.'))
def test_generate_3d_mesh(segmented_image):
    """Test 3D mesh generation."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_3d.pickle'

    tetra_mesh = generate_3d_mesh(segmented_image)
    compare_mesh_results(tetra_mesh, expected_fn)


@pytest.mark.xfail(
    os.name != GENERATED_ON,
    raises=AssertionError,
    reason=('No way of currently ensuring meshes on OSX / Linux / Windows '
            'are exactly the same.'))
def test_generate_3d_mesh_region_markers(segmented_image):
    """Test 3D mesh generation."""
    expected_fn = Path(__file__).parent / 'segmented_mesh_3d_markers.pickle'

    region_markers = get_region_markers(segmented_image)

    tetra_mesh = generate_3d_mesh(segmented_image,
                                  region_markers=region_markers)
    compare_mesh_results(tetra_mesh, expected_fn)


def test_BoundingBox_center():
    """Test BoundingBox init / center."""
    bbox = BoundingBox(
        xmin=0.0,
        xmax=10.0,
        ymin=20.0,
        ymax=30.0,
        zmin=40.0,
        zmax=50.0,
    )
    np.testing.assert_equal(bbox.center, (5, 25, 45))


def test_BoundingBox_from_shape():
    """Test BoundingBox .from_shape / .dimensions."""
    shape = np.array((10, 20, 30))
    bbox = BoundingBox.from_shape(shape)
    np.testing.assert_equal(bbox.dimensions, shape - 1)


def test_BoundingBox_from_points():
    """Test BoundingBox .from_points / to_points."""
    points = np.array([
        (10, 20, 30),
        (5, 5, 0),
        (0, 0, 5),
    ])
    bbox = BoundingBox.from_points(points)
    corners = bbox.to_points()
    expected = ([0, 0, 0], [0, 0, 30], [0, 20, 0], [0, 20, 30], [10, 0, 0],
                [10, 0, 30], [10, 20, 0], [10, 20, 30])

    np.testing.assert_equal(corners, expected)
