import numpy as np
import pytest
from helpers import get_expected_if_it_exists

from nanomesh.mesh3d import BoundingBox, generate_3d_mesh


def compare_mesh_results(mesh_container, expected_fn):
    """`result_mesh` is an instance of TetraMesh, and `expected_fn` the
    filename of the mesh to compare to."""
    expected_mesh_container = get_expected_if_it_exists(expected_fn,
                                                        result=mesh_container)

    cell_type = 'tetra'

    mesh = mesh_container.get(cell_type)
    expected_mesh = expected_mesh_container.get(cell_type)

    assert mesh.points.shape == expected_mesh.points.shape
    assert mesh.cells.shape == expected_mesh.cells.shape
    np.testing.assert_allclose(mesh.points, expected_mesh.points)
    np.testing.assert_allclose(mesh.cells, expected_mesh.cells)

    np.testing.assert_allclose(mesh.region_markers,
                               expected_mesh.region_markers)

    for key in expected_mesh.cell_data:
        try:
            np.testing.assert_allclose(mesh.cell_data[key],
                                       expected_mesh.cell_data[key])
        except KeyError:
            if key not in ('physical', 'geometrical'):
                raise


@pytest.mark.xfail(
    pytest.OS_DOES_NOT_MATCH_DATA_GEN,
    raises=AssertionError,
    reason=('No way of currently ensuring meshes on OSX / Linux / Windows '
            'are exactly the same.'))
def test_generate_3d_mesh(segmented_image_3d):
    """Test 3D mesh generation."""
    expected_fn = 'segmented_mesh_3d.msh'

    mesh_container = generate_3d_mesh(segmented_image_3d)

    assert 'tetgen:ref' in mesh_container.cell_data

    compare_mesh_results(mesh_container, expected_fn)


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
