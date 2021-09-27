import pickle
from pathlib import Path

import numpy as np
import pytest

from nanomesh.mesh_container import TriangleMesh


@pytest.fixture
def triangle_mesh():
    """Generate mesh.

    This mesh is a cube with a rectangular 'pore' through the middle.
    """
    vertices = np.array([
        # cube
        [0.0, 0.0, 0.0],  # A
        [4.0, 0.0, 0.0],  # B
        [4.0, 4.0, 0.0],  # C
        [0.0, 4.0, 0.0],  # D
        [0.0, 0.0, 4.0],  # E
        [4.0, 0.0, 4.0],  # F
        [4.0, 4.0, 4.0],  # G
        [0.0, 4.0, 4.0],  # H
        # inside rectangle ('pore')
        [1.0, 1.0, 0.0],  # a
        [3.0, 1.0, 0.0],  # b
        [3.0, 3.0, 0.0],  # c
        [1.0, 3.0, 0.0],  # d
        [1.0, 1.0, 4.0],  # e
        [3.0, 1.0, 4.0],  # f
        [3.0, 3.0, 4.0],  # g
        [1.0, 3.0, 4.0],  # h
    ])

    faces = np.array([
        # top face
        [0, 11, 8],
        [1, 8, 9],
        [2, 9, 10],
        [3, 10, 11],
        [0, 8, 1],
        [1, 9, 2],
        [2, 10, 3],
        [3, 11, 0],
        # side faces
        [0, 1, 5],
        [5, 4, 0],
        [1, 2, 6],
        [6, 5, 1],
        [3, 2, 6],
        [6, 7, 3],
        [0, 3, 7],
        [7, 4, 0],
        # bottom face
        [4, 15, 12],
        [5, 12, 13],
        [6, 13, 14],
        [7, 14, 15],
        [4, 12, 5],
        [5, 13, 6],
        [6, 14, 7],
        [7, 15, 4],
        # inside rectangle ('pore')
        [8, 9, 10],
        [10, 11, 8],
        [8, 9, 13],
        [13, 12, 8],
        [9, 10, 14],
        [14, 13, 9],
        [11, 10, 14],
        [14, 15, 11],
        [8, 11, 15],
        [15, 12, 8],
        [13, 14, 15],
        [15, 12, 13],
    ])

    mesh = TriangleMesh(vertices=vertices, faces=faces)
    return mesh


def test_generate_3d_mesh(triangle_mesh):
    """Test 3D mesh generation."""
    expected_fn = Path(__file__).parent / 'expected_tetra_mesh.pickle'

    np.random.seed(1234)  # set seed for reproducible clustering

    region_markers = {
        10: [0.5, 0.5, 0.5],
        20: [2.0, 2.0, 2.0],
    }

    tetra_mesh = triangle_mesh.tetrahedralize(region_markers=region_markers)

    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(tetra_mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    assert tetra_mesh.vertices.shape == expected_mesh.vertices.shape
    assert tetra_mesh.faces.shape == expected_mesh.faces.shape
    np.testing.assert_allclose(tetra_mesh.vertices, expected_mesh.vertices)
    np.testing.assert_allclose(tetra_mesh.faces, expected_mesh.faces)

    np.testing.assert_allclose(tetra_mesh.metadata['tetgen.ref'],
                               expected_mesh.metadata['tetgen.ref'])
