import pickle
from pathlib import Path

import numpy as np
import pygalmesh

from nanomesh.structures import XDIM, YDIM, ZDIM, Pore3D


def assert_mesh_almost_equal(this, other):
    """Compare two meshes."""
    assert np.allclose(this.points, other.points)

    this_cells = this.cells_dict
    other_cells = other.cells_dict

    assert this_cells.keys() == other_cells.keys()

    for key, this_cell_data in this_cells.items():
        other_cell_data = other_cells[key]
        assert np.allclose(this_cell_data, other_cell_data)


def test_full_periodic_1domain():
    """Test whether Pore3D gives the expected result with pygalmesh."""
    mesh = pygalmesh.generate_periodic_mesh(
        Pore3D(),
        [0, 0, 0, XDIM, YDIM, ZDIM],
        max_cell_circumradius=0.025,
        min_facet_angle=30,
        max_radius_surface_delaunay_ball=0.025,
        max_facet_distance=0.025,
        max_circumradius_edge_ratio=2,
        number_of_copies_in_output=1,
        make_periodic=True,
        exude=False,
        perturb=False,
        odt=False,
        lloyd=False,
        verbose=True,
        seed=1234,
    )

    expected_fn = Path(__file__).parent / 'full_periodic_1domain.pickle'

    if expected_fn.exists():
        with open(expected_fn, 'rb') as f:
            expected_mesh = pickle.load(f)
    else:
        with open(expected_fn, 'wb') as f:
            pickle.dump(mesh, f)

        raise RuntimeError(f'Wrote expected mesh to {expected_fn.absolute()}')

    assert_mesh_almost_equal(mesh, expected_mesh)


def test_full_periodic_2domain():
    pass
