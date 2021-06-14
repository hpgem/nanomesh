import numpy as np


def assert_mesh_almost_equal(this, other, tol=0.0):
    """Compare two meshes."""
    this_shape = this.points.shape
    other_shape = other.points.shape
    assert this_shape == other_shape, (
        f'{this_shape=} not equal to {other_shape=}')

    if tol:
        dist = np.linalg.norm(this.points - other.points, axis=1)
        assert dist.mean() < tol
    else:
        assert np.allclose(this.points, other.points)

    this_cells = this.cells_dict
    other_cells = other.cells_dict

    assert this_cells.keys() == other_cells.keys()

    for key, this_cell_data in this_cells.items():
        other_cell_data = other_cells[key]
        assert np.allclose(this_cell_data, other_cell_data)
