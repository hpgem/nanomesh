import numpy as np


def assert_mesh_almost_equal(this, other):
    """Compare two meshes."""
    # There is a minor discrepancy between the generated points between
    # Linux/Mac and Windows. Allow for some deviation.
    tolerance = 0.0025
    dist = np.linalg.norm(this.points - other.points, axis=1)
    assert dist.mean() < tolerance

    this_cells = this.cells_dict
    other_cells = other.cells_dict

    assert this_cells.keys() == other_cells.keys()

    for key, this_cell_data in this_cells.items():
        other_cell_data = other_cells[key]
        assert np.allclose(this_cell_data, other_cell_data)
