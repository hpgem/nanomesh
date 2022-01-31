from functools import partial
from pathlib import Path

import numpy as np
from matplotlib.testing.decorators import image_comparison

from nanomesh import MeshContainer

image_comparison2 = partial(
    image_comparison,
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)


def assert_mesh_almost_equal(this, other, tol=0.0):
    """Compare two meshes."""
    this_shape = this.points.shape
    other_shape = other.points.shape
    assert this_shape == other_shape, (
        f'{this_shape=} not equal to {other_shape=}')

    if tol:
        dist = np.linalg.norm(this.points - other.points, axis=1)
        assert dist.mean() < tol, f'{dist.mean()=} {tol=}'
    else:
        assert np.allclose(this.points, other.points)

    this_cells = this.cells_dict
    other_cells = other.cells_dict

    assert this_cells.keys() == other_cells.keys()

    for key, this_cell_data in this_cells.items():
        other_cell_data = other_cells[key]
        assert np.allclose(this_cell_data, other_cell_data)


def get_expected_if_it_exists(filename: str, result: MeshContainer):
    """Read expected mesh if it exists, otherwise write result to filename."""
    path = Path(__file__).parents[1].joinpath(filename)

    if path.exists():
        expected_mesh = MeshContainer.read(path)
    else:
        result.write(path, file_format='gmsh22', binary=False)

        raise RuntimeError(f'Wrote expected mesh to {path.absolute()}')

    return expected_mesh
