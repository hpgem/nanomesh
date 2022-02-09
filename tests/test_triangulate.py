from pathlib import Path

import numpy as np
import pytest
from helpers import get_expected_if_it_exists

from nanomesh import LineMesh, RegionMarker


@pytest.mark.parametrize('opts', (
    'pa2',
    'pAa2',
    'q30a2',
    'Aq30a2',
    'Aq30a2',
    'pAq30a',
    'pAq30a2',
))
def test_triangulate_from_line_mesh(opts):
    points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    cells = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 3]])

    region_markers = [
        RegionMarker(0, (2.5, 2.5), constraint=1),
        RegionMarker(1, (7.5, 7.5), constraint=4),
    ]

    line_mesh = LineMesh(points=points, cells=cells)
    line_mesh.add_region_markers(region_markers)

    mesh = line_mesh.triangulate(opts=opts)

    fn = Path(f'triangle/{opts}.msh')
    expected_mesh = get_expected_if_it_exists(fn, result=mesh)

    a = expected_mesh.get('triangle')
    b = mesh.get('triangle')

    np.testing.assert_equal(a.points, b.points)
    np.testing.assert_equal(a.cells, b.cells)
    np.testing.assert_equal(a.labels, b.labels)
