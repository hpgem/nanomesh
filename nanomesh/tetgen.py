import os
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from .mesh_container import TriangleMesh


def write_smesh(filename: os.PathLike,
                mesh: TriangleMesh,
                region_markers: List[Tuple[int, np.ndarray]] = None):
    """Save a mesh to a `.smesh` format (Tetgen). http://wias-
    berlin.de/software/tetgen/1.5/doc/manual/manual006.html#ff_smesh.

    Parameters
    ----------
    filename : os.PathLike
        Filename to save the data to.
    mesh : TriangleMesh
        Mesh data to be saved.
    region_markers : list, optional
        Coordinates of region markers.
    """
    if region_markers is None:
        region_markers = []

    path = Path(filename)
    with path.open('w') as f:
        n_nodes, n_dim = mesh.points.shape
        n_attrs = 0
        node_markers = 0

        print(f'{n_nodes} {n_dim} {n_attrs} {node_markers}', file=f)

        node_fmt = '{:4d}' + ' {:8.2f}' * n_dim

        for i, node in enumerate(mesh.points):
            print(node_fmt.format(i + 1, *node), file=f)

        n_facets, n_corners = mesh.cells.shape
        facet_markers = 0

        print(f'{n_facets} {facet_markers}', file=f)

        facet_fmt = '{:4d}' + ' {:8d}' * n_corners

        for facet in mesh.cells + 1:  # tetgen uses 1-indexing
            print(facet_fmt.format(n_corners, *facet), file=f)

        # TODO, store holes in TriangleMesh?
        n_holes = 0
        hole_dim = 3

        print(f'{n_holes}', file=f)
        holes: Tuple[Any, ...] = ()

        hole_fmt = '{:4d}' + ' {:8.2f}' * hole_dim

        for i, hole in enumerate(holes):
            print(hole_fmt.format(i + 1, *hole), file=f)

        # TODO, store regions in TriangleMesh?
        n_regions = len(region_markers)
        region_dim = 3

        print(f'{n_regions}', file=f)

        region_fmt = '{:4d}' + ' {:8.2f}' * region_dim + ' {label:8}'
        # Define region numer, region attributes

        for i, (label, coord) in enumerate(region_markers):
            print(region_fmt.format(i + 1, *coord, label=label), file=f)


def tetrahedralize(fname: os.PathLike, opts: str = '-pAq1.2'):
    """Tetrahedralize a surface mesh.

    Parameters
    ----------
    fname : os.PathLike
        Description
    opts : str
        Command-line options passed to `tetgen`.

        More info:
        http://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html
    """
    # -A: Assigns attributes to tetrahedra in different regions.
    # -p: Tetrahedralizes a piecewise linear complex (PLC).
    # -q: Refines mesh (to improve mesh quality).
    # -a: Applies a maximum tetrahedron volume constraint.

    import subprocess as sp
    sp.run(f'tetgen {opts} {fname}')
