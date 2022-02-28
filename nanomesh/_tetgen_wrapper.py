from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

from ._doc import doc
from .region_markers import RegionMarkerList

if TYPE_CHECKING:
    from .mesh import TriangleMesh
    from .mesh_container import MeshContainer


def write_smesh(filename: os.PathLike,
                mesh: TriangleMesh,
                region_markers: RegionMarkerList = None):
    """Save a mesh to a `.smesh` format (Tetgen). http://wias-
    berlin.de/software/tetgen/1.5/doc/manual/manual006.html#ff_smesh.

    Parameters
    ----------
    filename : os.PathLike
        Filename to save the data to.
    mesh : TriangleMesh
        Mesh data to be saved.
    region_markers : RegionMarkerList, optional
        Override region markers from input mesh.
    """
    if region_markers is None:
        region_markers = mesh.region_markers

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

        n_regions = len(region_markers)
        region_dim = 3

        print(f'{n_regions}', file=f)

        region_fmt = ('{:4d}' + ' {:8.2f}' * region_dim +
                      ' {label:8} {constraint:8}')

        for i, marker in enumerate(region_markers):
            constraint = marker.constraint if marker.constraint else ''
            print(region_fmt.format(i + 1,
                                    *marker.point,
                                    label=marker.label,
                                    constraint=constraint),
                  file=f)


def call_tetgen(fname: os.PathLike, opts: str = '-pAq'):
    """Call tetgen via subprocess.

    Parameters
    ----------
    fname : os.PathLike
        Location of tetgen input file ('.smesh')
    opts : str
        Command-line options passed to `tetgen`.

        More info:
        http://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html

        Some useful flags:

        - `-A`: Assigns attributes to tetrahedra in different regions.
        - `-p`: Tetrahedralizes a piecewise linear complex (PLC).
        - `-q`: Refines mesh (to improve mesh quality).
        - `-a`: Applies a maximum tetrahedron volume constraint.
    """
    import subprocess as sp
    sp.run(['tetgen', opts, fname])


@doc(prefix='Tetrahedralize a surface mesh')
def tetrahedralize(mesh: TriangleMesh, opts: str = '-pAq') -> MeshContainer:
    """{prefix}.

    Parameters
    ----------
    mesh : TriangleMesh
        Input contour mesh
    opts : str, optional
        Command-line options passed to `tetgen`.

        More info:
        http://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html

        Some useful flags:

        - `-A`: Assigns attributes to tetrahedra in different regions.
        - `-p`: Tetrahedralizes a piecewise linear complex (PLC).
        - `-q`: Refines mesh (to improve mesh quality).
        - `-a`: Applies a maximum tetrahedron volume constraint.

    Returns
    -------
    MeshContainer
        Tetrahedralized mesh.
    """
    from .mesh_container import MeshContainer

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, 'nanomesh.smesh')
        write_smesh(path, mesh)
        call_tetgen(path, opts)

        tetras = MeshContainer.read(path.with_suffix('.1.ele'))

    return MeshContainer(tetras.points,
                         tetras.cells,
                         cell_data=tetras.cell_data)
