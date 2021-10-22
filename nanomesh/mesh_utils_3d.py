import numpy as np

from nanomesh.mesh3d import BoundingBox
from nanomesh.mesh_container import TetraMesh


def pad3d(mesh: TetraMesh,
          *,
          side: str,
          width: int,
          opts: str = '',
          label: int = None) -> TetraMesh:
    """Pad a tetra mesh.

    Parameters
    ----------
    mesh : TetraMesh
        The mesh to pad.
    side : str
        Side to pad, must be one of `left`, `right`, `top`, `bottom`.
    width : int
        Width of the padded area.
    opts : str, optional
        Optional arguments passed to `triangle.triangulate`.
    label : int, optional
        The label to assign to the padded area. If not defined, generates the
        next unique label based on the existing ones.

    Returns
    -------
    new_mesh : TetraMesh
        Padded tetrahedral mesh.

    Raises
    ------
    ValueError
        When the value of `side` is invalid.
    """
    if label is None:
        label = mesh.unique_labels.max() + 1

    if width == 0:
        return mesh

    back_edge, right_edge, top_edge = mesh.points.max(axis=0)
    front_edge, left_edge, bottom_edge = mesh.points.min(axis=0)

    bbox = BoundingBox(
        xmin=front_edge,
        xmax=back_edge,
        ymin=left_edge,
        ymax=right_edge,
        zmin=top_edge,
        zmax=bottom_edge,
    )

    if side == 'top':
        edge_col = 2
        edge_value = top_edge
        bbox.zmax = top_edge
        bbox.zmin = top_edge - width
    elif side == 'bottom':
        edge_col = 2
        edge_value = bottom_edge
        bbox.zmax = bottom_edge + width
        bbox.zmin = bottom_edge
    elif side == 'left':
        edge_col = 1
        edge_value = left_edge
        bbox.ymin = left_edge - width
        bbox.ymax = right_edge
    elif side == 'right':
        edge_col = 1
        edge_value = right_edge
        bbox.ymin = right_edge
        bbox.ymax = right_edge + width
    elif side == 'front':
        edge_col = 0
        edge_value = right_edge
        bbox.xmin = front_edge
        bbox.xmax = front_edge - width
    elif side == 'back':
        edge_col = 0
        edge_value = right_edge
        bbox.xmin = back_edge + width
        bbox.xmax = back_edge

    else:
        raise ValueError('Side must be one of `right`, `left`, `bottom`'
                         f'`top`, `front`, `back`. Got {side=}')

    is_edge = mesh.points[:, edge_col] == edge_value
    edge_coords = mesh.points[is_edge]

    ## modified bbox is the piece to be filled with tetrahedra
    return bbox

    # TODO: update this part for 3d

    coords = np.vstack([edge_coords, corners])

    pad_mesh = simple_tetrahedralize(points=coords, opts=opts)

    mesh_edge_index = np.argwhere(is_edge).flatten()
    pad_edge_index = np.arange(len(mesh_edge_index))
    edge_mapping = np.vstack([pad_edge_index, mesh_edge_index])

    n_verts = len(mesh.points)
    n_edge_verts = len(edge_coords)
    n_pad_verts = len(pad_mesh.points) - n_edge_verts

    mesh_index = np.arange(n_verts, n_verts + n_pad_verts)
    pad_index = np.arange(n_edge_verts, n_edge_verts + n_pad_verts)
    pad_mapping = np.vstack([pad_index, mesh_index])

    # mapping for the cell indices cells in `pad_mesh` to the source mesh.
    mapping = np.hstack([edge_mapping, pad_mapping])

    shape = pad_mesh.cells.shape
    pad_cells = pad_mesh.cells.copy().ravel()

    mask = np.in1d(pad_cells, mapping[0, :])
    pad_cells[mask] = mapping[1,
                              np.searchsorted(mapping[0, :], pad_cells[mask])]
    pad_cells = pad_cells.reshape(shape)

    pad_verts = pad_mesh.points[n_edge_verts:]
    pad_labels = np.ones(len(pad_cells)) * label

    # append values to source mesh
    points = np.vstack([mesh.points, pad_verts])
    cells = np.vstack([mesh.cells, pad_cells])
    labels = np.hstack([mesh.labels, pad_labels])

    new_mesh = TriangleMesh(points=points, cells=cells, labels=labels)

    return new_mesh
