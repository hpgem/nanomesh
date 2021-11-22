from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import triangle as tr

if TYPE_CHECKING:
    from nanomesh.mesh import TriangleMesh
    from nanomesh.mesh_container import MeshContainer


def compare_mesh_with_image(image: np.ndarray, mesh: TriangleMesh):
    """Compare mesh with image.

    Parameters
    ----------
    image : 2D array
        Image to compare mesh with
    mesh : TriangleMesh
        Triangle mesh to plot on image

    Returns
    -------
    ax : matplotlib.Axes
    """
    fig, ax = plt.subplots()

    mesh.plot_mpl(ax=ax)

    ax.imshow(image)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def simple_triangulate(points: np.ndarray,
                       *,
                       segments: np.ndarray = None,
                       regions: np.ndarray = None,
                       opts: str = '') -> MeshContainer:
    """Simple triangulation using `triangle`.

    Parameters
    ----------
    points : i,2 np.ndarray
        Vertex coordinates.
    segments : j,2 np.ndarray, optional
        Index array describing segments.
        Segments are edges whose presence in the triangulation
        is enforced (although each segment may be subdivided into smaller
        edges). Each segment is specified by listing the indices of its
        two endpoints. A closed set of segments describes a contour.
    regions : k,2 np.ndarray, optional
        Coordinates describing regions. A region is a coordinate inside
        (e.g. at the center) of a region/contour (i.e. enclosed by segments).
    opts : str, optional
        Additional options passed to `triangle.triangulate` documented here:
        https://rufat.be/triangle/API.html#triangle.triangulate

    Returns
    -------
    mesh : MeshContainer
        Triangle mesh
    """
    from nanomesh.mesh_container import MeshContainer

    triangle_dict_in = {'vertices': points}

    if segments is not None:
        triangle_dict_in['segments'] = segments

    if regions is not None:
        triangle_dict_in['regions'] = regions

    triangle_dict_out = tr.triangulate(triangle_dict_in, opts=opts)

    mesh = MeshContainer.from_triangle_dict(triangle_dict_out)

    return mesh


def pad(mesh: TriangleMesh,
        *,
        side: str,
        width: int,
        opts: str = '',
        label: int = None) -> TriangleMesh:
    """Pad a triangle mesh.

    Parameters
    ----------
    mesh : TriangleMesh
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
    new_mesh : TriangleMesh
        Padded triangle mesh.

    Raises
    ------
    ValueError
        When the value of `side` is invalid.
    """
    from nanomesh.mesh import TriangleMesh

    if label is None:
        label = mesh.labels.max() + 1

    if width == 0:
        return mesh

    top_edge, right_edge = mesh.points.max(axis=0)
    bottom_edge, left_edge = mesh.points.min(axis=0)

    if side == 'bottom':
        is_edge = mesh.points[:, 0] == bottom_edge
        corners = np.array([[bottom_edge - width, right_edge],
                            [bottom_edge - width, left_edge]])
    elif side == 'left':
        is_edge = mesh.points[:, 1] == left_edge
        corners = np.array([[bottom_edge, left_edge - width],
                            [top_edge, left_edge - width]])
    elif side == 'top':
        is_edge = mesh.points[:, 0] == top_edge
        corners = np.array([[top_edge + width, right_edge],
                            [top_edge + width, left_edge]])
    elif side == 'right':
        is_edge = mesh.points[:, 1] == right_edge
        corners = np.array([[bottom_edge, right_edge + width],
                            [top_edge, right_edge + width]])
    else:
        raise ValueError('Side must be one of `right`, `left`, `bottom`'
                         f'`top`. Got {side=}')

    edge_coords = mesh.points[is_edge]

    coords = np.vstack([edge_coords, corners])

    pad_mesh = simple_triangulate(points=coords, opts=opts)

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

    cells = pad_mesh.cells_dict['triangle'].copy()
    shape = cells.shape
    pad_cells = cells.ravel()

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
