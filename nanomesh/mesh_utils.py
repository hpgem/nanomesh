import matplotlib.pyplot as plt
import numpy as np
import triangle as tr

from nanomesh.mesh_container import TriangleMesh


def _legend_with_triplot_fix(ax: plt.Axes):
    """Add legend for triplot with fix that avoids duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    # reverse to avoid blank line color
    by_label = dict(zip(reversed(labels), reversed(handles)))
    ax.legend(by_label.values(), by_label.keys())


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

    ax.set_title('Mesh')

    mesh.plot(ax)

    _legend_with_triplot_fix(ax)

    ax.imshow(image)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def pad(mesh: TriangleMesh,
        *,
        side: str,
        width: int,
        opts: str = '',
        label: int = None) -> TriangleMesh:
    """Pad a mesh.

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
        Description

    Raises
    ------
    ValueError
        When the value of `side` is invalid.
    """
    if label is None:
        label = mesh.unique_labels.max() + 1

    right_edge, top_edge = mesh.vertices.max(axis=0)
    left_edge, bottom_edge = mesh.vertices.min(axis=0)

    if side == 'left':
        is_edge = mesh.vertices[:, 0] == left_edge
        corners = np.array([[left_edge - width, top_edge],
                            [left_edge - width, bottom_edge]])
    elif side == 'bottom':
        is_edge = mesh.vertices[:, 1] == bottom_edge
        corners = np.array([[left_edge, bottom_edge - width],
                            [right_edge, bottom_edge - width]])
    elif side == 'right':
        is_edge = mesh.vertices[:, 0] == right_edge
        corners = np.array([[right_edge + width, top_edge],
                            [right_edge + width, bottom_edge]])
    elif side == 'top':
        is_edge = mesh.vertices[:, 1] == top_edge
        corners = np.array([[left_edge, top_edge + width],
                            [right_edge, top_edge + width]])
    else:
        raise ValueError('Side must be one of `right`, `left`, `bottom`'
                         f'`top`. Got {side=}')

    edge_coords = mesh.vertices[is_edge]

    coords = np.vstack([edge_coords, corners])

    triangle_dict_in = {'vertices': coords}
    triangle_dict_out = tr.triangulate(triangle_dict_in, opts)

    buffer_mesh = TriangleMesh.from_triangle_dict(triangle_dict_out)

    mesh_edge_index = np.argwhere(is_edge)
    buffer_edge_index = np.arange(len(mesh_edge_index)).reshape(-1, 1)
    edge_mapping = np.hstack([buffer_edge_index, mesh_edge_index]).T

    n_vertices = len(mesh.vertices)
    n_duplicate = len(edge_coords)
    n_new = len(buffer_mesh.vertices) - n_duplicate

    mesh_index = np.arange(n_vertices, n_vertices + n_new)
    buffer_index = np.arange(n_duplicate, n_duplicate + n_new)
    buffer_mapping = np.vstack([buffer_index, mesh_index])

    mapping = np.hstack([edge_mapping, buffer_mapping])

    shape = buffer_mesh.faces.shape
    buffer_faces = buffer_mesh.faces.copy().ravel()

    mask = np.in1d(buffer_faces, mapping[0, :])
    buffer_faces[mask] = mapping[1,
                                 np.searchsorted(mapping[
                                     0, :], buffer_faces[mask])]
    buffer_faces = buffer_faces.reshape(shape)

    buffer_vertices = buffer_mesh.vertices[n_duplicate:]
    buffer_labels = np.ones(len(buffer_faces)) * label

    vertices = np.vstack([mesh.vertices, buffer_vertices])
    faces = np.vstack([mesh.faces, buffer_faces])
    labels = np.hstack([mesh.metadata['labels'], buffer_labels])

    new_mesh = TriangleMesh(vertices=vertices, faces=faces, labels=labels)

    return new_mesh
