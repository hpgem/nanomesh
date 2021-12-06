from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nanomesh.utils import pairwise

from ..region_markers import RegionMarker
from .bounding_box import BoundingBox

if TYPE_CHECKING:
    from nanomesh.mesh import TriangleMesh


def pad(mesh: TriangleMesh,
        *,
        side: str,
        width: int,
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
    label : int, optional
        The label to assign to the padded area. If not defined, generates the
        next unique label based on the existing ones.

    Returns
    -------
    new_mesh : TriangleMesh
        Padded tetrahedral mesh.

    Raises
    ------
    ValueError
        When the value of `side` is invalid.
    """
    if label is None:
        label = mesh.labels.max() + 1

    if width == 0:
        return mesh

    bbox = BoundingBox.from_points(mesh.points)

    if side == 'top':
        edge_col = 2
        edge_value = bbox.zmax
        extra_coords = np.array([
            [bbox.xmin, bbox.ymin, bbox.zmax + width],
            [bbox.xmin, bbox.ymax, bbox.zmax + width],
            [bbox.xmax, bbox.ymin, bbox.zmax + width],
            [bbox.xmax, bbox.ymax, bbox.zmax + width],
        ])
        column_order = (0, 1, 1, 0)
    elif side == 'bottom':
        edge_col = 2
        edge_value = bbox.zmin
        extra_coords = np.array([
            [bbox.xmin, bbox.ymin, bbox.zmin - width],
            [bbox.xmin, bbox.ymax, bbox.zmin - width],
            [bbox.xmax, bbox.ymin, bbox.zmin - width],
            [bbox.xmax, bbox.ymax, bbox.zmin - width],
        ])
        column_order = (0, 1, 1, 0)
    elif side == 'left':
        edge_col = 1
        edge_value = bbox.ymin
        extra_coords = np.array([
            [bbox.xmin, bbox.ymin - width, bbox.zmin],
            [bbox.xmin, bbox.ymin - width, bbox.zmax],
            [bbox.xmax, bbox.ymin - width, bbox.zmin],
            [bbox.xmax, bbox.ymin - width, bbox.zmax],
        ])
        column_order = (0, 2, 2, 0)
    elif side == 'right':
        edge_col = 1
        edge_value = bbox.ymax
        extra_coords = np.array([
            [bbox.xmin, bbox.ymax + width, bbox.zmin],
            [bbox.xmin, bbox.ymax + width, bbox.zmax],
            [bbox.xmax, bbox.ymax + width, bbox.zmin],
            [bbox.xmax, bbox.ymax + width, bbox.zmax],
        ])
        column_order = (0, 2, 2, 0)
    elif side == 'front':
        edge_col = 0
        edge_value = bbox.xmin
        extra_coords = np.array([
            [bbox.xmin - width, bbox.ymin, bbox.zmin],
            [bbox.xmin - width, bbox.ymin, bbox.zmax],
            [bbox.xmin - width, bbox.ymax, bbox.zmin],
            [bbox.xmin - width, bbox.ymax, bbox.zmax],
        ])
        column_order = (1, 2, 2, 1)
    elif side == 'back':
        edge_col = 0
        edge_value = bbox.xmax
        extra_coords = np.array([
            [bbox.xmax + width, bbox.ymin, bbox.zmin],
            [bbox.xmax + width, bbox.ymin, bbox.zmax],
            [bbox.xmax + width, bbox.ymax, bbox.zmin],
            [bbox.xmax + width, bbox.ymax, bbox.zmax],
        ])
        column_order = (1, 2, 2, 1)
    else:
        raise ValueError('Side must be one of `right`, `left`, `bottom`'
                         f'`top`, `front`, `back`. Got {side=}')

    n_points = len(mesh.points)
    points = np.vstack([mesh.points, extra_coords])

    new_triangles = [
        np.array((0, 1, 2)) + n_points,
        np.array((3, 1, 2)) + n_points,
    ]

    for corner, col in zip(extra_coords, column_order):
        connect_to = np.argwhere((points[:, edge_col] == edge_value)
                                 & (points[:, col] == corner[col]))

        additional_points = np.argwhere(
            extra_coords[:, col] == corner[col]) + n_points

        first, last = additional_points
        first_point = points[first]

        sorted_by_distance = np.argsort(
            np.linalg.norm(first_point - points[connect_to].squeeze(), axis=1))
        connect_to = connect_to[sorted_by_distance]
        connect_to = np.vstack([connect_to, last]).squeeze()

        for pair in pairwise(connect_to):
            tri = np.hstack((first, pair))
            new_triangles.append(tri)

    new_triangles = np.array(new_triangles).squeeze()

    cells = np.vstack([mesh.cells, new_triangles])

    new_mesh = mesh.__class__(
        points=points,
        cells=cells,
        region_markers=mesh.region_markers,
    )

    # add marker for new region
    center = extra_coords.mean(axis=0)
    center[col] = (center[col] + edge_value) / 2

    new_mesh.add_region_marker(RegionMarker(label, center))

    return new_mesh
