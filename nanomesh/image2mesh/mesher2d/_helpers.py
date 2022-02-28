from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from scipy.spatial.distance import cdist

from nanomesh._doc import doc
from nanomesh.region_markers import RegionMarker, RegionMarkerList

if TYPE_CHECKING:
    from nanomesh.mesh import LineMesh


@doc(prefix='Pad a triangle mesh (2D)')
def pad(
    mesh: LineMesh,
    *,
    side: str,
    width: int,
    label: int = None,
    name: str = None,
) -> LineMesh:
    """{prefix}.

    Parameters
    ----------
    mesh : LineMesh
        The mesh to pad.
    side : str
        Side to pad, must be one of `left`, `right`, `top`, `bottom`.
    width : int
        Width of the padded area.
    label : int, optional
        The label to assign to the padded area. If not defined, generates the
        next unique label based on the existing ones.
    name : str, optional
        Name of the added region. Note that in case of conflicts, the `label`
        takes presedence over the `name`.

    Returns
    -------
    new_mesh : LineMesh
        Padded line mesh.

    Raises
    ------
    ValueError
        When the value of `side` is invalid.
    """
    labels = mesh.region_markers.labels
    names = mesh.region_markers.names

    if (label in labels) and (name is None):
        name = [m.name for m in mesh.region_markers if m.label == label][0]

    if name and (name in names) and (label is None):
        label = [m.label for m in mesh.region_markers if m.name == name][0]

    if label is None:
        label = max(max(labels) + 1, 2) if labels else 2

    if width == 0:
        return mesh

    bottom_edge, right_edge = mesh.points.max(axis=0)
    top_edge, left_edge = mesh.points.min(axis=0)

    if side == 'top':
        corners = np.array([
            [top_edge, right_edge],
            [top_edge - width, right_edge],
            [top_edge - width, left_edge],
            [top_edge, left_edge],
        ])
    elif side == 'left':
        corners = np.array([
            [top_edge, left_edge],
            [top_edge, left_edge - width],
            [bottom_edge, left_edge - width],
            [bottom_edge, left_edge],
        ])
    elif side == 'bottom':
        corners = np.array([
            [bottom_edge, right_edge],
            [bottom_edge + width, right_edge],
            [bottom_edge + width, left_edge],
            [bottom_edge, left_edge],
        ])
    elif side == 'right':
        corners = np.array([
            [top_edge, right_edge],
            [top_edge, right_edge + width],
            [bottom_edge, right_edge + width],
            [bottom_edge, right_edge],
        ])
    else:
        raise ValueError('Side must be one of `right`, `left`, `bottom`'
                         f'`top`. Got {side=}')

    all_points = mesh.points

    corner_idx = np.argwhere(cdist(corners, all_points) == 0)

    if len(corner_idx) < len(corners):
        # Add missing corners and add them where necessary
        missing_corners = np.delete(corners, corner_idx[:, 0], axis=0)
        all_points = np.vstack([all_points, missing_corners])
        corner_idx = np.argwhere(cdist(corners, all_points) == 0)

    R = corner_idx[:, 1].tolist()
    additional_segments = list(zip(R, R[1:] + R[:1]))
    cells = np.vstack([mesh.cells, additional_segments])

    center = corners.mean(axis=0)
    region_markers = RegionMarkerList(
        (*mesh.region_markers, RegionMarker(label, center, name)))

    segment_markers = mesh.cell_data.get(
        'segment_markers',
        np.zeros(len(mesh.cells)),
    )
    segment_markers = append_to_segment_markers(
        segment_markers,
        additional_segments,
    )

    new_mesh = mesh.__class__(
        points=all_points,
        cells=cells,
        region_markers=region_markers,
        segment_markers=segment_markers,
    )

    return new_mesh


def generate_segment_markers(segments: List[np.ndarray],
                             ones: bool = False) -> np.ndarray:
    """Generate array of sequential markers for segments.

    Parameters
    ----------
    segments : List[numpy.ndarray]
        List of segment markers
    ones : bool, optional
        Assign the label (1) to all segments

    Returns
    -------
    segment_markers : numpy.ndarray
    """
    if ones:
        n_items = sum(len(segment) for segment in segments)
        return np.ones(n_items, dtype=int)
    else:
        return np.hstack([
            np.ones(len(segment), dtype=int) * (i + 1)
            for i, segment in enumerate(segments)
        ])


def append_to_segment_markers(segment_markers: np.ndarray,
                              segments: List[np.ndarray],
                              same_label: bool = False) -> np.ndarray:
    """Append sequential markers to existing array of segment markers.

    Parameters
    ----------
    segment_markers : numpy.ndarray
        List of existing markers
    segments : List[numpy.ndarray]
        List of segments to label sequentially and append to segment markers
    same_label : bool, optional
        Assign the next available integer label to all additional segments

    Returns
    -------
    segment_markers : numpy.ndarray
    """
    offset = segment_markers.max() + 1

    if same_label:
        additional_markers = np.ones(len(segments)) * offset
    else:
        additional_markers = [i + offset for i in range(len(segments))]

    return np.hstack([segment_markers, additional_markers])
