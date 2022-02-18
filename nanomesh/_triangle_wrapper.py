from typing import Any, Dict, Sequence, Tuple

import numpy as np
import triangle as tr

from .mesh_container import MeshContainer


def triangulate(points: np.ndarray,
                *,
                segments: np.ndarray = None,
                regions: Sequence[Tuple[float, float, int, float, ]] = None,
                segment_markers: np.ndarray = None,
                opts: str = '') -> MeshContainer:
    """Simple triangulation using :mod:`triangle`.

    Parameters
    ----------
    points : (i,2) numpy.ndarray
        Vertex coordinates.
    segments : (j,2) numpy.ndarray, optional
        Index array describing segments.
        Segments are edges whose presence in the triangulation
        is enforced (although each segment may be subdivided into smaller
        edges). Each segment is specified by listing the indices of its
        two endpoints. A closed set of segments describes a contour.
    regions : list, optional
        In each row, the first two numbers are the x,y point describing a
        regions. This must be a point inside, e.g. at the center,) of a
        region or polygon (i.e. enclosed by segments).
        The third number is the label given to the region, and the fourth
        number the maximum area constraint for the region.
    segment_markers : (j,1) numpy.ndarray, optional
        Array with labels for segments.
    opts : str, optional
        Additional options passed to `triangle.triangulate` documented here:
        https://rufat.be/triangle/API.html#triangle.triangulate

    Returns
    -------
    mesh : MeshContainer
        Triangulated 2D mesh
    """
    from nanomesh import MeshContainer

    triangle_dict_in: Dict['str', Any] = {'vertices': points}

    if segments is not None:
        triangle_dict_in['segments'] = segments

    if regions is not None:
        triangle_dict_in['regions'] = regions

    if segment_markers is not None:
        triangle_dict_in['segment_markers'] = segment_markers

    triangle_dict_out = tr.triangulate(triangle_dict_in, opts=opts)

    mesh = MeshContainer.from_triangle_dict(triangle_dict_out)

    return mesh
