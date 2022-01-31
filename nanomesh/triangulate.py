from typing import Any, Dict, Sequence, Tuple

import numpy as np
import triangle as tr

from .mesh_container import MeshContainer


def simple_triangulate(points: np.ndarray,
                       *,
                       segments: np.ndarray = None,
                       regions: Sequence[Tuple[float, float, int,
                                               float]] = None,
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
    regions : list, optional
        In each row, the first two numbers are the x,y point describing a
        regions. This must be a point inside, e.g. at the center,) of a
        region or polygon (i.e. enclosed by segments).
        The third number is the label given to the region, and the fourth
        number the maximum area constraint for the region.
    opts : str, optional
        Additional options passed to `triangle.triangulate` documented here:
        https://rufat.be/triangle/API.html#triangle.triangulate

    Returns
    -------
    mesh : MeshContainer
        Triangle mesh
    """
    from nanomesh.mesh_container import MeshContainer

    triangle_dict_in: Dict['str', Any] = {'vertices': points}

    if segments is not None:
        triangle_dict_in['segments'] = segments

    if regions is not None:
        triangle_dict_in['regions'] = regions

    triangle_dict_out = tr.triangulate(triangle_dict_in, opts=opts)

    mesh = MeshContainer.from_triangle_dict(triangle_dict_out)

    return mesh
