from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np
import triangle as tr

from ._doc import doc
from .mesh_container import MeshContainer
from .utils import _to_opts_string

if TYPE_CHECKING:
    from nanomesh import LineMesh


@doc(prefix='Triangulate a contour mesh')
def triangulate(mesh: LineMesh,
                opts: Optional[str | dict] = None,
                default_opts: dict = None) -> MeshContainer:
    """{prefix}.

    Parameters
    ----------
    mesh : LineMesh
        Input contour mesh
    opts : str | dict, optional
        Triangulation options passed to `triangle.triangulate` documented here:
        https://rufat.be/triangle/API.html#triangle.triangulate

        Can be passed as a raw string, `opts='pAq30', or dict,
        `opts=dict('p'= True, 'A'= True, 'q'=30)`.
    default_opts : dict, optional
        Dictionary with default options. These will be merged with `opts`.

    Returns
    -------
    mesh : MeshContainer
        Triangulated 2D mesh.
    """
    opts = _to_opts_string(opts, defaults=default_opts)

    points = mesh.points
    segments = mesh.cells
    regions = [(m.point[0], m.point[1], m.label, m.constraint)
               for m in mesh.region_markers]

    segment_markers = mesh.cell_data.get('segment_markers', None)

    mesh_container = simple_triangulate(
        points=points,
        segments=segments,
        regions=regions,
        segment_markers=segment_markers,
        opts=opts,
    )

    fields = {m.label: m.name for m in mesh.region_markers if m.name}
    mesh_container.set_field_data('triangle', fields)

    return mesh_container


def simple_triangulate(points: np.ndarray,
                       *,
                       segments: np.ndarray = None,
                       regions: Sequence[Tuple[float, float, int,
                                               float, ]] = None,
                       segment_markers: np.ndarray = None,
                       opts: str = None) -> MeshContainer:
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
    opts : str | dict, optional
        Additional options passed to `triangle.triangulate` documented here:
        https://rufat.be/triangle/API.html#triangle.triangulate

        Can be passed as a raw string, `opts='pAq30', or dict,
        `opts=dict('p'= True, 'A'= True, 'q'=30)`.

    Returns
    -------
    mesh : MeshContainer
        Triangulated 2D mesh
    """
    from nanomesh import MeshContainer

    opts = _to_opts_string(opts)

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
