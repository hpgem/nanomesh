from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .._doc import doc

if TYPE_CHECKING:
    from nanomesh import LineMesh, MeshContainer, TriangleMesh


def _get_color_cycle(colors) -> cycle:
    """Get default matplotlib color cycle.

    Returns
    -------
    itertools.cycle
        Cycles through color strings in hex format (#XXXXXX).
    """
    if not colors:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    return cycle(colors)


def _get_point(mesh, label: int, method: str = 'mean') -> Tuple[float, float]:
    """Pick middle point from mesh matching default key.

    Parameters
    ----------
    mesh : Mesh
        Input mesh
    label : int
        Input label

    Returns
    -------
    Tuple[float, float]
        (x, y) point
    """
    idx = (mesh.cell_data[mesh.default_key] == label)
    cells = mesh.cells[idx]
    points = mesh.points[np.unique(cells)]

    if method == 'middle':
        return points[len(points) // 2]  # take middle point as anchor
    else:
        return points.mean(axis=0)


def _annotate(ax: plt.Axes,
              name: str,
              xy: Tuple[float, float],
              flip_xy: bool = True):
    """Annotate point on axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes
    name : str
        Annotation text
    xy : Tuple[float, float]
        The point (x, y) to annotate
    flip_xy : bool, optional
        If true, (x,y) -> (y,x)

    Returns
    -------
    matplotlib.text.Annotation
    """
    if flip_xy:
        xy = xy[::-1]

    ax.annotate(name,
                xy,
                textcoords='offset pixels',
                xytext=(4, 4),
                color='red',
                va='bottom')


def _deduplicate_labels(
    handles_labels: Tuple[List[Any],
                          List[str]]) -> Tuple[List[Any], List[str]]:
    """Deduplicate legend handles and labels.

    Parameters
    ----------
    handles_labels : Tuple[List[Any], List[str]]
        Legend handles and labels.

    Returns
    -------
    (new_handles, new_labels) : Tuple[List[Any], List[str]]
        Deduplicated legend handles and labels
    """
    new_handles = []
    new_labels = []

    for handle, label in zip(*handles_labels):
        if label not in new_labels:
            new_handles.append(handle)
            new_labels.append(label)

    return (new_handles, new_labels)


def _legend_with_triplot_fix(ax: plt.Axes, **kwargs):
    """Add legend for triplot with fix that avoids duplicate labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to apply legend to.
    **kwargs
        These parameters are passed to :func:`matplotlib.pyplot.legend`.

    Returns
    -------
    matplotlib.legend.Legend
    """
    handles_labels = ax.get_legend_handles_labels()

    new_handles_labels = _deduplicate_labels(handles_labels)

    return ax.legend(*new_handles_labels, **kwargs)


def _legend_with_field_names_only(ax: plt.Axes,
                                  triplot_fix: bool = False,
                                  **kwargs):
    """Add legend with named fields only.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to apply legend to.
    **kwargs
        These parameters are passed to :func:`matplotlib.pyplot.legend`.

    Returns
    -------
    matplotlib.legend.Legend
    """
    handles_labels = ax.get_legend_handles_labels()

    new_handles = []
    new_labels = []

    for handle, label in zip(*handles_labels):
        try:
            float(label)
        except ValueError:
            new_handles.append(handle)
            new_labels.append(label)

    new_handles_labels = (new_handles, new_labels)

    if triplot_fix:
        new_handles_labels = _deduplicate_labels(new_handles_labels)

    return ax.legend(*new_handles_labels, **kwargs)


def _legend(ax: plt.Axes, title: str, triplot_fix: bool = False):
    """Wrapper around ax.legend with dispatch for fix.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object
    title : str
        Legend title
    triplot_fix : bool, optional
        If true, apply fix for triplot.
    """
    if triplot_fix:
        return _legend_with_triplot_fix(ax, title=title)
    else:
        return ax.legend(title=title)


@doc(cell_type='triangle',
     cell_dim=3,
     description='Shallow interface to :func:`matplotlib.pyplot.triplot`')
def triplot(ax: plt.Axes, **kwargs):
    """Plot collection of {cell_type}s.

    {description}.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Description
    x : (n, 1) numpy.ndarray
        x-coordinates of points.
    y : (n, 1) numpy.ndarray
        y-coordinates of points.
    cells : (m, {cell_dim}) numpy.ndarray
        Integer array describing the {cell_type}s.
    mask : (m, 1) numpy.ndarray, optional
        Mask for {cell_type}s.
    label : str, optional
        Label for legend.
    **kwargs
        Extra keywords arguments passed to :func:`matplotlib.pyplot.plot`

    Returns
    -------
    list of :class:`matplotlib.lines.Line2D`
        A list of lines representing the {cell_type}s and nodes.
    """
    x = kwargs.pop('x')
    y = kwargs.pop('y')
    kwargs['triangles'] = kwargs.pop('cells')
    return ax.triplot(x, y, **kwargs)


@doc(triplot,
     cell_type='line segment',
     cell_dim=3,
     description='API mimicks :func:`triplot`')
def lineplot(ax: plt.Axes,
             *,
             x: np.ndarray,
             y: np.ndarray,
             cells: np.ndarray,
             mask: np.ndarray = None,
             label: str = None,
             **kwargs):
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('markersize', 1)

    if mask is not None:
        cells = cells[~mask.squeeze()]

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(float)

    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(float)

    lines_x = np.insert(x[cells], 2, np.nan, axis=1)
    lines_y = np.insert(y[cells], 2, np.nan, axis=1)

    return ax.plot(lines_x.ravel(), lines_y.ravel(), label=label, **kwargs)


def _meshplot(mesh: LineMesh | TriangleMesh,
              ax: plt.Axes = None,
              key: str = None,
              legend: str = 'fields',
              show_labels: Sequence[int | str] = None,
              hide_labels: Sequence[int | str] = None,
              show_region_markers: bool = True,
              colors: Sequence[str] = None,
              color_map: Dict[str | int, str] = None,
              flip_xy: bool = True,
              **kwargs) -> plt.Axes:
    """Plot a :class:`nanomesh.TriangleMesh` or :class:`nanomesh.LineMesh`
    using :mod:`matplotlib`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to use for plotting.
    key : str, optional
        Label of cell data item to plot, defaults to
        :attr:`nanomesh.LineMesh.default_key` or
        :attr:`nanomesh.TriangleMEsh.default_key`.
    legend : str
        Style for the legend.
        - off : No legend
        - all : Create legend with all labels
        - fields : Create legend with field names only
        - floating : Add floating labels to plot
    show_labels : Sequence[int | str]
        List of labels or field names of cell data to show
    hide_labels : Sequence[int | str]
        List of labels or field names of cell data to hide
    show_region_markers : bool, default True
        If True, show region markers on the plot
    colors : Sequence[str]
        List of colors to cycle through
    color_map : dict
        Mapping of labels or field names to colors.
    flip_xy : bool, optional
        Flip x/y coordinates. This is sometimes necessary to combine the
        plot with other plots.
    **kwargs
        These parameters are passed to :func:`lineplot` or :func:`triplot`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    dispatch: Dict[str, Callable] = {
        'line': lineplot,
        'triangle': triplot,
    }
    plotting_func = dispatch[mesh.cell_type]

    if not ax:
        fig, ax = plt.subplots()

    if not key:
        key = mesh.default_key

    color_cycle = _get_color_cycle(colors)

    if not color_map:
        color_map = {}

    vert_x, vert_y = mesh.points.T

    if flip_xy:
        vert_x, vert_y = vert_y, vert_x

    # https://github.com/python/mypy/issues/9430
    cell_data = mesh.cell_data.get(key, mesh.zero_labels)  # type: ignore

    for cell_data_val in np.unique(cell_data):
        name = mesh.number_to_field.get(cell_data_val, cell_data_val)

        if show_labels and (name not in show_labels):
            continue

        if hide_labels and (name in hide_labels):
            continue

        color = color_map.get(name, next(color_cycle))

        plotting_func(
            ax=ax,
            x=vert_x,
            y=vert_y,
            cells=mesh.cells,
            mask=cell_data != cell_data_val,
            label=name,
            color=color,
            **kwargs,
        )

        if legend == 'floating':
            method = 'middle' if mesh.cell_type == 'triangle' else 'mean'
            xy = _get_point(mesh, cell_data_val, method=method)
            _annotate(ax, name, xy, flip_xy=flip_xy)

    if show_region_markers and mesh.region_markers:
        mark_x, mark_y = np.array([m.point for m in mesh.region_markers]).T
        ax.scatter(mark_y,
                   mark_x,
                   marker='*',
                   color='red',
                   label='Region markers')
        for marker in mesh.region_markers:
            label = marker.name if marker.name else marker.label
            ax.annotate(label,
                        marker.point[::-1],
                        textcoords='offset pixels',
                        xytext=(4, -4),
                        color='red',
                        va='top')

    ax.set_title(f'{mesh.cell_type} mesh')
    ax.axis('equal')

    # prevent double entries in legend for triangles
    triplot_fix = (mesh.cell_type == 'triangle')

    if legend == 'all':
        _legend(ax=ax, title=key, triplot_fix=triplot_fix)
    elif legend == 'fields':
        _legend_with_field_names_only(ax=ax,
                                      title=key,
                                      triplot_fix=triplot_fix)

    return ax


trianglemeshplot = _meshplot
linemeshplot = _meshplot


def linetrianglemeshplot(mesh: MeshContainer,
                         **kwargs) -> Tuple[plt.Axes, plt.Axes]:
    """Plot line/triangle mesh together.

    Parameters
    ----------
    mesh : MeshContainer
        Input mesh containing line and triangle cells.
    **kwargs
        Exstra keyword arguments passed to
        - :func:`linemeshplot`
        - :func:`trianglemeshplot`

    Returns
    -------
    Tuple(matplotlib.axes.Axes, matplotlib.axes.Axes)
        Tuple of matplotlib axes
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    line_mesh = mesh.get('line')
    linemeshplot(line_mesh, ax=ax1, **kwargs)

    triangle_mesh = mesh.get('triangle')
    trianglemeshplot(triangle_mesh, ax=ax2, **kwargs)

    return ax1, ax2
