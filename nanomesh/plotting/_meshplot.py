from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Any, Generator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from nanomesh.mesh import LineMesh, TriangleMesh
    from nanomesh.mesh_container import MeshContainer


def _get_color_cycle() -> Generator[str, None, None]:
    """Get default matplotlib color cycle.

    Yields
    ------
    Generator[str]
        Generates color string in hex format (#XXXXXX).
    """
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'])
    while True:
        yield next(color_cycle)['color']


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
    ax : plt.Axes
        Matplotlib axes to apply legend to.
    **kwargs
        Extra keyword arguments passed to `ax.legend()`.
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
    ax : plt.Axes
        Matplotlib axes to apply legend to.

    **kwargs
        Extra keyword arguments passed to `ax.legend()`.
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


def line_triangle_plot(mesh: MeshContainer,
                       **kwargs) -> Tuple[plt.Axes, plt.Axes]:
    """Plot line/triangle mesh together.

    Parameters
    ----------
    mesh : MeshContainer
        Input mesh containing line and triangle cells.
    **kwargs
        Exstra keyword arguments passed to
        - .linemeshplot()
        - .trianglemeshplot()

    Returns
    -------
    Tuple(plt.Axes, plt.Axes)
        Tuple of matplotlib axes
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    mesh.plot_mpl('line', ax=ax1, **kwargs)
    mesh.plot_mpl('triangle', ax=ax2, **kwargs)
    return ax1, ax2


def lineplot(ax: plt.Axes,
             *,
             x: np.ndarray,
             y: np.ndarray,
             lines: np.ndarray,
             mask: np.ndarray = None,
             label: str = None,
             **kwargs):
    """Plot collection of lines, similar to `ax.triplot`.

    Parameters
    ----------
    ax : plt.Axes
        Description
    x : (n, 1) np.ndarray
        x-coordinates of points.
    y : (n, 1) np.ndarray
        y-coordinates of points.
    lines : (m, 2) np.ndarray
        Integer array describing the connected lines.
    mask : (m, 1) np.ndarray, optional
        Mask for line segments.
    label : str, optional
        Label for legend.
    **kwargs : dict
        Extra keywords arguments passed to `ax.plot`

    Returns
    -------
    list of `matplotlib.lines.Line2D`
        A list of lines representing the plotted data.
    """
    kwargs.setdefault('marker', '.')

    if mask is not None:
        lines = lines[~mask.squeeze()]

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(float)

    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(float)

    lines_x = np.insert(x[lines], 2, np.nan, axis=1)
    lines_y = np.insert(y[lines], 2, np.nan, axis=1)

    return ax.plot(lines_x.ravel(), lines_y.ravel(), label=label, **kwargs)


def linemeshplot(mesh: LineMesh,
                 ax: plt.Axes = None,
                 key: str = None,
                 legend: str = 'fields',
                 **kwargs) -> plt.Axes:
    """Simple line mesh plot using `matplotlib`.

    Parameters
    ----------
    ax : plt.Axes, optional
        Axes to use for plotting.
    key : str, optional
        Label of cell data item to plot, defaults to `.default_key`.
    legend : str
        Style for the legend.
        - off : No legend
        - all : Create legend with all labels
        - fields : Create legend with field names only
        - floating : Add floating labels to plot
    **kwargs
        Extra keyword arguments passed to `.plotting.lineplot`

    Returns
    -------
    plt.Axes
    """
    if not ax:
        fig, ax = plt.subplots()

    if key is None:
        key = mesh.default_key

    # https://github.com/python/mypy/issues/9430
    cell_data = mesh.cell_data.get(key, mesh.zero_labels)  # type: ignore

    for cell_data_val in np.unique(cell_data):
        vert_x, vert_y = mesh.points.T

        name = mesh.number_to_field.get(cell_data_val, cell_data_val)

        lineplot(
            ax,
            x=vert_y,
            y=vert_x,
            lines=mesh.cells,
            mask=cell_data != cell_data_val,
            label=name,
            **kwargs,
        )

        if legend == 'floating':
            idx = (mesh.cell_data[mesh.default_key] == cell_data_val)
            cells = mesh.cells[idx]
            points = mesh.points[np.unique(cells)]
            point = points.mean(axis=0)

            ax.annotate(name,
                        point[::-1],
                        textcoords='offset pixels',
                        xytext=(4, 4),
                        color='red',
                        va='bottom')

    if mesh.region_markers:
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

    ax.set_title(f'{mesh._cell_type} mesh')
    ax.axis('equal')

    if legend == 'all':
        ax.legend(title=key)
    elif legend == 'fields':
        _legend_with_field_names_only(ax=ax, title=key)

    return ax


def trianglemeshplot(mesh: TriangleMesh,
                     ax: plt.Axes = None,
                     key: str = None,
                     legend: str = 'fields',
                     flip_xy: bool = False,
                     **kwargs) -> plt.Axes:
    """Simple line mesh plot using `matplotlib`.

    Parameters
    ----------
    mesh : TriangleMesh
        Input triangle mesh.
    ax : plt.Axes, optional
        Axes to use for plotting.
    key : str, optional
        Label of cell data item to plot, defaults to `.default_key`.
    legend : str, optional
        Style for the legend.
        - off : No legend
        - all : Create legend with all labels
        - fields : Create legend with field names only
        - floating : Add floating labels to plot
    flip_xy : bool, optional
        Flip x/y coordinates. This is sometimes necessary to combine the
        plot with other plots.
    **kwargs
        Extra keyword arguments passed to `ax.triplot`

    Returns
    -------
    plt.Axes
    """
    if not ax:
        fig, ax = plt.subplots()

    if not key:
        key = mesh.default_key

    # https://github.com/python/mypy/issues/9430
    cell_data = mesh.cell_data.get(key, mesh.zero_labels)  # type: ignore

    # control color cycle to avoid skipping colors in `ax.triplot`
    color_cycle = _get_color_cycle()

    for cell_data_val in np.unique(cell_data):
        vert_x, vert_y = mesh.points.T

        if flip_xy:
            vert_x, vert_y = vert_y, vert_x

        name = mesh.number_to_field.get(cell_data_val, cell_data_val)

        ax.triplot(vert_y,
                   vert_x,
                   triangles=mesh.cells,
                   mask=cell_data != cell_data_val,
                   label=name,
                   color=next(color_cycle),
                   **kwargs)

        if legend == 'floating':
            idx = (mesh.cell_data[mesh.default_key] == cell_data_val)
            cells = mesh.cells[idx]
            points = mesh.points[np.unique(cells)]

            x, y = points[len(points) // 2]  # take middle point as anchor

            if flip_xy:
                x, y = y, x

            ax.annotate(name, (y, x),
                        textcoords='offset pixels',
                        xytext=(4, 4),
                        color='red',
                        va='bottom')

    ax.set_title(f'{mesh._cell_type} mesh')
    ax.axis('equal')

    if legend == 'all':
        _legend_with_triplot_fix(ax=ax, title=key)
    elif legend == 'fields':
        _legend_with_field_names_only(ax=ax, triplot_fix=True, title=key)

    return ax
