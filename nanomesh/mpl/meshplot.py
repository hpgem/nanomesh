from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from nanomesh.mesh import LineMesh, TriangleMesh
    from nanomesh.mesh_container import MeshContainer


def _legend_with_triplot_fix(ax: plt.Axes, **kwargs):
    """Add legend for triplot with fix that avoids duplicate labels.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to apply legend to.
    **kwargs
        Extra keyword arguments passed to `ax.legend()`.
    """
    handles, labels = ax.get_legend_handles_labels()
    # reverse to avoid blank line color
    by_label = dict(zip(reversed(labels), reversed(handles)))
    return ax.legend(by_label.values(), by_label.keys(), **kwargs)


def _legend_with_field_names_only(ax: plt.Axes, **kwargs):
    """Add legend with named fields only.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to apply legend to.

    **kwargs
        Extra keyword arguments passed to `ax.legend()`.
    """
    new_handles_labels = []
    for handle, label in zip(*ax.get_legend_handles_labels()):
        try:
            float(label)
        except ValueError:
            new_handles_labels.append((handle, label))

    return ax.legend(*zip(*new_handles_labels), **kwargs)


def plot_line_triangle(mesh: MeshContainer, **kwargs):
    """Plot line/triangle mesh together."""
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    mesh.plot_mpl('line', ax=ax1, **kwargs)
    mesh.plot_mpl('triangle', ax=ax2, **kwargs)
    return fig, (ax1, ax2)


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
                 legend: str = 'all',
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
        Extra keyword arguments passed to `.mpl.lineplot`

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
                     **kwargs) -> plt.Axes:
    """Simple line mesh plot using `matplotlib`.

    Parameters
    ----------
    ax : plt.Axes, optional
        Axes to use for plotting.
    key : str, optional
        Label of cell data item to plot, defaults to `.default_key`.
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

    for cell_data_val in np.unique(cell_data):
        vert_x, vert_y = mesh.points.T

        name = mesh.number_to_field.get(cell_data_val, cell_data_val)

        ax.triplot(vert_y,
                   vert_x,
                   triangles=mesh.cells,
                   mask=cell_data != cell_data_val,
                   label=name,
                   **kwargs)

    ax.set_title(f'{mesh._cell_type} mesh')
    ax.axis('equal')

    _legend_with_triplot_fix(ax, title=key)

    return ax
