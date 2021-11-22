from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from nanomesh.mesh_container import MeshContainer


def _legend_with_triplot_fix(ax: plt.Axes, **kwargs):
    """Add legend for triplot with fix that avoids duplicate labels.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to apply legend to.
    **kwargs
        Extra keyword arguments passed to `ax.legend`.
    """
    handles, labels = ax.get_legend_handles_labels()
    # reverse to avoid blank line color
    by_label = dict(zip(reversed(labels), reversed(handles)))
    ax.legend(by_label.values(), by_label.keys(), **kwargs)


def plot_line_triangle(mesh: MeshContainer, label: str, **kwargs):
    """Plot line/triangle mesh together."""
    assert set(mesh.cells_dict.keys()) == {'line', 'triangle'}
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    mesh.plot_mpl('line', ax=ax1, label=label, **kwargs)
    mesh.plot_mpl('triangle', ax=ax2, label=label, **kwargs)
    return fig, (ax1, ax2)
