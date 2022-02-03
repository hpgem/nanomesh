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
