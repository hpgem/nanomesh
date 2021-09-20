import matplotlib.pyplot as plt
import numpy as np

from nanomesh.mesh_container import TriangleMesh


def _legend_with_triplot_fix(ax: plt.Axes):
    """Add legend for triplot with fix that avoids duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    # reverse to avoid blank line color
    by_label = dict(zip(reversed(labels), reversed(handles)))
    ax.legend(by_label.values(), by_label.keys())


def compare_mesh_with_image(image: np.ndarray, mesh: TriangleMesh):
    """Compare mesh with image.

    Parameters
    ----------
    image : 2D array
        Image to compare mesh with
    mesh : TriangleMesh
        Triangle mesh to plot on image

    Returns
    -------
    ax : matplotlib.Axes
    """
    fig, ax = plt.subplots()

    ax.set_title('Mesh')

    mesh.plot(ax)

    _legend_with_triplot_fix(ax)

    ax.imshow(image)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
