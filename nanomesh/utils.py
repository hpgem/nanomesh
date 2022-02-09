from itertools import tee

import matplotlib.pyplot as plt
import numpy as np

from .mesh import TriangleMesh


# https://docs.python.org/3.8/library/itertools.html#itertools-recipes
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def compare_mesh_with_image(image: np.ndarray,
                            mesh: TriangleMesh,
                            cmap: str = None,
                            **kwargs):
    """Compare mesh with image.

    Parameters
    ----------
    image : 2D array
        Image to compare mesh with
    mesh : TriangleMesh
        Triangle mesh to plot on image
    cmap : str
        Matplotlib color map for `ax.imshow`
    **kwargs :
        Extra keyword arguments passed on to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
    """
    fig, ax = plt.subplots()

    mesh.plot_mpl(ax=ax, **kwargs)

    ax.imshow(image, cmap=cmap)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
