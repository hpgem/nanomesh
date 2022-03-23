from __future__ import annotations

from functools import singledispatch
from itertools import tee
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .mesh import TriangleMesh


@singledispatch
def _to_opts_string(inp: Any,
                    *,
                    sep: str = '',
                    prefix: str = '',
                    defaults: dict = None) -> str:
    """Convert raw opts input to opts string for tetgen or triangle.

    Parameters
    ----------
    inp : Any
        Input object, str, dict, or None
    sep : str, optional
        Separator for parameters.
    prefix : str, optional
        Prefix for paramaters.
    defaults : dict
        Dictionary with default options.

    Returns
    -------
    opts : str
        Opts string
    """
    if defaults:
        for k, v in defaults.items():
            if v is False:
                continue
            elif v is True:
                v = ''
            if k not in inp:
                inp = f'{inp}{sep}{prefix}{k}{v}'

    return inp


@_to_opts_string.register
def _(inp: None, **kwargs) -> str:
    return _to_opts_string('', **kwargs)


@_to_opts_string.register
def _(inp: dict,
      sep: str = '',
      prefix: str = '',
      defaults: dict = None) -> str:
    if defaults:
        inp = {**defaults, **inp}

    opts_list = []

    for k, v in inp.items():
        if v is False:
            continue
        elif v is True:
            v = ''

        opts_list.append(f'{prefix}{k}{v}')

    opts = sep.join(opts_list)

    return opts


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
        Matplotlib color map for :func:`matplotlib.pyplot.imshow`
    **kwargs
        These parameters are passed on to plotting function.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots()

    mesh.plot_mpl(ax=ax, **kwargs)

    ax.imshow(image, cmap=cmap)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
