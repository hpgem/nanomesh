import matplotlib.pyplot as plt
import numpy as np


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

    lines_x = np.insert(x[lines], 2, np.nan, axis=1)
    lines_y = np.insert(y[lines], 2, np.nan, axis=1)
    return ax.plot(lines_x.ravel(), lines_y.ravel(), label=label, **kwargs)
