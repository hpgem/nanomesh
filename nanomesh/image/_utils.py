import time

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, RadioButtons, interact


class SliceViewer:
    """Simple slice viewer for volumes using :mod:`matplotlib`.

    Parameters
    ----------
    data : (i,j,k) numpy.ndarray
        Volume to display.
    update_delay : int
        Minimum delay between events in milliseconds. Reduces lag
        by limiting the Limit update rate.
    **kwargs
        These parametes are passed to first call of
        :meth:`SliceViewer.update`.
    """

    def __init__(
        self,
        data: np.ndarray,
        update_delay: int = 50,
        **kwargs,
    ):
        self.fig, self.ax = plt.subplots()
        self.data = data
        self.update_delay = update_delay / 1000

        self.max_vals = dict(zip('zyx', np.array(data.shape) - 1))
        self.labels = {
            'x': ('y', 'z'),
            'y': ('x', 'z'),
            'z': ('x', 'y'),
        }

        self.last_update = 0.0

        # Enable direct specification of slice, i.e. x=123
        for along in 'xyz':
            if along in kwargs:
                kwargs['along'] = along
                kwargs['index'] = kwargs[along]
                break

        along = kwargs.get('along', 'x')
        init_max_val = self.max_vals[along]
        init_val = kwargs.get('index', int(init_max_val / 2))

        self.int_slider = IntSlider(value=init_val, min=0, max=init_max_val)
        self.radio_buttons = RadioButtons(options=('x', 'y', 'z'), value=along)

        self.im = self.ax.imshow(data[0], interpolation=None)
        self.im.set_clim(vmin=data.min(), vmax=data.max())
        self.update(index=init_val, along=along)

    def get_slice(self, *, index: int, along: str):
        """Get slice associated with index along given axes."""
        if along == 'x':
            return self.data[:, :, index]
        elif along == 'y':
            return self.data[:, index, :]
        elif along == 'z':
            return self.data[index, ...]
        else:
            raise ValueError('`along` must be one of `x`,`y`,`z`')

    def update(self, index: int, along: str):
        """Update the image in place."""
        now = time.time()
        diff = now - self.last_update

        if diff < self.update_delay:
            return

        max_val = self.max_vals[along]
        xlabel, ylabel = self.labels[along]
        index = min(index, max_val)
        self.int_slider.max = max_val

        slice = self.get_slice(along=along, index=index)

        top, right = slice.shape

        self.im.set_data(slice)
        self.im.set_extent((0, right, 0, top))

        self.ax.set_title(f'slice {index} along {along}')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.fig.canvas.draw()

        self.last_update = time.time()

    def interact(self):
        """Call interactive `ipywidgets` widget."""
        interact(self.update, index=self.int_slider, along=self.radio_buttons)


def show_image(image,
               *,
               ax: plt.Axes = None,
               title: str = None,
               **kwargs) -> 'plt.Axes':
    """Simple function to plot an image using :mod:`matplotlib`.

    Parameters
    ----------
    image : (i,j) numpy.ndarray
        Image to display.
    ax : matplotlib.axes.Axes, optional
        Axes to use for plotting.
    title : str, optional
        Title for the plot.
    **kwargs
        These parameters are passed to `plt.imshow`.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    kwargs.setdefault('interpolation', None)

    if not ax:
        fig, ax = plt.subplots()

    ax.imshow(image, **kwargs)

    if title:
        plt.title(title)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax
