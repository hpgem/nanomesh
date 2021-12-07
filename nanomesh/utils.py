import time
from itertools import tee

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, RadioButtons, interact


class SliceViewer:
    """Simple slice viewer for volumes using matplotlib.

    Parameters
    ----------
    data : 3D np.ndarray
        Volume to display.
    update_delay : int
        Minimum delay between events in milliseconds. Reduces lag
        by limiting the Limit update rate.
    **kwargs :
        Passed to first call of `SliceViewer.update`.
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
               dpi: int = 80,
               title: str = None,
               **kwargs) -> 'plt.Axes':
    """Simple function to show an image using matplotlib.

    Parameters
    ----------
    image : 2D np.ndarray
        Image to display.
    dpi : int, optional
        DPI to render at.
    title : str, optional
        Title for the plot.
    **kwargs : dict
        Extra keyword arguments to pass to `plt.imshow`.

    Returns
    -------
    plt.Axes
        Description
    """
    kwargs.setdefault('interpolation', None)

    fig = plt.figure(dpi=dpi)
    plt.set_cmap('gray')

    ax = fig.add_subplot()
    ax.imshow(image, **kwargs)

    if title:
        plt.title(title)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax


# https://docs.python.org/3.8/library/itertools.html#itertools-recipes
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
