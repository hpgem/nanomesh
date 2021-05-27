import time
import warnings

import itkwidgets as itkw
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from ipywidgets import interact

try:
    import pygalmesh
except ImportError:
    pygalmesh = None


class requires:
    """Decorate functions to mark them as unavailable based if `condition` does
    not evaluate to `True`."""
    def __init__(self, *, condition, message='requires optional dependencies'):
        self.condition = condition
        self.message = message

    def __call__(self, func):
        if not self.condition:

            def dummy(*args, **kwargs):
                warnings.warn(f'`{func.__qualname__}` {self.message}.')

            return dummy
        else:
            return func


class SliceViewer:
    """Simple slice viewer for volumes using matplotlib.

    Parameters
    ----------
    data : 3D np.ndarray
        Volume to display.
    """
    def __init__(self, data: np.ndarray):
        self.fig, self.ax = plt.subplots()
        self.data = data

        self.last_update = 0.0

        self.im = self.ax.imshow(np.empty_like(data[0]))
        self.update()

    def update(self, index: int = 0, along: str = 'x'):
        """Update the image in place."""
        now = time.time()
        diff = now - self.last_update

        # Limit update rate to avoid lag
        if diff < 0.05:
            return

        if along == 'x':
            slice = self.data[:, :, index]
            xlabel, ylabel = 'y', 'z'
        elif along == 'y':
            slice = self.data[:, index, :]
            xlabel, ylabel = 'x', 'z'
        elif along == 'z':
            slice = self.data[index, ...]
            xlabel, ylabel = 'x', 'y'

        self.im.set_data(slice)
        self.ax.set_title(f'slice {index} along {along}')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.fig.canvas.draw()

        self.last_update = time.time()

    def interact(self):
        """Call interactive `ipywidgets` widget."""
        interact(self.update,
                 index=(0, max(self.data.shape)),
                 along=['x', 'y', 'z'])


def show_slice(img, overlay=None):
    if isinstance(img, np.ndarray):
        img = sitk.GetImageFromArray(img.astype('uint8'))

    if isinstance(overlay, np.ndarray):
        overlay = sitk.GetImageFromArray(overlay.astype('uint8'))

    if overlay is not None:
        img = sitk.LabelOverlay(img, overlay)

    nda = sitk.GetArrayFromImage(img)

    return SliceViewer(nda).interact()


def show_image(data, dpi=80, title=None):
    """Simple function to show an image using matplotlib.

    Parameters
    ----------
    data : np.ndarray
        Image to display.
    dpi : int, optional
        DPI to render at.
    title : None, optional
        Title for the plot.
    """
    fig = plt.figure(dpi=dpi)
    plt.set_cmap('gray')

    ax = fig.add_subplot()
    ax.imshow(data, interpolation=None)

    if title:
        plt.title(title)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()


def show_volume(data):
    return itkw.view(data)


@requires(condition=pygalmesh, message='requires pygalmesh')
def generate_mesh_from_binary_image(img, h=(1.0, 1.0, 1.0), **kwargs):
    """Generate mesh from binary image using pygalmesh.

    Parameters
    ----------
    img : 2D np.ndarray
        Input image
    h : tuple, optional
        Voxel size in x, y, z
    **kwargs
        Keyword arguments passed to `pygalmesh.generate_from_array`.

    Returns
    -------
    meshio.Mesh
        Output mesh.
    """
    img_array = sitk.GetArrayFromImage(img)
    mesh = pygalmesh.generate_from_array(img_array, h, **kwargs)
    return mesh
