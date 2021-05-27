import logging

import meshio
import numpy as np
import SimpleITK as sitk

from .plane import Plane
from .utils import requires, show_slice

try:
    import pygalmesh
except ImportError:
    pygalmesh = None

logger = logging.getLogger(__name__)


class Volume:
    def __init__(self, image):

        self.image = image

    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.array_view.shape})'

    @property
    def array_view(self):
        """Return a view of the data as a numpy array."""
        return sitk.GetArrayViewFromImage(self.image)

    def to_array(self):
        return sitk.GetArrayFromImage(self.image)

    @classmethod
    def from_array(cls, array):
        image = sitk.GetImageFromArray(array)
        return cls(image)

    @classmethod
    def from_image(cls, image):
        return cls(image)

    @classmethod
    def load(cls, filename: str) -> 'Volume':
        """Load the data.

        Supported filetypes: `.npy`

        Parameters
        ----------
        filename : str
            Name of the file to load.

        Returns
        -------
        Volume
            Instance of this class.
        """
        array = np.load(filename)
        return cls.from_array(array)

    def apply(self, function, **kwargs) -> 'Volume':
        """Apply function to `.image` and return new instance of `Volume`.

        Parameters
        ----------
        function : callable
            Function to apply to `self.image`.
        **kwargs
            Keyword arguments to pass to `function`.

        Returns
        -------
        Volume
            New instance of `Volume`.
        """
        new_image = function(self.image, **kwargs)
        return Volume(new_image)

    def apply_np(self, function, **kwargs) -> 'Volume':
        """Apply function to `.array_view` and return new instance of `Volume`.

        Parameters
        ----------
        function : callable
            Function to apply to `self.array_view`.
        **kwargs
            Keyword arguments to pass to `function`.

        Returns
        -------
        Volume
            New instance of `Volume`.
        """
        new_image = function(self.array_view, **kwargs)
        return Volume.from_array(new_image)

    def show_slice(self, overlay=None, **kwargs):
        """Show slice using `nanomesh.utils.show_slice`.

        Extra arguments are passed on.
        """
        show_slice(self.image, overlay=overlay, **kwargs)

    def show_volume(self, renderer='itkwidgets', **kwargs):
        """Show volume using `itkwidgets` or `ipyvolume`.

        Extra keyword arguments (`kwargs`) are passed to
        `itkwidgets.view` or `ipyvolume.quickvolshow`.
        """
        if renderer in ('ipyvolume', 'ipv'):
            import ipyvolume as ipv
            return ipv.quickvolshow(self.array_view, **kwargs)
        elif renderer in ('itkwidgets', 'itk', 'itkw'):
            import itkwidgets as itkw
            return itkw.view(self.image)
        else:
            raise ValueError(f'No such renderer: {renderer!r}')

    @requires(condition=pygalmesh, message='requires pygalmesh')
    def generate_mesh(self, h=(1.0, 1.0, 1.0), **kwargs) -> 'meshio.Mesh':
        """Generate mesh from binary image.

        Parameters
        ----------
        h : tuple, optional
            ?
        **kwargs
            Description?

        Returns
        -------
        meshio.Mesh
            Mesh representation of volume.
        """
        mesh = pygalmesh.generate_from_array(self.array_view, h, **kwargs)
        return mesh

    def select_plane(self,
                     x: int = None,
                     y: int = None,
                     z: int = None) -> 'Plane':
        """Select a slice in the volume. Either `x`, `y` or `z` must be
        specified.

        Parameters
        ----------
        x : int, optional
            Index along the x-axis
        y : int, optional
            Index along the y-axis
        z : int, optional
            Index along the z-axis

        Returns
        -------
        Plane
            Return 2D plane representation.

        Raises
        ------
        ValueError
            If none of the `x`, `y`, or `z` arguments have been specified
        """
        if x is not None:
            slice = np.s_[:, :, x]
        elif y is not None:
            slice = np.s_[:, y, :]
        elif z is not None:
            slice = np.s_[z, ...]
        else:
            raise ValueError(
                'One of the arguments `x`, `y`, or `z` must be specified.')

        array = self.array_view[slice]
        return Plane.from_array(array)
