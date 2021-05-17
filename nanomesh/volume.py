import logging

import meshio
import numpy as np
import SimpleITK as sitk

from .utils import show_slice

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

    def show_slice(self, along: str = 'x', overlay=None, **kwargs):
        """Show slice using `nanomesh.utils.show_slice`.

        Extra arguments are passed on.
        """
        show_slice(self.image, along=along, overlay=overlay, **kwargs)

    def show_volume(self, renderer='ipyvolume', **kwargs):
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
        import pygalmesh
        mesh = pygalmesh.generate_from_array(self.array_view, h, **kwargs)
        return mesh
