import logging

import itkwidgets as itkw
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

    def show_slice(self, along: str = 'x', **kwargs):
        """Show slice using `nanomesh.utils.show_slice`.

        Extra arguments are passed on.
        """
        show_slice(self.image, along=along, **kwargs)

    def show_volume(self, **kwargs):
        """Show volume using `itkwidgets`.

        Extra keyword arguments (`kwargs`) are passed to
        `itkwidgets.view`.
        """
        return itkw.view(self.image)
