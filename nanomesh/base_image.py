import os
from typing import Callable, Union

import numpy as np


class BaseImage:
    """Data class for image data.

    Parameters
    ----------
    image : np.array
        N-dimensional numpy array containing image data.
    """
    def __init__(self, image: np.ndarray):
        self.image = image

    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.image.shape})'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.all(other.image == self.image)
        elif isinstance(other, np.ndarray):
            return np.all(other == self.image)
        else:
            return False

    def to_sitk_image(self):
        """Return instance of `SimpleITK.Image` from `.image`."""
        import SimpleITK as sitk
        return sitk.GetImageFromArray(self.image)

    @classmethod
    def from_sitk_image(cls, sitk_image) -> 'BaseImage':
        """Return instance from `SimpleITK.Image`."""
        import SimpleITK as sitk
        image = sitk.GetArrayFromImage(sitk_image)
        return cls(image)

    def save(self, filename: os.PathLike):
        """Save the data. Supported filetypes: `.npy`.

        Parameters
        ----------
        filename : Pathlike
            Name of the file to save to.
        """
        np.save(filename, self.image)

    @classmethod
    def load(cls, filename: os.PathLike, **kwargs):
        """Load the data. Supported filetypes: `.npy`.

        Parameters
        ----------
        filename : Pathlike
            Name of the file to load.
        **kwargs : dict
            Extra keyword arguments passed to `np.load`.
        """
        image = np.load(filename, **kwargs)
        return cls(image)

    def apply(self, function: Callable, **kwargs):
        """Apply function to `.image` array. Return an instance of `BaseImage`
        if the result is of the same dimensions, otherwise return the result of
        the operation.

        Parameters
        ----------
        function : callable
            Function to apply to `self.image`.
        **kwargs
            Keyword arguments to pass to `function`.

        Returns
        -------
        BaseImage
            New instance of `BaseImage`.
        """
        ret = function(self.image, **kwargs)
        if isinstance(ret, np.ndarray) and (ret.ndim == self.image.ndim):
            return self.__class__(ret)

        return ret

    def gaussian(self, sigma: int = 1, **kwargs):
        """Apply Gaussian blur to image.

        Parameters
        ----------
        sigma : int
            Standard deviation for Gaussian kernel.
        **kwargs : dict
            Extra arguments passed to `skimage.filters.gaussian`.

        Returns
        -------
        BaseImage
            New instance of `BaseImage`.
        """
        from skimage import filters
        return self.apply(filters.gaussian, sigma=sigma, **kwargs)

    def digitize(self, bins: Union[list, tuple], **kwargs):
        """Digitize image.

        For more info see `numpy.digitize`.

        Parameters
        ----------
        bins : list, tuple
            List of bin values. Must be monotonic and one-dimensional.
        **kwargs : dict
            Extra arguments passed to `numpy.digitize`.

        Returns
        -------
        BaseImage
            New instance of `BaseImage`.
        """
        return self.apply(np.digitize, bins=bins, **kwargs)

    def binary_digitize(self, threshold: float = None):
        """Convert into a binary image.

        Parameters
        ----------
        threshold : float, optional
            Threshold used for segmentation. Defaults to median value.

        Returns
        -------
        BaseImage
            New instance of `BaseImage`.
        """
        if not threshold:
            threshold = np.median(self.image)
        return self.apply(np.digitize, bins=[threshold])
