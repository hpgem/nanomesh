import operator
import os
from typing import Callable, Union

import numpy as np


def normalize_values(image: np.ndarray):
    """Rescale values to 0.0 to 1.0.

    Parameters
    ----------
    image : (m, n) np.ndarray
        Input image.

    Returns
    -------
    out : (m, n) np.ndarray
        Normalized image
    """
    out = (image - image.min()) / (image.max() - image.min())
    return out


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

    def __gt__(self, other):
        return self._compare(other, op=operator.gt)

    def __lt__(self, other):
        return self._compare(other, op=operator.lt)

    def __ge__(self, other):
        return self._compare(other, op=operator.ge)

    def __le__(self, other):
        return self._compare(other, op=operator.le)

    def _compare(self, other, *, op: Callable):
        """Helper function to implement overload functions.

        Parameters
        ----------
        other :
            Other instance, can be a numpy array.
        op : callable
            Operator (see operator module).

        Returns
        -------
        BaseImage
            Image with boolean data.
        """
        this = self.image
        if isinstance(other, self.__class__):
            other = other.image
        return self.__class__(op(this, other))

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

    def normalize_values(self):
        """Rescale values to 0.0 to 1.0.

        Returns
        -------
        out : BaseImage
            Normalized image
        """
        return self.apply(normalize_values)

    def binary_digitize(self, threshold: Union[float, str] = None):
        """Convert into a binary image.

        Parameters
        ----------
        threshold : float, optional
            Threshold used for segmentation. If given as a string,
            apply corresponding theshold via `BaseImage.threshold`.
            Defaults to median.

        Returns
        -------
        BaseImage
            New instance of `BaseImage`.
        """
        if not threshold:
            threshold = np.median(self.image)
        if isinstance(threshold, str):
            threshold = self.threshold(threshold)
        return self.apply(np.digitize, bins=[threshold])

    def threshold(self, method: str = 'otsu', **kwargs) -> float:
        """Compute threshold value using given method.

        For more info, see:
            https://scikit-image.org/docs/dev/api/skimage.filters.html

        Parameters
        ----------
        method : str
            Thresholding method to use. Defaults to `otsu`.
        **kwargs : dict
            Extra keyword arguments passed to threshold method.

        Returns
        -------
        threshold : float
            Threshold value.
        """
        from skimage import filters
        dispatch_table = {
            'isodata': filters.threshold_isodata,
            'li': filters.threshold_li,
            'local': filters.threshold_local,
            'mean': filters.threshold_mean,
            'minimum': filters.threshold_minimum,
            'multiotsu': filters.threshold_multiotsu,
            'niblack': filters.threshold_niblack,
            'otsu': filters.threshold_otsu,
            'sauvola': filters.threshold_sauvola,
            'triangle': filters.threshold_triangle,
            'yen': filters.threshold_yen,
        }
        try:
            func = dispatch_table[method]
        except KeyError:
            raise KeyError(f'`method` must be one of {dispatch_table.keys()}')

        return self.apply(func, **kwargs)

    def fft(self) -> 'BaseImage':
        """Apply fourier transform to image.

        Returns
        -------
        BaseImage
            Real component of fourier transform with the zero-frequency
            component shifted to the center of the spectrum.
        """
        fourier = np.fft.fftn(self.image)
        shifted = np.abs(np.fft.fftshift(fourier))
        return self.__class__(shifted)
