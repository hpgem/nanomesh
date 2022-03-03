from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import numpy as np

from .._doc import doc
from ._image import Image
from ._utils import show_image

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .mesh import TriangleMesh


@doc(Image, prefix='Data class for working with 2D image data', shape='(i,j) ')
class Plane(Image, ndim=2):

    def show(self,
             *,
             ax: plt.Axes = None,
             title: str = None,
             **kwargs) -> 'plt.Axes':
        """Plot the image using :mod:`matplotlib`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to use for plotting.
        title : str, optional
            Title for the plot.
        **kwargs
            These parameters are passed to
            :func:`matplotlib.pyplot.imshow`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Instance of :class:`matplotlib.axes.Axes`
        """
        return show_image(self.image, ax=ax, title=title, **kwargs)

    @doc(show)
    def plot(self, *args, **kwargs):
        return self.show(*args, **kwargs)

    def generate_mesh(self, **kwargs) -> TriangleMesh:
        """Generate mesh from binary (segmented) image.

        Parameters
        ----------
        **kwargs:
            Keyword arguments are passed to
            :func:`nanomesh.plane2mesh`

        Returns
        -------
        mesh : TriangleMesh
            Description of the mesh.
        """
        from nanomesh.image2mesh import plane2mesh
        return plane2mesh(image=self.image, **kwargs)

    def select_roi(self, from_points: np.ndarray = None):
        """Select region of interest in interactive matplotlib figure.

        Parameters
        ----------
        from_points : (n, 2) numpy.ndarray, optional
            List of points that are used as anchors for the roi
            selection.

        Returns
        -------
        roi : `nanomesh.image._roi2d.ROISelector`
            Region of interest object. Bounding box is stored in
            :attr:`roi.bbox`.
        """
        from ._roi2d import ROISelector
        ax = self.show(title='Select region of interest')
        if from_points is not None:
            # reverse columns to match image
            from_points = from_points[:, ::-1]
            ax.scatter(*from_points.T)
        roi = ROISelector(ax, snap_to=from_points)
        return roi

    def crop(self, left: int, top: int, right: int, bottom: int) -> Plane:
        """Crop image to pixel indices.

        Parameters
        ----------
        left, top, right, bottom : int
            Index of pixel delimiting cropping box.

        Returns
        -------
        Plane
            New instance of :class:`Plane`.
        """
        return Plane(self.image[top:bottom, left:right])

    def crop_to_roi(self, bbox: np.ndarray) -> Plane:
        """Crop plane to rectangle defined by bounding box.

        Parameters
        ----------
        bbox : (4,2) numpy.ndarray
            List of points describing region of interest. The bounding box
            may be rotated.

        Returns
        -------
        Plane
            Cropped region as :class:`Plane` object.
        """
        from ._roi2d import extract_rectangle
        cropped = extract_rectangle(self.image, bbox=bbox)
        return Plane(cropped)

    def compare_with_mesh(self, mesh: TriangleMesh) -> 'plt.Axes':
        """Make a plot comparing the image with the given mesh.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to compare the image with.

        Returns
        -------
        plt.Axes
        """
        from ..utils import compare_mesh_with_image
        return compare_mesh_with_image(image=self.image, mesh=mesh)

    def compare_with_digitized(self,
                               digitized: Union[np.ndarray, 'Plane'],
                               cmap: str = None,
                               **kwargs) -> 'plt.Axes':
        """Compare image with digitized (segmented) image. Returns a plot with
        the overlay of the digitized image.

        Parameters
        ----------
        digitized : numpy.ndarray, Plane
            Digitized image of the same dimensions to overlay
        cmap : str
            Matplotlib color map for :func:`matplotlib.pyplot.imshow`
        **kwargs
            These parameters are passed to :func:`skimage.color.label2rgb`.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from skimage.color import label2rgb

        if isinstance(digitized, Plane):
            digitized = digitized.image

        # bg_label=0 is default for scikit-image from 0.19 onwards
        kwargs.setdefault('bg_label', 0)
        image_overlay = label2rgb(digitized, image=self.image, **kwargs)

        fig, ax = plt.subplots()

        ax.imshow(image_overlay, interpolation='none', cmap=cmap)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Image comparison')

        return ax

    def compare_with_other(self,
                           other: Union[np.ndarray, 'Plane'],
                           cmap: str = None,
                           **kwargs) -> 'plt.Axes':
        """Compare image with other image.

        Parameters
        ----------
        other : numpy.ndarray, Plane
            Other image of the same dimensions to overlay
        cmap : str
            Matplotlib color map for :func:`matplotlib.pyplot.imshow`
        **kwargs
            These parameters are passed to :func:`skimage.util.compare_images`.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from skimage.util import compare_images

        if isinstance(other, Plane):
            other = other.image

        kwargs.setdefault('method', 'checkerboard')
        kwargs.setdefault('n_tiles', (4, 4))
        comp = compare_images(self.image, other, **kwargs)

        fig, ax = plt.subplots()

        ax.imshow(comp, interpolation='none', cmap=cmap)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Image comparison ({kwargs["method"]})')

        return ax

    def clear_border(self, *, object_label: int, fill_val: int,
                     **kwargs) -> Plane:
        """Remove objects at the border of the image.

        Parameters
        ----------
        object_label : int
            Label of the objects to remove.
        fill_val : int
            Cleared objects are set to this value.
        **kwargs
            These parameters are passed to
            :func:`skimage.segmentation.clear_border`.

        Returns
        -------
        Plane
            New instance of :class:`Plane`.
        """
        from skimage import segmentation

        objects = (self.image == object_label).astype(int)
        border_cleared = segmentation.clear_border(objects, **kwargs)
        mask = (border_cleared != objects)

        out = self.image.copy()
        out[mask] = fill_val
        return self.__class__(out)

    def try_all_threshold(self, **kwargs):
        """Produce a plot trying all available thresholds using
        :func:`skimage.filters.try_all_threshold`.

        Parameters
        ----------
        **kwargs
            These parameters are passed to
            :func:`skimage.filters.try_all_threshold`.
        """
        from skimage import filters
        kwargs.setdefault('verbose', False)
        self.apply(filters.try_all_threshold, **kwargs)
