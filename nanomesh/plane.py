import logging
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .base_image import BaseImage
from .mesh_container import TriangleMesh
from .utils import show_image

logger = logging.getLogger(__name__)


class Plane(BaseImage):
    @classmethod
    def load(cls, filename: os.PathLike, **kwargs) -> 'Plane':
        """Load the data. Supported filetypes: `.npy`.

        Parameters
        ----------
        filename : Pathlike
            Name of the file to load.
        **kwargs : dict
            Extra keyword arguments passed to `np.load`.

        Returns
        -------
        Plane
            Instance of this class.
        """
        array = np.load(filename, **kwargs)
        return cls(array)

    def apply(self, function: Callable, **kwargs):
        """Apply function to `.image` array. Return an instance of `Plane` if
        the result is a 2D image, otherwise return the result of the operation.

        Parameters
        ----------
        function : callable
            Function to apply to `self.image`.
        **kwargs
            Keyword arguments to pass to `function`.

        Returns
        -------
        Plane
            New instance of `Plane`.
        """
        return super().apply(function, **kwargs)

    def show(self, *, dpi: int = 80, title: str = None):
        """Plot the image using matplotlib.

        Parameters
        ----------
        dpi : int, optional
            DPI to render at.
        title : str, optional
            Set the title of the plot.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            Instance of `matplotlib.axes.Axes`
        """
        return show_image(self.image, dpi=dpi, title=title)

    def generate_mesh(self, **kwargs) -> 'TriangleMesh':
        """Generate mesh from binary (segmented) image.

        Parameters
        ----------
        **kwargs:
            Keyword arguments are passed to `mesh2d.generate_2d_mesh`

        Returns
        -------
        mesh : TriangleMesh
            Description of the mesh.
        """
        from .mesh2d import generate_2d_mesh
        return generate_2d_mesh(image=self.image, **kwargs)

    def select_roi(self):
        """Select region of interest in interactive matplotlib figure.

        Returns
        -------
        roi : `nanomesh.roi2d.ROISelector`
            Region of interest object. Bounding box is stored in `roi.bbox`.
        """
        from .roi2d import ROISelector
        ax = self.show(title='Select region of interest')
        roi = ROISelector(ax)
        return roi

    def crop(self, left: int, top: int, right: int, bottom: int) -> 'Plane':
        """Crop image to pixel indices.

        Parameters
        ----------
        left, top, right, bottom : int
            Index of pixel delimiting cropping box.

        Returns
        -------
        Plane
            New instance of `Plane`.
        """
        return Plane(self.image[top:bottom, left:right])

    def crop_to_roi(self, bbox: np.ndarray) -> 'Plane':
        """Crop plane to rectangle defined by bounding box.

        Parameters
        ----------
        bbox : (4,2) np.ndarray
            List of points describing region of interest. The bounding box
            may be rotated.

        Returns
        -------
        `nanomesh.Plane`
            Cropped region as `nanomesh.Plane` object.
        """
        from .roi2d import extract_rectangle
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
        ax : matplotlib.Axes
        """
        from .mesh_utils import compare_mesh_with_image
        return compare_mesh_with_image(image=self.image, mesh=mesh)
