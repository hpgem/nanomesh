import logging

import meshio
import numpy as np

from .utils import show_image

logger = logging.getLogger(__name__)


class Plane:
    def __init__(self, image):

        self.image = image

    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.image.shape})'

    def to_sitk_image(self):
        """Return instance of `SimpleITK.Image` from `.image`."""
        import SimpleITK as sitk
        return sitk.GetImageFromArray(self.image)

    @classmethod
    def from_sitk_image(cls, sitk_image) -> 'Plane':
        """Return instance of `Volume` from `SimpleITK.Image`."""
        import SimpleITK as sitk
        image = sitk.GetImageFromArray(sitk_image)
        return cls(image)

    @classmethod
    def load(cls, filename: str) -> 'Plane':
        """Load the data. Supported filetypes: `.npy`.

        Parameters
        ----------
        filename : str
            Name of the file to load.

        Returns
        -------
        Plane
            Instance of this class.
        """
        array = np.load(filename)
        return cls(array)

    def apply(self, function, **kwargs) -> 'Plane':
        """Apply function to `.image` and return new instance of `Plane`.

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
        new_image = function(self.image, **kwargs)
        return Plane(new_image)

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

    def generate_mesh(self, **kwargs) -> 'meshio.Mesh':
        """Generate mesh from binary (segmented) image.

        Parameters
        ----------
        **kwargs:
            Keyword arguments are passed to `mesh2d.generate_2d_mesh`

        Returns
        -------
        meshio.Mesh
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

    def crop_to_roi(self, bbox):
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
