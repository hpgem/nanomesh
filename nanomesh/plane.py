import logging

import meshio
import numpy as np
import SimpleITK as sitk

from .utils import show_image

logger = logging.getLogger(__name__)


class Plane:
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
    def load(cls, filename: str) -> 'Plane':
        """Load the data.

        Supported filetypes: `.npy`

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
        return cls.from_array(array)

    def apply(self, function, **kwargs) -> 'Plane':
        """Apply function to `.image` and return new instance of `Plane`.

        Parameters
        ----------
        function : callable
            Function to apply to `self.array_view`.
        **kwargs
            Keyword arguments to pass to `function`.

        Returns
        -------
        Plane
            New instance of `Plane`.
        """
        new_image = function(self.image, **kwargs)
        return Plane(new_image)

    def apply_np(self, function, **kwargs) -> 'Plane':
        """Apply function to `.array_view` and return new instance of `Plane`.

        Parameters
        ----------
        function : callable
            Function to apply to `self.array_view`.
        **kwargs
            Keyword arguments to pass to `function`.

        Returns
        -------
        Plane
            New instance of `Plane`.
        """
        new_image = function(self.array_view, **kwargs)
        return Plane.from_array(new_image)

    def show(self, *, dpi: int = 80, title: str = None):
        """Plot the image using matplotlib.

        Parameters
        ----------
        dpi : int, optional
            Description
        title : str, optional
            Description
        """
        show_image(self.array_view, dpi=dpi, title=title)

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
        return generate_2d_mesh(image=self.array_view, **kwargs)

    def select_roi(self):
        from .select_roi_2d import select_roi
        bbox = select_roi(self.array_view)
        return bbox
