import logging
import os
from pathlib import Path

import meshio
import numpy as np

from .io import load_vol
from .plane import Plane
from .utils import requires

try:
    import pygalmesh
except ImportError:
    pygalmesh = None

logger = logging.getLogger(__name__)


class Volume:
    def __init__(self, image):

        self.image = image

    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.image.shape})'

    def __eq__(self, other):
        if isinstance(other, Volume):
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
    def from_sitk_image(cls, sitk_image) -> 'Volume':
        """Return instance of `Volume` from `SimpleITK.Image`."""
        import SimpleITK as sitk
        image = sitk.GetArrayFromImage(sitk_image)
        return cls(image)

    def save(self, filename: str):
        """Save the data. Supported filetypes: `.npy`.

        Parameters
        ----------
        filename : str
            Name of the file to save to.
        """
        np.save(filename, self.image)

    @classmethod
    def load(cls, filename: os.PathLike, mmap: bool = False) -> 'Volume':
        """Load the data. Supported filetypes: `.npy`, `.vol`.

        Parameters
        ----------
        filename : PathLike
            Name of the file to load.
        mmap : bool, optional
            If True, load the file using memory mapping. Memory-mapped
            files are used for accessing small segments of large files on
            disk, without reading the entire file into memory. Note that this
            can still result in some slow / unexpected behaviour with some
            operations. Memory-mapped files are read-only by default.

            More info:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

        Returns
        -------
        Volume
            Instance of this class.

        Raises
        ------
        IOError
            Raised if the file extension is unknown.
        """
        mmap_mode = 'r' if mmap else None
        filename = Path(filename)
        suffix = filename.suffix.lower()

        if suffix == '.npy':
            array = array = np.load(filename, mmap_mode=mmap_mode)
        elif suffix == '.vol':
            array = load_vol(filename, mmap_mode=mmap_mode)
        else:
            raise IOError(f'Unknown file extension: {suffix}')
        return cls(array)

    def apply(self, function, **kwargs) -> 'Volume':
        """Apply function to `.image` array. Return an instance of `Volume` if
        the result is a 3D image, otherwise return the result of the operation.

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
        ret = function(self.image, **kwargs)
        if isinstance(ret, np.ndarray) and (ret.ndim == self.image.ndim):
            return Volume(ret)

        return ret

    def show_slice(self, **kwargs):
        """Show slice using `nanomesh.utils.SliceViewer`.

        Extra arguments are passed on.
        """
        from .utils import SliceViewer
        sv = SliceViewer(self.image, **kwargs)
        sv.interact()
        return sv

    def show_volume(self, renderer='itkwidgets', **kwargs):
        """Show volume using `itkwidgets` or `ipyvolume`.

        Extra keyword arguments (`kwargs`) are passed to
        `itkwidgets.view` or `ipyvolume.quickvolshow`.
        """
        if renderer in ('ipyvolume', 'ipv'):
            import ipyvolume as ipv
            return ipv.quickvolshow(self.image, **kwargs)
        elif renderer in ('itkwidgets', 'itk', 'itkw'):
            import itkwidgets as itkw
            return itkw.view(self.image)
        else:
            raise ValueError(f'No such renderer: {renderer!r}')

    def generate_mesh(self, **kwargs) -> 'meshio.Mesh':
        """Generate mesh from binary (segmented) image.

        Parameters
        ----------
        **kwargs:
            Keyword arguments are passed to `mesh3d.generate_3d_mesh`

        Returns
        -------
        meshio.Mesh
            Description of the mesh.
        """
        from .mesh3d import generate_3d_mesh
        return generate_3d_mesh(image=self.image, **kwargs)

    @requires(condition=pygalmesh, message='requires pygalmesh')
    def generate_mesh_cgal(self, h=(1.0, 1.0, 1.0), **kwargs) -> 'meshio.Mesh':
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
        mesh = pygalmesh.generate_from_array(self.image, h, **kwargs)
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

        array = self.image[slice]
        return Plane(array)
