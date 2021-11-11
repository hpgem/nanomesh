import logging
import os
from pathlib import Path
from typing import Callable

import meshio
import numpy as np

from .base_image import BaseImage
from .io import load_vol
from .plane import Plane

logger = logging.getLogger(__name__)


class Volume(BaseImage):
    @classmethod
    def load(cls, filename: os.PathLike, **kwargs) -> 'Volume':
        """Load the data. Supported filetypes: `.npy`, `.vol`.

        For memory mapping, use `mmap_mode='r'`. Memory-mapped
            files are used for accessing small segments of large files on
            disk, without reading the entire file into memory. Note that this
            can still result in some slow / unexpected behaviour with some
            operations.

            More info:
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

        Parameters
        ----------
        filename : PathLike
            Name of the file to load.
        **kwargs : dict
            Extra keyword arguments passed onto data readers.

        Returns
        -------
        Volume
            Instance of this class.

        Raises
        ------
        IOError
            Raised if the file extension is unknown.
        """
        filename = Path(filename)
        suffix = filename.suffix.lower()

        if suffix == '.npy':
            array = np.load(filename, **kwargs)
        elif suffix == '.vol':
            array = load_vol(filename, **kwargs)
        else:
            raise IOError(f'Unknown file extension: {suffix}')
        return cls(array)

    def apply(self, function: Callable, **kwargs) -> 'Volume':
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
        return super().apply(function, **kwargs)

    def show_slice(self, **kwargs):
        """Show slice using `nanomesh.utils.SliceViewer`.

        Extra arguments are passed on.
        """
        from nanomesh.utils import SliceViewer
        sv = SliceViewer(self.image, **kwargs)
        sv.interact()
        return sv

    def show_volume(self, renderer: str = 'itkwidgets', **kwargs) -> None:
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
        from nanomesh.mesh3d import generate_3d_mesh
        return generate_3d_mesh(image=self.image, **kwargs)

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

    def select_subvolume(self,
                         *,
                         xs: tuple = None,
                         ys: tuple = None,
                         zs: tuple = None) -> 'Volume':
        """Select a subvolume from the current volume.

        Each range must include a start and stop value, for example:

        `vol.select_subvolume(xs=(10, 20))` is equivalent to:
        `vol.image[[:,:,10:20]`

        or

        `vol.select_subvolume(ys=(20, 25), zs=(40, 50))` is equivalent to:
        `vol.image[[40:50,20:25,:]`

        Parameters
        ----------
        xs : tuple, optional
            Range to select along y-axis
        ys : tuple, optional
            Range to select along y-axis
        zs : tuple, optional
            Range to select along z-axis

        Returns
        -------
        Plane
            Return 3D volume representation.
        """
        default = slice(None)
        slices = tuple(slice(*r) if r else default for r in (zs, ys, xs))

        array = self.image[slices]
        return Volume(array)
