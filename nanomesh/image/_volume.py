import logging
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

from .._doc import doc
from ._image import Image
from ._plane import Plane

if TYPE_CHECKING:
    from nanomesh import MeshContainer

logger = logging.getLogger(__name__)


@doc(Image,
     prefix='Data class for working with 3D (volumetric) image data.',
     shape='(i,j,k) ')
class Volume(Image, ndim=3):

    def show_slice(self, **kwargs):
        """Show slice using :class:`nanomesh.image.SliceViewer`.

        Extra arguments are passed on.

        Parameters
        ----------
        **kwargs
            These parameters are passed to
            :class:`nanomesh.image.SliceViewer`

        Returns
        -------
        SliceViewer
        """
        from ._utils import SliceViewer
        data = np.flip(self.image, axis=0)
        sv = SliceViewer(data, **kwargs)
        sv.interact()
        return sv

    def plot(self, *args, **kwargs):
        """Shortcut for :meth:`Volume.show`."""
        return self.show(*args, **kwargs)

    def show(self, renderer: str = 'itkwidgets', **kwargs) -> None:
        """Show volume using `itkwidgets` or `ipyvolume`.

        Parameters
        ----------
        renderer : str, optional
            Select renderer (`ipvolume`, `itkwidgets`)
        **kwargs
            These parameters are passed to
            :func:`itkwidgets.view` or :func:`ipyvolume.quickvolshow`.

        Raises
        ------
        ValueError
            Raised if the renderer is unknown.
        """
        if renderer in ('ipyvolume', 'ipv'):
            import ipyvolume as ipv
            return ipv.quickvolshow(self.image, **kwargs)
        elif renderer in ('itkwidgets', 'itk', 'itkw'):
            import itkwidgets as itkw
            return itkw.view(self.image)
        else:
            raise ValueError(f'No such renderer: {renderer!r}')

    def generate_mesh(self, **kwargs) -> 'MeshContainer':
        """Generate mesh from binary (segmented) image.

        Parameters
        ----------
        **kwargs:
            These parameters are passed to `mesh3d.volume2mesh`

        Returns
        -------
        MeshContainer
            Instance of :class:`MeshContainer`
        """
        from nanomesh import volume2mesh
        return volume2mesh(image=self.image, **kwargs)

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
            Return plane as :class:`Plane`.

        Raises
        ------
        ValueError
            If none of the `x`, `y`, or `z` arguments have been specified
        """
        index: Tuple[Union[int, slice], ...]

        if x is not None:
            index = np.s_[:, :, x]
        elif y is not None:
            index = np.s_[:, y, :]
        elif z is not None:
            index = np.s_[z, :, :]
        else:
            raise ValueError(
                'One of the arguments `x`, `y`, or `z` must be specified.')

        array = self.image[index]
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
        Volume
            Subvolume as :class:`Volume`
        """
        default = slice(None)
        slices = tuple(slice(*r) if r else default for r in (zs, ys, xs))

        array = self.image[slices]
        return Volume(array)
