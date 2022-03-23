"""Module containing sample data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from skimage.data import binary_blobs

from nanomesh._doc import doc

data_dir = Path(__file__).parent

if TYPE_CHECKING:
    from nanomesh import MeshContainer


@doc(dim='2d')
def binary_blobs2d(**kwargs) -> np.ndarray:
    """Generate {dim} binary blobs.

    Parameters
    ----------
    **kwargs
        These parameters are passed to :func:`skimage.data.binary_blobs`

    Returns
    -------
    numpy.ndarray
        {dim} array with binary blobs
    """
    kwargs.setdefault('length', 50)
    kwargs.setdefault('n_dim', 2)
    kwargs.setdefault('volume_fraction', 0.2)
    kwargs.setdefault('blob_size_fraction', 0.3)
    return binary_blobs(**kwargs).astype(int)


@doc(dim='3d')
def binary_blobs3d(**kwargs) -> np.ndarray:
    kwargs.setdefault('length', 50)
    kwargs.setdefault('n_dim', 3)
    kwargs.setdefault('volume_fraction', 0.2)
    kwargs.setdefault('blob_size_fraction', 0.2)
    return binary_blobs(**kwargs).astype(int)


def nanopores() -> np.ndarray:
    """Fetch 2D slice of nanopore dataset.

    Returns
    -------
    nanopores : numpy.ndarray
        2D image of nanopores
    """
    i = 30
    return nanopores3d()[i]


def nanopores_gradient() -> np.ndarray:
    """Fetch 2D slice of nanopore dataset with a gradient.

    Returns
    -------
    nanopores : (i,j) numpy.ndarray
        2D image of nanopores with gradient
    """
    return np.rot90(np.load(data_dir / 'nanopores_gradient.npy'))


def nanopores3d() -> np.ndarray:
    """Fetch 3D nanopore dataset.

    Returns
    -------
    nanopores : (i,j,k) numpy.ndarray
        3D image of nanopores
    """
    return np.load(data_dir / 'nanopores3d.npy')


@doc(dim='2d', kind='triangle', func='triangulate')
def blob_mesh2d(opts: str = 'q30a10', **kwargs) -> MeshContainer:
    """Return a {dim} {kind} mesh generated from binary blobs.

    Parameters
    ----------
    opts : str, optional
        Options passed to :func:`{func}`.
    **kwargs
        These parameters are passed to :func:`binary_blobs{dim}`.

    Returns
    -------
    mesh : MeshContainer
        {dim} {kind} mesh generated from binary blobs.
    """
    from nanomesh import plane2mesh
    data = binary_blobs2d(**kwargs)
    return plane2mesh(data, opts=opts)


@doc(blob_mesh2d, dim='3d', kind='tetrahedral', func='tetrahedralize')
def blob_mesh3d(opts: str = '-pAq', **kwargs) -> MeshContainer:
    from nanomesh import volume2mesh
    data = binary_blobs3d(**kwargs)
    return volume2mesh(data, opts=opts)


__all__ = [
    'nanopores',
    'nanopores3d',
    'binary_blobs3d',
    'blob_mesh2d',
    'blob_mesh3d',
]
