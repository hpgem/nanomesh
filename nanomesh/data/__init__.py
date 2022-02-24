from pathlib import Path

import numpy as np
from skimage.data import binary_blobs

from nanomesh._doc import doc

data_dir = Path(__file__).parent


@doc(dim='2d')
def binary_blobs2d(**kwargs):
    """Generate {dim} binary blobs.

    Parameters
    ----------
    Parameters
    ----------
    **kwargs
        These parameters are passed to :func:`skimage.data.binary_blobs`

    Returns
    -------
    numpy.array
        {dim} array with binary blobs
    """
    kwargs.setdefault('length', 50)
    kwargs.setdefault('n_dim', 2)
    kwargs.setdefault('volume_fraction', 0.2)
    kwargs.setdefault('blob_size_fraction', 0.3)
    return binary_blobs(**kwargs).astype(int)


@doc(dim='3d')
def binary_blobs3d(**kwargs):
    kwargs.setdefault('length', 50)
    kwargs.setdefault('n_dim', 3)
    kwargs.setdefault('volume_fraction', 0.2)
    kwargs.setdefault('blob_size_fraction', 0.2)
    return binary_blobs(**kwargs).astype(int)


def nanopores() -> np.ndarray:
    """Fetch 2D slice of nanopore dataset.

    Returns
    -------
    nanopores : np.ndarray
        2D image of nanopores
    """
    i = 30
    return nanopores3d()[i]


def nanopores_gradient() -> np.ndarray:
    """Fetch 2D slice of nanopore dataset with a gradient.

    Returns
    -------
    nanopores : (i,j) np.ndarray
        2D image of nanopores with gradient
    """
    return np.rot90(np.load(data_dir / 'nanopores_gradient.npy'))


def nanopores3d() -> np.ndarray:
    """Fetch 3D nanopore dataset.

    Returns
    -------
    nanopores : (i,j,k) np.ndarray
        3D image of nanopores
    """
    return np.load(data_dir / 'nanopores3d.npy')


def mesh():
    """Return a 2d mesh."""
    raise NotImplementedError


def mesh3d():
    """Return a 3d mesh."""
    raise NotImplementedError


__all__ = [
    'nanopores',
    'nanopores3d',
    'binary_blobs3d',
    'triangle_mesh',
    'triangle_mesh3d',
]
