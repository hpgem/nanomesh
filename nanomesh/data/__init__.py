from pathlib import Path

import numpy as np
from skimage.data import binary_blobs

data_dir = Path(__file__).parent


def nanopores() -> np.ndarray:
    """Fetch 2D slice of nanopore dataset.

        Returns
        -------
    nanopores : (i,j) np.ndarray
        2D image of nanopores
    """
    return nanopores3d()[30]


def nanopores3d() -> np.ndarray:
    """Fetch 3D nanopore dataset.

    Returns
    -------
    nanopores 3d : (i,j,k) np.ndarray
            3D image of nanopores
    """
    return np.load(data_dir / 'nanopores3d.npy')


def binary_blobs3d(depth: int = 20, **kwargs):
    """Generate synthetic pores based on :func:`skimage.data.binary_blobs`.

    Parameters
    ----------
    depth : int, optional
        Number of pixels along z-direction.
    **kwargs
        These parameters are passed to :func:`skimage.data.binary_blobs`.

    Returns
    -------
    numpy.array
        3D binary pores.
    """
    blobs = binary_blobs(**kwargs)
    return np.tile(blobs, (depth, 1, 1))


def triangle_mesh():
    raise NotImplementedError


def triangle_mesh3d():
    raise NotImplementedError


def tetra_mesh():
    raise NotImplementedError


__all__ = [
    'nanopores',
    'nanopores3d',
    'binary_blobs3d',
    'triangle_mesh',
    'triangle_mesh3d',
]
