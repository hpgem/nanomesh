from pathlib import Path

import numpy as np
from skimage.data import binary_blobs, cells3d, coins, horse

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


__all__ = [
    'nanopores',
    'nanopores3d',
    'cells3d',
    'horse',
    'binary_blobs',
    'coins',
]
