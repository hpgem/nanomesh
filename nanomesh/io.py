import os
from ast import literal_eval
from pathlib import Path

import numpy as np


def read_info(filename: os.PathLike) -> dict:
    """Read volume metadata.

    Parameters
    ----------
    filename : PathLike
        Path to the file.

    Returns
    -------
    dct : dict
        Dictionary with the metadata.
    """
    dct = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith('!'):
                continue

            key, val = line.split('=')

            key = key.strip()
            val = val.strip()

            try:
                val = literal_eval(val)
            except ValueError:
                pass

            dct[key] = val

    return dct


def load_bin(
    filename: os.PathLike,
    dtype=np.float32,
    mmap_mode=None,
) -> np.ndarray:
    """Summary.

    Parameters
    ----------
    filename : os.PathLike
        Path to the file.
    dtype : dtype, optional
        Numpy dtype of the data.
    mmap_mode : None, optional
        If not None, open the file using memory mapping. For more info on
        the modes, see:
        https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

    Returns
    -------
    result : np.ndarray
        Data stored in the file.
    """
    filename = Path(filename)
    filename_info = filename.with_suffix(filename.suffix + '.info')

    info = read_info(filename_info)

    shape = info['NUM_Z'], info['NUM_Y'], info['NUM_X']

    if mmap_mode:
        result = np.memmap(filename, dtype=dtype, shape=shape, mode=mmap_mode)
    else:
        result = np.fromfile(filename, dtype=dtype)
        result = result.reshape(shape)

    return result
