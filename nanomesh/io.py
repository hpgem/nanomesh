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


def load_bin(filename: os.PathLike, input_dtype=np.float32) -> np.ndarray:
    """Summary.

    Parameters
    ----------
    filename : os.PathLike
        Path to the file.
    input_dtype : dtype, optional
        Numpy dtype of the data.

    Returns
    -------
    result : np.ndarray
        Data stored in the file.
    """
    filename = Path(filename)
    filename_info = filename.with_suffix(filename.suffix + '.info')

    info = read_info(filename_info)

    shape = info['NUM_Z'], info['NUM_Y'], info['NUM_X']

    with open(filename, 'rb') as f:
        result = np.fromfile(f, dtype=input_dtype)

    result = result.reshape(shape)

    return result
