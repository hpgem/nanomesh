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


def load_vol(filename: os.PathLike,
             dtype=np.float32,
             mmap_mode: str = None,
             shape: tuple = None) -> np.ndarray:
    """Load data from `.vol` file.

    The image shape is deduced from the `.vol.info` file. If this file is
    not present, the shape can be specified using the `shape` keyword.

    Parameters
    ----------
    filename : os.PathLike
        Path to the file.
    dtype : dtype, optional
        Numpy dtype of the data.
    mmap_mode : None, optional
        If not None, open the file using memory mapping. For more info on
        the modes, see: :func:`numpy.memmap`
    shape : tuple, optional
        Tuple of three ints specifying the shape of the data (order: z, y, x).

    Returns
    -------
    result : numpy.ndarray
        Data stored in the file.
    """
    filename = Path(filename)

    if not filename.exists():
        raise IOError(f'No such file: {filename}')

    try:
        filename_info = filename.with_suffix(filename.suffix + '.info')
        if not shape:
            info = read_info(filename_info)
            shape = info['NUM_Z'], info['NUM_Y'], info['NUM_X']
    except FileNotFoundError:
        raise ValueError(
            f'Info file not found: {filename_info.name}, specify '
            'the volume shape using the `shape` parameter.') from None

    result: np.ndarray

    if mmap_mode:
        result = np.memmap(filename, dtype=dtype, shape=shape,
                           mode=mmap_mode)  # type: ignore
    else:
        result = np.fromfile(filename, dtype=dtype)
        result = result.reshape(shape)

    return result
