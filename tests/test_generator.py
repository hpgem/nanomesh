import math
from pathlib import Path

import numpy as np

from nanomesh.generator import Generator


def test_generate_vect():
    """Test `Generator` / `generate_vect`."""
    expected_fn = Path(__file__).parent / 'generator_vol.npy'

    a_axis = 680
    c_axis = 680 * math.sqrt(2)

    gen = Generator(
        a=a_axis,
        c=c_axis,
        radius=0.24 * a_axis,
    )

    # Possible rotation/transformation of the coordinate system
    theta = math.radians(1.0)
    c = math.cos(theta)
    s = math.sin(theta)
    trans = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])

    data = gen.generate_vect(
        sizes=(100, 100, 100),
        resolution=(10, 10, 10),
        transform=trans,
        bin_val=[0., 1.],
    )

    if expected_fn.exists():
        expected_data = np.load(expected_fn)
    else:
        np.save(expected_fn, data)
        raise RuntimeError(f'Wrote expected data to {expected_fn.absolute()}')

    np.testing.assert_allclose(data, expected_data)
