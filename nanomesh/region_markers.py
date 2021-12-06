from dataclasses import dataclass

import numpy.typing as npt


@dataclass
class RegionMarker:
    label: int
    coordinates: npt.NDArray
