from dataclasses import dataclass
from typing import Tuple, Union

import numpy.typing as npt


@dataclass
class RegionMarker:
    label: int
    coordinates: npt.NDArray


RegionMarkerLike = Union[RegionMarker, Tuple[int, npt.NDArray]]
