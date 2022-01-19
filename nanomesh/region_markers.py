from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy.typing as npt


@dataclass
class RegionMarker:
    label: int
    coordinates: npt.NDArray
    name: Optional[str] = None


RegionMarkerLike = Union[RegionMarker, Tuple[int, npt.NDArray]]
