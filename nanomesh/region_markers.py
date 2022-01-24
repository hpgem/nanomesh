from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy.typing as npt


@dataclass
class RegionMarker:
    label: int
    point: Union[Tuple[float, ...], npt.NDArray]
    name: Optional[str] = None

    def __post_init__(self):
        self.point = tuple(self.point)


RegionMarkerLike = Union[RegionMarker, Tuple[int, Tuple[float, ...],
                                             Optional[str]]]
