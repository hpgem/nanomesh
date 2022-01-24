from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass
class RegionMarker:
    label: int
    point: Tuple[float, ...]
    name: Optional[str] = None

    def __post_init__(self):
        self.point = tuple(self.point)


RegionMarkerLike = Union[RegionMarker, Tuple[int, Tuple[float, ...],
                                             Optional[str]]]
