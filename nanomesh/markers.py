from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy.typing as npt

# tetgen:
# https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual006.html#sec75

# triangle:
# https://www.cs.cmu.edu/~quake/triangle.poly.html


@dataclass
class RegionMarker:
    label: int
    point: Union[Tuple[float, ...], npt.NDArray]
    name: Optional[str] = None
    constraint: float = 0  # max area or volume constraint

    def __post_init__(self):
        self.point = tuple(self.point)


RegionMarkerLike = Union[RegionMarker, Tuple[int, Tuple[float, ...],
                                             Optional[str]]]
