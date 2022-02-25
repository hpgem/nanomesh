from __future__ import annotations

from collections import abc
from copy import copy
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import List, Optional, Tuple, Union

import numpy.typing as npt

# tetgen:
# https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual006.html#sec75

# triangle:
# https://www.cs.cmu.edu/~quake/triangle.poly.html


@dataclass
class RegionMarker:
    """Data class to store region info.

    A region is typically an area or volume bounded
    by segments or cells.

    Attributes
    ----------
    label : int
        Label used to identify the region.
    point : tuple[float]
        Point inside the region.
    name : str, optional
        Name of the region.
    constraint : float, default=0
        This value can be used to set the maximum size constraint
        for cells in the region during meshing.
    """
    label: int
    point: Union[Tuple[float, ...], npt.NDArray]
    name: Optional[str] = None
    constraint: float = 0  # max area or volume constraint

    def __post_init__(self):
        self.point = tuple(self.point)

    def update(self, **kwargs) -> RegionMarker:
        new_marker = copy(self)
        new_marker.__dict__.update(kwargs)
        return new_marker


RegionMarkerLike = Union[RegionMarker, Tuple[int, Tuple[float, ...],
                                             Optional[str]]]


class RegionMarkerList(List[RegionMarker]):
    """List of region markers."""

    def __repr__(self):
        s = ',\n'.join(f'    {marker}' for marker in self)
        return f'{self.__class__.__name__}(\n{s}\n)'

    @singledispatchmethod
    def relabel(self, old: abc.Callable, new: int):
        markers = (m.update(label=new) if old(m.label) else m for m in self)
        return RegionMarkerList(markers)

    @relabel.register
    def _(self, old: list, new: int):

        def f(x):
            return x in old

        return self.relabel(f, new)

    @relabel.register
    def _(self, old: int, new: int):

        def f(x):
            return x == old

        return self.relabel(f, new)

    @singledispatchmethod
    def label_sequentially(self, old: abc.Callable):
        markers = [copy(marker) for marker in self]
        i = max(self.labels) + 1
        for marker in markers:
            if old(marker.label):
                marker.label = i
                i += 1
        return RegionMarkerList(markers)

    @label_sequentially.register
    def _(self, old: list):

        def f(x):
            return x in old

        return self.label_sequentially(f)

    @property
    def labels(self) -> set:
        return set(m.label for m in self)

    @property
    def names(self) -> set:
        return set(m.name for m in self)

    def rename(self, old, new):
        raise NotImplementedError


if __name__ == '__main__':
    k = [RegionMarker(x, point=(x, x)) for x in range(10)]
    m = RegionMarkerList(k)

    new1 = m.relabel(1, 11)
    new2 = m.relabel([2, 3, 4], 11)
    new3 = m.relabel(lambda x: x < 4, 11)

    new4 = new3.label_sequentially(lambda x: x == 11)
