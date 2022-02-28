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
    """Collection of region markers.

    Sub-classes :func:`list`.
    """

    def __repr__(self):
        s = ',\n'.join(f'    {marker}' for marker in self)
        return f'{self.__class__.__name__}(\n{s}\n)'

    @singledispatchmethod
    def relabel(self,
                old: abc.Callable,
                new: int,
                name: str = None) -> RegionMarkerList:
        """Relabel a sub-group of region markers.

        Parameters
        ----------
        old : callable or list or int
            The old label(s) to replace. The value can be of type:
            - int, matches this exact label
            - list, matches all labels in this list
            - callable, returns True if label is a match
        new : int
            New label to assign
        name : str, optional
            Optional name to assign to the new region markers.

        Returns
        -------
        RegionMarkerList
            New list of region markers with updated labels.
        """
        if not name:
            names = [m.name for m in self if old(m.label)]
            if len(set(names)) == 1:
                name = tuple(names)[0]

        markers = (m.update(label=new, name=name) if old(m.label) else m
                   for m in self)

        return RegionMarkerList(markers)

    @relabel.register
    def _(self, old: abc.Sequence, new: int, *args, **kwargs):

        def f(x):
            return x in old

        return self.relabel(f, new, *args, **kwargs)

    @relabel.register
    def _(self, old: int, new: int, *args, **kwargs):

        def f(x):
            return x == old

        return self.relabel(f, new, *args, **kwargs)

    @singledispatchmethod
    def label_sequentially(self, old: abc.Callable, fmt_name: str = None):
        """Re-label a set of regions sequentially.

        `old` matches a sub-group of region markers and assigns new labels.
        The new labels are incremented sequentially. E.g.:
        [1,1,1,2] -> [3,4,5,6]

        Parameters
        ----------
        old : callable or list or int
            The old label(s) to replace. The value can be of type:
            - int, matches this exact label
            - list, matches all labels in this list
            - callable, returns True if label is a match
        fmt_name : str, optional
            Optional format string for the label name, e.g. 'feature{}'. The
            placeholder (`{}`) will be substituted by the new label.

        Returns
        -------
        RegionMarkerList
            New list of region markers with updated labels.
        """
        markers = [copy(marker) for marker in self]
        i = max(self.labels) + 1
        for marker in markers:
            if not old(marker.label):
                continue

            marker.label = i
            if fmt_name:
                marker.name = fmt_name.format(i)

            i += 1
        return RegionMarkerList(markers)

    @label_sequentially.register
    def _(self, old: abc.Sequence, *args, **kwargs):

        def f(x):
            return x in old

        return self.label_sequentially(f, *args, **kwargs)

    @label_sequentially.register
    def _(self, old: int, *args, **kwargs):

        def f(x):
            return x == old

        return self.label_sequentially(f, *args, **kwargs)

    @property
    def labels(self) -> set:
        """Return all unique region labels."""
        return set(m.label for m in self)

    @property
    def names(self) -> set:
        """Return all unique region names."""
        return set(m.name for m in self)
