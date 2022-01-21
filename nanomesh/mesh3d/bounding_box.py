from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BoundingBox:
    """Container for bounding box coordinates."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @classmethod
    def from_shape(cls, shape: tuple):
        """Generate bounding box from data shape."""
        xmin, ymin, zmin = 0, 0, 0
        xmax, ymax, zmax = np.array(shape) - 1
        return cls(
            xmin=xmin,
            ymin=ymin,
            zmin=zmin,
            xmax=xmax,
            ymax=ymax,
            zmax=zmax,
        )

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Return dimensions of bounding box."""
        return (
            self.xmax - self.xmin,
            self.ymax - self.ymin,
            self.zmax - self.zmin,
        )

    @classmethod
    def from_points(cls, points: np.ndarray):
        """Generate bounding box from set of points or coordinates."""
        xmax, ymax, zmax = np.max(points, axis=0)
        xmin, ymin, zmin = np.min(points, axis=0)

        return cls(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
        )

    def to_points(self) -> np.ndarray:
        """Return (m,3) array with corner points."""
        return np.array([
            [self.xmin, self.ymin, self.zmin],
            [self.xmin, self.ymin, self.zmax],
            [self.xmin, self.ymax, self.zmin],
            [self.xmin, self.ymax, self.zmax],
            [self.xmax, self.ymin, self.zmin],
            [self.xmax, self.ymin, self.zmax],
            [self.xmax, self.ymax, self.zmin],
            [self.xmax, self.ymax, self.zmax],
        ])

    @property
    def center(self) -> np.ndarray:
        """Return center of the bounding box."""
        return np.array((
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
            (self.zmin + self.zmax) / 2,
        ))
