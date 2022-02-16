from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


@dataclass
class Polygon:
    # List of coordinates describing a polygon.
    points: np.ndarray

    def __len__(self):
        return len(self.points)

    def find_point(self) -> np.ndarray:
        """Use rejection sampling to find point in polygon.

        Returns
        -------
        point : numpy.ndarray
            Coordinate of point in the polygon
        """
        # start with guess in center of polygon
        point = self.points.mean(axis=0)

        while not self.contains_point(point):
            xmin, ymin = self.points.min(axis=0)
            xmax, ymax = self.points.max(axis=0)
            point = np.random.uniform(xmin,
                                      xmax), np.random.uniform(ymin, ymax)

        return point

    def close_corner(self, shape: tuple) -> 'Polygon':
        """Check if polygons are in the corner, and close them if needed.

        Polygons which cover a corner cannot be closed by joining the first
        and last element, because some of the area is missed. This algorithm
        adds the corner point to close the polygons.

        Parameters
        ----------
        shape : tuple
            Shape of the source image. Used to check which corners the
            polygon touches.

        Returns
        -------
        polygon : Polygon
            Return a polygon with a corner point added if needed (n+1,2),
            otherwise return the input polygon (n,2)
        """
        xmin, ymin = self.points.min(axis=0)
        xmax, ymax = self.points.max(axis=0)

        xdim, ydim = np.array(shape) - 1

        left = (xmin == 0)
        right = (xmax == xdim)
        bottom = (ymin == 0)
        top = (ymax == ydim)

        if bottom and left:
            extra_point = (0, 0)
        elif top and left:
            extra_point = (0, ydim)
        elif top and right:
            extra_point = (xdim, ydim)
        elif bottom and right:
            extra_point = (xdim, 0)
        else:
            # all good
            return self

        points = np.vstack([self.points, extra_point])
        return Polygon(points)

    def subdivide(self, max_dist: int = 10, plot: bool = False) -> 'Polygon':
        """This algorithm looks for long edges in the polygon and subdivides
        them so they are no longer than `max_dist`

        Parameters
        ----------
        max_dist : int, optional
            Maximum distance between neighbouring coordinates.
        plot : bool, optional
            Show plot of the generated points.

        Returns
        -------
        Polygon
            Polygon with updated coordinate array.
        """
        points = self.points

        new_points: Any = []
        rolled = np.roll(points, shift=-1, axis=0)
        diffs = rolled - points
        # ignore last point, do not wrap around
        dist = np.linalg.norm(diffs[:-1], axis=1)

        last_i = 0

        for i in np.argwhere(dist > max_dist).reshape(-1, ):
            new_points.append(points[last_i:i])
            start = points[i]
            stop = rolled[i]
            to_add = int(dist[i] // max_dist)
            interpolated = np.linspace(start, stop, to_add, endpoint=False)
            new_points.append(interpolated)

            last_i = i + 1

        new_points.append(points[last_i:])
        new_points = np.vstack(new_points)

        if plot:
            plt.scatter(*points.T[::-1], color='red', s=100, marker='x')
            plt.plot(*points.T[::-1], color='red')
            plt.scatter(*new_points.T[::-1], color='green', s=100, marker='+')
            plt.plot(*new_points.T[::-1], color='green')
            plt.axis('equal')
            plt.show()

        return Polygon(new_points)

    def remove_duplicate_points(self) -> 'Polygon':
        """Remove duplicate points from polygon.

        For a polygon it is implied that the last point connects to the
        first point. In case the first point equals the last point, this
        results in errors down the line.

        Returns
        -------
        Polygon
        """
        points = self.points

        first = points[0]
        last = points[-1]

        if np.all(first == last):
            points = points[:-1]

        return Polygon(points)

    def approximate(self, *args, **kwargs) -> 'Polygon':
        """Approximate polygon.

        Parameters
        ----------
        *args : list
            Extra arguments padded to `skimage.measure.approximate_polygon`.
        **kwargs
            These parameters are passed to
            `skimage.measure.approximate_polygon`.

        Returns
        -------
        new_polygon : Polygon
        """
        new_points = measure.approximate_polygon(self.points, *args, **kwargs)
        return Polygon(new_points)

    def contains_point(self, point: np.ndarray) -> bool:
        """Test whether point lies inside polygon.

        Parameters
        ----------
        point : (2,) numpy.ndarray
            Point coordinates

        Returns
        -------
        bool
            True if corresponding point is inside the polygon
        """
        return measure.points_in_poly([point], self.points)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """Test whether points lie inside polygon.

        Parameters
        ----------
        points : (n,2) numpy.ndarray
            List of points

        Returns
        -------
        mask : (n,) numpy.ndarray[bool]
            Boolean array where true corresponds to points
            lying inside the polygon
        """
        return measure.points_in_poly(points, self.points)
