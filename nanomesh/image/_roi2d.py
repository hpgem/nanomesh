from types import SimpleNamespace

import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from skimage import transform

from ..plotting import PolygonSelectorWithSnapping


def minimum_bounding_rectangle(coords: np.ndarray) -> np.ndarray:
    """Find the smallest bounding rectangle for a set of coordinates.

    Based on: https://stackoverflow.com/a/33619018

    Parameters
    ----------
    coords : (n,2) numpy.ndarray
        List of coordinates.

    Returns
    -------
    bbox_coords: (4,2) numpy.ndarray
        List of coordinates representing the corners of the bounding box.
    """
    # get the convex hull for the coords
    hull_coords = coords[ConvexHull(coords).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_coords) - 1, 2))
    edges = hull_coords[1:] - hull_coords[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, np.pi / 2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - np.pi / 2),
        np.cos(angles + np.pi / 2),
        np.cos(angles)
    ]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_coords = np.dot(rotations, hull_coords.T)

    # find the bounding coords
    min_x = np.nanmin(rot_coords[:, 0], axis=1)
    max_x = np.nanmax(rot_coords[:, 0], axis=1)
    min_y = np.nanmin(rot_coords[:, 1], axis=1)
    max_y = np.nanmax(rot_coords[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    rotation = rotations[best_idx]

    bbox_coords = np.zeros((4, 2))
    bbox_coords[0] = np.dot([x1, y2], rotation)
    bbox_coords[1] = np.dot([x2, y2], rotation)
    bbox_coords[2] = np.dot([x2, y1], rotation)
    bbox_coords[3] = np.dot([x1, y1], rotation)

    return bbox_coords


def extract_rectangle(image: np.ndarray, *, bbox: np.ndarray):
    """Extract rectangle from image.

    The image is straightened using an Euclidean transform via
    :func:`skimage.transform.EuclideanTransform()`.

    Parameters
    ----------
    image : (i,j) numpy.ndarray
        Image to extract rectangle from.
    bbox : (4,2) numpy.ndarray
        Four coordinate describing the corners of the bounding box.

    Returns
    -------
    warped : (i,j) numpy.ndarray
        The warped input image extracted from the bounding box.
    """
    a = int(np.linalg.norm(bbox[0] - bbox[1]))
    b = int(np.linalg.norm(bbox[1] - bbox[2]))

    src = np.array([[0, 0], [0, a], [b, a], [b, 0]])
    dst = np.array(bbox)

    tform3 = transform.EuclideanTransform()
    tform3.estimate(src, dst)
    warped = transform.warp(image, tform3, output_shape=(a, b))

    return warped


class ROISelector:
    ROTATE = True
    """Select a region of interest points in the figure by enclosing them
    within a polygon. A rectangle is fitted to the polygon.

    - Press the 'esc' key to start a new polygon.
    - Hold the 'shift' key to move all of the vertices.
    - Hold the 'ctrl' key to move a single vertex.

    Attributes
    ----------
    bbox : (4,2) numpy.ndarray
        Coordinates describing the corners of the polygon
    """

    def __init__(self, ax, snap_to: np.ndarray = None):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.bbox = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        self.verts = None

        self.poly = PolygonSelectorWithSnapping(ax,
                                                self.onselect,
                                                snap_to=snap_to)

    def onselect(self, verts):
        """Trigger this function when a polygon is closed."""
        self.verts = np.array(verts)
        self.bbox = self.bounding_rectangle(rotate=self.ROTATE)

        bounds = self.get_bounds()
        self.ax.set_title(f'left {bounds.left:.0f} '
                          f'right {bounds.right:.0f}'
                          f'\ntop {bounds.top:.0f} '
                          f'bottom {bounds.bottom:.0f}')
        self.draw_bbox()
        self.canvas.draw_idle()

    def disconnect(self):
        """Disconnect the selector."""
        self.poly.disconnect_events()
        self.canvas.draw_idle()

    def draw_bbox(self):
        """Draw bounding box as a patch on the image."""
        # remove existing patches in case the roi is modified
        self.ax.patches.clear()
        polygon = Polygon(self.bbox, facecolor='red', alpha=0.3)
        self.ax.add_patch(polygon)

    def get_bounds(self) -> SimpleNamespace:
        """Get bounds of bbox (left, right, top, bottom)."""
        left, top = self.bbox.min(axis=0)
        right, bottom = self.bbox.max(axis=0)
        bounds = SimpleNamespace(left=left,
                                 top=top,
                                 right=right,
                                 bottom=bottom)
        return bounds

    def bounding_rectangle(self, rotate=True) -> np.ndarray:
        """Return bounding rectangle.

        Parameters
        ----------
        rotate : bool, optional
            If True, allow rotation of the bounding box to find the minumum
            bounding rectangle.

        Returns
        -------
        (4,2) numpy.ndarray
            Array containing the corners of the bounding rectangle.
        """
        if self.verts is None:
            raise ValueError('No vertices have been selected!')

        if rotate:
            return minimum_bounding_rectangle(self.verts)
        else:
            left, top = self.verts.min(axis=0)
            right, bottom = self.verts.max(axis=0)

            return np.array([[right, bottom], [right, top], [left, top],
                             [left, bottom]])
