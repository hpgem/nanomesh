from typing import Union

import numpy as np
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from scipy.spatial import ConvexHull
from skimage import transform


def minimum_bounding_rectangle(coords: np.ndarray) -> np.ndarray:
    """Find the smallest bounding rectangle for a set of coordinates.

    Based on: https://stackoverflow.com/a/33619018

    Parameters
    ----------
    coords : (n,2) np.ndarray
        List of coordinates.

    Returns
    -------
    bbox_coords: (4,2) np.ndarray
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


def extract_rectangle(image: np.ndarray, *, bbox: Union[list, np.ndarray]):
    """Extract rectangle from image. The image is straightened using an
    Euclidean transform.

    Parameters
    ----------
    image : 2D np.ndarray
        Image to extract rectangle from.
    bbox : (4,2) list or np.ndarray
        Four coordinate describing the corners of the bounding box.

    Returns
    -------
    warped : 2D np.ndarray
        The warped input image extracted from the bounding box.
    """
    a = int(np.linalg.norm(bbox[0] - bbox[1]))
    b = int(np.linalg.norm(bbox[1] - bbox[2]))

    src = np.array([[0, 0], [0, a], [b, a], [b, 0]])
    dst = np.array(bbox)

    tform3 = transform.EuclideanTransform()
    tform3.estimate(src, dst)
    warped = transform.warp(image, tform3, output_shape=(a, b))
    warped = np.rot90(warped, k=3)

    return warped


class ROISelector:
    """Select a region of interest points in the figure by enclosing them
    within a polygon. A rectangle is fitted to the polygon.

    - Press the 'esc' key to start a new polygon.
    - Hold the 'shift' key to move all of the vertices.
    - Hold the 'ctrl' key to move a single vertex.

    Attributes
    ----------
    bbox : (4,2) np.ndarray
        Coordinates describing the corners of the polygon
    """
    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.bbox = None

        self.poly = PolygonSelector(ax, self.onselect)

    def onselect(self, verts):
        """Trigger this function when a polygon is closed."""
        verts = np.array(verts)
        self.bbox = minimum_bounding_rectangle(verts)
        self.draw_bbox()
        self.canvas.draw_idle()

    def disconnect(self):
        """Disconnect the selector."""
        self.poly.disconnect_events()
        self.canvas.draw_idle()

    def draw_bbox(self):
        """Draw bounding box as a patch on the image."""
        # remove existing patches in case the roi is modified
        self.ax.patches = []
        polygon = Polygon(self.bbox, facecolor='red', alpha=0.3)
        self.ax.add_patch(polygon)
