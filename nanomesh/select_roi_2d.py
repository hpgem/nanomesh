import numpy as np
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from scipy.spatial import ConvexHull
from skimage import transform


def minimum_bounding_rectangle(coords):
    """Find the smallest bounding rectangle for a set of coordinates.

    Based on: https://stackoverflow.com/a/33619018

    Parameters
    ----------
    coords : (n,2) np.ndarray
        List of coordinates

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
    angle = angles[best_idx]

    bbox_coords = np.zeros((4, 2))
    bbox_coords[0] = np.dot([x1, y2], rotation)
    bbox_coords[1] = np.dot([x2, y2], rotation)
    bbox_coords[2] = np.dot([x2, y1], rotation)
    bbox_coords[3] = np.dot([x1, y1], rotation)

    return bbox_coords, angle


def extract_rectangle(data, *, bbox, plot=True):
    """Extract rectangle from image."""
    a = int(np.linalg.norm(bbox[0] - bbox[1]))
    b = int(np.linalg.norm(bbox[1] - bbox[2]))

    src = np.array([[0, 0], [0, a], [b, a], [b, 0]])
    dst = np.array(bbox)

    tform3 = transform.EuclideanTransform()
    tform3.estimate(src, dst)
    warped = transform.warp(data, tform3, output_shape=(a, b))
    warped = np.rot90(warped, k=3)

    return warped


def select_roi(array: np.ndarray):
    """Summary.

    Parameters
    ----------
    array : np.ndarray
        Description

    Returns
    -------
    bbox : (4,2) np.ndarray
        Bounding box
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(array)
    roi = GetBBox(ax)
    return roi


class GetBBox:
    def __init__(self, ax, data=None):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.bbox = None
        self.angle = None

        self.poly = PolygonSelector(ax, self.onselect)

        print('Select a region of interest points in the figure by enclosing '
              'them within a polygon.')
        print("- Press the 'esc' key to start a new polygon.")
        print("- Hold the 'shift' key to move all of the vertices.")
        print("- Hold the 'ctrl' key to move a single vertex.")

    def onselect(self, verts):
        self.path = Path(verts)

        self.ax.patches = []

        verts = self.path.vertices
        bbox, angle = minimum_bounding_rectangle(verts)
        polygon = Polygon(bbox, facecolor='red', alpha=0.3)
        self.ax.add_patch(polygon)

        self.bbox = bbox
        self.angle = angle

        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()
