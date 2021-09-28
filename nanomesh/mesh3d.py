import logging
from dataclasses import dataclass
from typing import Dict

import meshio
import numpy as np
from skimage import measure

from nanomesh.mesh_utils import simple_triangulate

from ._mesh_shared import BaseMesher
from .mesh_container import TriangleMesh

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @classmethod
    def from_shape(cls, shape):
        xmin, ymin, zmin = 0, 0, 0
        xmax, ymax, zmax = np.array(shape) - 1
        return cls(xmin=xmin,
                   ymin=ymin,
                   zmin=zmin,
                   xmax=xmax,
                   ymax=ymax,
                   zmax=zmax)


def add_corner_points(mesh, bbox) -> None:
    """Summary.

    Parameters
    ----------
    mesh : TYPE
        Mesh to add corner points to.
    bbox : TYPE
        Description

    Deleted Parameters
    ------------------
    min_vals : TYPE
        Description
    max_vals : TYPE
        Description
    """
    corners = np.array([
        [bbox.xmin, bbox.ymin, bbox.zmin],
        [bbox.xmin, bbox.ymin, bbox.zmax],
        [bbox.xmin, bbox.ymax, bbox.zmin],
        [bbox.xmin, bbox.ymax, bbox.zmax],
        [bbox.xmax, bbox.ymin, bbox.zmin],
        [bbox.xmax, bbox.ymin, bbox.zmax],
        [bbox.xmax, bbox.ymax, bbox.zmin],
        [bbox.xmax, bbox.ymax, bbox.zmax],
    ])

    mesh.vertices = np.vstack([mesh.vertices, corners])


def close_side(mesh, *, side: str, bbox: BoundingBox):
    """Fill a side of the bounding box with triangles.

    Parameters
    ----------
    mesh : TriangleMesh
        Input contour mesh.
    side : str
        Side of the volume to close. Must be one of
        `left`, `right`, `top`, `bottom`, `front`, `back`.
    bbox : BoundingBox
        Coordinates of the bounding box.

    Returns
    -------
    mesh : TriangleMesh
        Triangle mesh with the given side closed.

    Raises
    ------
    ValueError
        When the value of `side` is invalid.
    """
    all_verts = mesh.vertices

    if side == 'top':
        edge_col = 2
        edge_value = bbox.zmin
    elif side == 'bottom':
        edge_col = 2
        edge_value = bbox.zmax
    elif side == 'left':
        edge_col = 1
        edge_value = bbox.ymin
    elif side == 'right':
        edge_col = 1
        edge_value = bbox.ymax
    elif side == 'front':
        edge_col = 0
        edge_value = bbox.xmin
    elif side == 'back':
        edge_col = 0
        edge_value = bbox.xmax
    else:
        raise ValueError('Side must be one of `right`, `left`, `bottom`'
                         f'`top`, `front`, `back`. Got {side=}')

    keep_cols = [col for col in (0, 1, 2) if col != edge_col]
    is_edge = all_verts[:, edge_col] == edge_value

    coords = all_verts[is_edge][:, keep_cols]

    edge_mesh = simple_triangulate(vertices=coords, opts='')
    edge_mesh.plot()

    mesh_edge_index = np.argwhere(is_edge).flatten()
    new_edge_index = np.arange(len(mesh_edge_index))
    mapping = np.vstack([new_edge_index, mesh_edge_index])

    shape = edge_mesh.faces.shape
    new_faces = edge_mesh.faces.copy().ravel()

    mask = np.in1d(new_faces, mapping[0, :])
    new_faces[mask] = mapping[1,
                              np.searchsorted(mapping[0, :], new_faces[mask])]
    new_faces = new_faces.reshape(shape)

    new_labels = np.ones(len(new_faces)) * 123

    vertices = all_verts
    faces = np.vstack([mesh.faces, new_faces])
    labels = np.hstack([mesh.labels, new_labels])

    mesh = TriangleMesh(vertices=vertices, faces=faces, labels=labels)
    return mesh


def wrap(mesh, *, bbox):
    """Wrap the surface mesh and close any open contours along the bounding
    box.

    Parameters
    ----------
    mesh : TriangleMesh
        Input mesh.
    bbox : BoundingBox
        Coordinates of the bounding box.

    Returns
    -------
    TriangleMesh
        Ouput mesh.
    """
    add_corner_points(mesh, bbox=bbox)

    sides = 'top', 'bottom', 'left', 'right', 'front', 'back'

    for side in sides:
        mesh = close_side(mesh, side=side, bbox=bbox)

    return mesh


class Mesher3D(BaseMesher):
    def __init__(self, image: np.ndarray):
        super().__init__(image)
        self.contours: Dict[int, TriangleMesh] = {}
        self.pad_width = 0

    def generate_contour(
        self,
        level: float = None,
        label: int = 1,
    ):
        """Generate contours using marching cubes algorithm.

        Parameters
        ----------
        level : float, optional
            Contour value to search for isosurfaces (i.e. the threshold value).
            By default takes the average of the min and max value. Can be 
            ignored if a binary image is passed to `Mesher3D`.
        label : int, optional
            Label to assign to contour.
        """
        verts, faces, *_ = measure.marching_cubes(
            self.image,
            allow_degenerate=False,
        )

        mesh = TriangleMesh(vertices=verts, faces=faces)

        bbox = BoundingBox.from_shape(self.image.shape)
        mesh = wrap(mesh, bbox=bbox)

        logger.info(f'Generated contour with {len(mesh.faces)} '
                    f' faces ({label=})')

        self.contours[label] = mesh

    def tetrahedralize(self, label: int = 1, **kwargs):
        """Tetrahedralize a surface contour mesh.

        Parameters
        ----------
        label : int
            Label of the contour to tetrahedralize.
        **kwargs
            Keyword arguments passed to
            `nanomesh.mesh_container.TriangleMesh.tetrahedralize`

        Returns
        -------
        TetraMesh
        """
        contour = self.contours[label]
        volume_mesh = contour.tetrahedralize(**kwargs)
        return volume_mesh


def generate_3d_mesh(
    image: np.ndarray,
    *,
    step_size: int = 2,
    pad_width: int = 2,
    point_density: float = 1 / 10000,
    n_faces: int = 1000,
) -> 'meshio.Mesh':
    """Generate mesh from binary (segmented) image.

    Parameters
    ----------
    image : 3D np.ndarray
        Input image to mesh.

    Returns
    -------
    volume_mesh : TetraMesh
        Description of the mesh.
    """
    mesher = Mesher3D(image)
    mesher.generate_contour(label=0)
    mesher.generate_contour(label=1)

    volume_mesh = mesher.tetrahedralize()
    return volume_mesh
