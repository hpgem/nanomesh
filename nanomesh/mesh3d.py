import logging
from typing import Dict

import meshio
import numpy as np
from skimage import measure

from ._mesh_shared import BaseMesher
from .mesh_container import TriangleMesh

logger = logging.getLogger(__name__)


class Mesher3D(BaseMesher):
    def __init__(self, image: np.ndarray):
        super().__init__(image)
        self.contours: Dict[int, TriangleMesh] = {}
        self.pad_width = 0

    def generate_contour(self, label: int = 1, smooth: bool = False):
        """Generate contours using marching cubes algorithm.

        Parameters
        ----------
        label : int, optional
            Label to generate contour for.
        smooth : bool, optional
            Lightly smooth the surface for a nicer looking result.
        """
        padded = np.pad(self.image, 1, mode='edge')
        padded = np.pad(padded, 1, mode='constant', constant_values=label + 1)
        obj = (padded == label).astype(int)

        verts, faces, *_ = measure.marching_cubes(
            obj,
            allow_degenerate=False,
        )

        # correct for padding
        verts -= np.array([2, 2, 2])
        mesh = TriangleMesh(vertices=verts, faces=faces)

        if smooth:
            mesh = mesh.smooth(iterations=10)
        # mesh = mesh.simplify(n_faces=10000)
        # mesh = mesh.simplify_by_vertex_clustering(voxel_size=2)

        logger.info(f'Generated contour with {len(mesh.faces)} '
                    f' faces ({label=})')

        self.contours[label] = mesh

    def tetrahedralize(self, **kwargs):
        """Tetrahedralize a surface contour mesh.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to
            `nanomesh.mesh_container.TriangleMesh.tetrahedralize`

        Returns
        -------
        TetraMesh
        """
        raise NotImplementedError

        mesh = ...
        volume_mesh = mesh.tetrahedralize(**kwargs)
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

    volume_mesh = mesher.tetrahedralize(label=1,
                                        order=1,
                                        mindihedral=30,
                                        minratio=1.1)
    return volume_mesh
