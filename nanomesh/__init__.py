import logging
import sys

from nanomesh.mesh2d import Mesher2D
from nanomesh.mesh3d import Mesher3D
from nanomesh.mesh_containers import TetraMesh, TriangleMesh
from nanomesh.mesh_utils import simple_triangulate
from nanomesh.plane import Plane
from nanomesh.volume import Volume

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

__author__ = 'Nicolas Renaud'
__email__ = 'n.renaud@esciencecenter.nl'
__version__ = '0.3.0'

__all__ = [
    '__author__',
    '__email__',
    '__version__',
    'Mesher2D',
    'Mesher3D',
    'Plane',
    'simple_triangulate',
    'TetraMesh',
    'TriangleMesh',
    'Volume',
]
