import logging
import sys

from .mesh import LineMesh, TetraMesh, TriangleMesh
from .mesh2d import Mesher2D, compare_mesh_with_image, simple_triangulate
from .mesh3d import Mesher3D
from .mesh_container import MeshContainer
from .plane import Plane
from .volume import Volume

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

__author__ = 'Nicolas Renaud'
__email__ = 'n.renaud@esciencecenter.nl'
__version__ = '0.4.0'

__all__ = [
    '__author__',
    '__email__',
    '__version__',
    'compare_mesh_with_image',
    'LineMesh',
    'Mesher2D',
    'Mesher3D',
    'MeshContainer',
    'Plane',
    'simple_triangulate',
    'TetraMesh',
    'TriangleMesh',
    'Volume',
]
