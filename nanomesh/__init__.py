import logging
import sys

from ._tetgen_wrapper import tetrahedralize
from ._triangle_wrapper import triangulate
from .image import Plane, Volume
from .mesh import LineMesh, TetraMesh, TriangleMesh
from .mesh2d import Mesher2D, compare_mesh_with_image
from .mesh3d import Mesher3D
from .mesh_container import MeshContainer
from .region_markers import RegionMarker

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

__author__ = 'Nicolas Renaud'
__email__ = 'n.renaud@esciencecenter.nl'
__version__ = '0.5.0'

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
    'triangulate',
    'tetrahedralize',
    'TetraMesh',
    'RegionMarker',
    'TriangleMesh',
    'Volume',
]
