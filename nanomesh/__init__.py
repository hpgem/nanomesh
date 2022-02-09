import logging
import sys

from ._tetgen_wrapper import tetrahedralize
from ._triangle_wrapper import triangulate
from .image import Plane, Volume
from .image2mesh import Mesher2D, Mesher3D, plane2mesh, volume2mesh
from .mesh import LineMesh, TetraMesh, TriangleMesh
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
    'LineMesh',
    'MeshContainer',
    'Mesher2D',
    'Mesher3D',
    'Plane',
    'plane2mesh',
    'RegionMarker',
    'tetrahedralize',
    'TetraMesh',
    'TriangleMesh',
    'triangulate',
    'Volume',
    'volume2mesh',
]
