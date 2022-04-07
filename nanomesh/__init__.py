import logging
import sys

from ._tetgen_wrapper import tetrahedralize
from ._triangle_wrapper import simple_triangulate, triangulate
from .image import Image, Plane, Volume
from .image2mesh import Mesher, Mesher2D, Mesher3D, plane2mesh, volume2mesh
from .mesh import LineMesh, Mesh, TetraMesh, TriangleMesh
from .mesh_container import MeshContainer
from .region_markers import RegionMarker, RegionMarkerList

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

__author__ = 'Nicolas Renaud'
__email__ = 'n.renaud@esciencecenter.nl'
__version__ = '0.7.0'

__all__ = [
    '__author__',
    '__email__',
    '__version__',
    'Mesh',
    'LineMesh',
    'MeshContainer',
    'Mesher2D',
    'Mesher3D',
    'Mesher',
    'Image',
    'Plane',
    'plane2mesh',
    'RegionMarker',
    'RegionMarkerList',
    'simple_triangulate',
    'tetrahedralize',
    'TetraMesh',
    'TriangleMesh',
    'triangulate',
    'Volume',
    'volume2mesh',
]
