from ._base import GenericMesh, registry
from ._line import LineMesh
from ._tetra import TetraMesh
from ._triangle import TriangleMesh

for _mesh_class in (LineMesh, TriangleMesh, TetraMesh):
    registry[_mesh_class.cell_type] = _mesh_class

__all__ = [
    'LineMesh',
    'GenericMesh',
    'registry',
    'TetraMesh',
    'TriangleMesh',
]
