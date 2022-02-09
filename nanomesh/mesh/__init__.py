from ._base import registry
from ._line import LineMesh
from ._tetra import TetraMesh
from ._triangle import TriangleMesh

for MeshClass in (LineMesh, TriangleMesh, TetraMesh):
    registry[MeshClass.cell_type] = MeshClass

__all__ = [
    'LineMesh',
    'TetraMesh',
    'TriangleMesh',
    'registry',
]
