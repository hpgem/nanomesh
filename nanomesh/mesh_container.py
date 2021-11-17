from __future__ import annotations

import meshio

from .mesh import BaseMesh, LineMesh, TetraMesh, TriangleMesh


class MeshContainer(meshio.Mesh):
    def extract(self, element_type: str = None) -> BaseMesh:
        """Extract points/cells of `element_type`.

        Parameters
        ----------
        element_type : str, optional
            Element type, such as line, triangle, tetra, etc.

        Returns
        -------
        Instance of BaseMesh or derived class
            Dataclass with `points`/`cells` attributes
        """
        if not element_type:
            element_type = self.get_default_type()

        cells = self.cells_dict[element_type]
        points = self.points
        return TriangleMesh(cells=cells, points=points)

    def plot(self, element_type: str = None):
        """Plot data."""
        mesh = self.extract(element_type)
        mesh.plot()

    def plot_mpl(self, element_type: str = None):
        mesh = self.extract(element_type)
        mesh.plot_mpl()

    def plot_itk(self, element_type: str = None):
        mesh = self.extract(element_type)
        mesh.plot_itk()

    def get_default_type(self) -> str:
        """Try to return highest dimension type.

        Default to first type `cells_dict`.
        """
        for type_ in ('tetra', 'triangle', 'line'):
            if type_ in self.cells_dict:
                return type_

        return list(self.cells_dict.keys())[0]
