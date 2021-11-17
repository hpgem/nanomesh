from __future__ import annotations

import meshio

from .mesh import BaseMesh, LineMesh, TetraMesh, TriangleMesh


class MeshContainer(meshio.Mesh):
    def get_default_type(self) -> str:
        """Try to return highest dimension type.

        Default to first type `cells_dict`.
        """
        for type_ in ('tetra', 'triangle', 'line'):
            if type_ in self.cells_dict:
                return type_

        return list(self.cells_dict.keys())[0]

    def get(self, element_type: str = None) -> BaseMesh:
        """Extract mesh with points/cells of `element_type`.

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

    def plot(self, element_type: str = None, **kwargs):
        """Plot data."""
        mesh = self.get(element_type)
        mesh.plot(**kwargs)

    def plot_mpl(self, element_type: str = None, **kwargs):
        """Plot data using matplotlib."""
        mesh = self.get(element_type)
        mesh.plot_mpl(**kwargs)

    def plot_itk(self, element_type: str = None, **kwargs):
        """Plot data using itk."""
        mesh = self.get(element_type)
        mesh.plot_itk(**kwargs)

    def plot_pyvista(self, element_type: str = None, **kwargs):
        """Plot data using pyvista."""
        mesh = self.get(element_type)
        mesh.plot_pyvista(**kwargs)

    @classmethod
    def from_mesh(cls, mesh: BaseMesh):
        """Convert from `BaseMesh` to `MeshContainer`."""
        tmp = mesh.to_meshio()
        return cls(points=tmp.points, cells=tmp.cells, cell_data=tmp.cell_data)
