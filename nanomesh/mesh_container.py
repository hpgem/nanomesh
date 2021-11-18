from __future__ import annotations

import meshio
import numpy as np

from .mesh import BaseMesh

DIM_NAMES = [None, 'line', 'triangle', 'tetra']


class MeshContainer(meshio.Mesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_field_data()

    def _init_field_data(self):
        """Set up mappings from field<->number from .field_data."""
        from collections import defaultdict

        number_to_field = defaultdict(dict)
        field_to_number = defaultdict(dict)

        for field, (number, dimension) in self.field_data.items():
            dim_name = DIM_NAMES[dimension]
            field_to_number[dim_name][field] = number
            number_to_field[dim_name][number] = field

        self.number_to_field = number_to_field
        self.field_to_number = field_to_number

    def get_default_type(self) -> str:
        """Try to return highest dimension type.

        Default to first type `cells_dict`.

        Returns
        -------
        cell_type : str
        """
        for type_ in ('tetra', 'triangle', 'line'):
            if type_ in self.cells_dict:
                return type_

        return list(self.cells_dict.keys())[0]

    def get(self, cell_type: str = None) -> BaseMesh:
        """Extract mesh with points/cells of `cell_type`.

        Parameters
        ----------
        cell_type : str, optional
            Element type, such as line, triangle, tetra, etc.

        Returns
        -------
        BaseMesh
            Dataclass with `points`/`cells` attributes
        """
        if not cell_type:
            cell_type = self.get_default_type()

        try:
            cells = self.cells_dict[cell_type]
        except KeyError as e:
            msg = (f'No such cell type: {cell_type!r}. '
                   f'Must be one of {tuple(self.cells_dict.keys())!r}')
            raise KeyError(msg) from e

        points = self.points

        cell_data = self.get_all_cell_data(cell_type)

        return BaseMesh.create(cells=cells, points=points, **cell_data)

    def get_all_cell_data(self, cell_type: str = None) -> dict:
        """Get all cell data for given `cell_type`.

        Parameters
        ----------
        cell_type : str, optional
            Element type, such as line, triangle, tetra, etc.

        Returns
        -------
        data_dict : dict
            Dictionary with cell data
        """
        if not cell_type:
            cell_type = self.get_default_type()

        data_dict = {}
        for key in self.cell_data:
            new_key = key.replace(':', '-')
            data_dict[new_key] = self.get_cell_data(key, cell_type)

        return data_dict

    def plot(self, cell_type: str = None, **kwargs):
        """Plot data.

        Parameters
        ----------
        cell_type : str, optional
            Cell type to plot.
        **kwargs
            Extra keyword arguments passed to plotting method.
        """
        mesh = self.get(cell_type)
        mesh.plot(**kwargs)

    def plot_mpl(self, cell_type: str = None, **kwargs):
        """Plot data using matplotlib.

        Parameters
        ----------
        cell_type : str, optional
            Cell type to plot.
        **kwargs
            Extra keyword arguments passed to plotting method.
        """
        mesh = self.get(cell_type)
        mesh.plot_mpl(**kwargs)

    def plot_itk(self, cell_type: str = None, **kwargs):
        """Plot data using itk.

        Parameters
        ----------
        cell_type : str, optional
            Cell type to plot.
        **kwargs
            Extra keyword arguments passed to plotting method.
        """
        mesh = self.get(cell_type)
        mesh.plot_itk(**kwargs)

    def plot_pyvista(self, cell_type: str = None, **kwargs):
        """Plot data using pyvista.

        Parameters
        ----------
        cell_type : str, optional
            Cell type to plot.
        **kwargs
            Extra keyword arguments passed to plotting method.
        """
        mesh = self.get(cell_type)
        mesh.plot_pyvista(**kwargs)

    @classmethod
    def from_mesh(cls, mesh: BaseMesh):
        """Convert from `BaseMesh` to `MeshContainer`.

        Parameters
        ----------
        mesh : BaseMesh
            Input mesh, must be a subclass of `BaseMesh`.

        Returns
        -------
        MeshContainer
        """
        meshio_mesh = mesh.to_meshio()
        return cls(points=meshio_mesh.points,
                   cells=meshio_mesh.cells,
                   cell_data=meshio_mesh.cell_data)

    @classmethod
    def from_triangle_dict(cls, triangle_dict: dict):
        """Return instance of `MeshContainer` from triangle results dict.

        Parameters
        ----------
        triangle_dict : dict
            Triangle triangulate output dictionary.

        Returns
        -------
        mesh : MeshContainer
        """
        points = triangle_dict['vertices']

        cell_data = {}
        cells = {}

        cells['triangle'] = triangle_dict['triangles']

        point_data = {}
        try:
            point_data['gmsh-physical'] = triangle_dict[
                'vertex_markers'].squeeze()
        except KeyError:
            pass

        try:
            cells['line'] = triangle_dict['edges']
            cell_data['gmsh-physical'] = [
                np.zeros(len(cells['triangle'])),
                triangle_dict['edge_markers'],
            ]
        except KeyError:
            pass

        mesh = cls(
            points=points,
            cells=cells,
            cell_data=cell_data,
            point_data=point_data,
        )

        return mesh
