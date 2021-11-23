from __future__ import annotations

from collections import defaultdict
from enum import Enum
from types import MappingProxyType
from typing import Dict, List

import meshio
import numpy as np

from .mesh import BaseMesh


class CellType(Enum):
    NULL = 0
    LINE = 1
    TRIANGLE = 2
    TETRA = 3


class MeshContainer(meshio.Mesh):
    @property
    def number_to_field(self):
        """Mapping from numbers to fields, proxy to `.field_data`."""
        number_to_field = defaultdict(dict)

        for field, (number, dimension) in self.field_data.items():
            dim_name = CellType(dimension).name.lower()
            number_to_field[dim_name][number] = field

        return MappingProxyType(dict(number_to_field))

    @property
    def field_to_number(self):
        """Mapping from fields to numbers, proxy to `.field_data`."""
        field_to_number = defaultdict(dict)

        for field, (number, dimension) in self.field_data.items():
            dim_name = CellType(dimension).name.lower()
            field_to_number[dim_name][field] = number

        return MappingProxyType(dict(field_to_number))

    def set_field_data(self, cell_type: str, field_data: Dict[int, str]):
        """Update the values in `.field_data`.

        Parameters
        ----------
        cell_type : str
            Cell type to update the values ofr.
        field_data : dict
            Dictionary with key-to-number mapping, i.e.
            `field_data={0: 'green', 1: 'blue', 2: 'red'}`
            maps `0` to `green`, etc.
        """
        try:
            input_field_data = dict(self.number_to_field)[cell_type]
        except KeyError:
            input_field_data = {}

        input_field_data.update(field_data)

        new_field_data = self.field_data.copy()

        remove_me = []

        for field, (value, field_cell_type) in new_field_data.items():
            if CellType(field_cell_type) == CellType[cell_type.upper()]:
                remove_me.append(field)

        for field in remove_me:
            new_field_data.pop(field)

        for value, field in input_field_data.items():
            CELL_TYPE = CellType[cell_type.upper()].value
            new_field_data[field] = [value, CELL_TYPE]

        self.field_data: Dict[str, List[int]] = new_field_data

    @property
    def cell_types(self):
        """Return cell types in order."""
        return tuple(cell.type for cell in self.cells)

    def set_cell_data(self, cell_type: str, key: str, value):
        """Set `key` to `value` for `cell_type` in `.cell_data_dict`."""
        index = self.cell_types.index(cell_type)
        assert len(value) == len(self.cells_dict[cell_type])

        try:
            self.cell_data[key][index] = value
        except KeyError:
            new_cell_data = []

            # set missing cells to 0
            for i, _ in enumerate(self.cell_types):
                if i == index:
                    new_cell_data.append(value)
                else:
                    new_cell_data.append(
                        np.zeros(len(self.cells[0].data), dtype=int))

            self.cell_data[key] = new_cell_data

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
        fields = self.number_to_field.get(mesh._cell_type, None)
        mesh.plot_mpl(fields=fields, **kwargs)

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
            point_data['physical'] = triangle_dict['vertex_markers'].squeeze()
        except KeyError:
            pass

        try:
            cells['line'] = triangle_dict['edges']
            # Order must match order of cell_data
            cell_data['physical'] = [
                np.zeros(len(cells['triangle'])),
                triangle_dict['edge_markers'].squeeze(),
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

    @classmethod
    def read(cls, *args, **kwargs):
        """Wrapper for `meshio.read`.

        For gmsh:
        - remaps `gmsh:physical` -> `physical`
        - remaps `gmsh:geometrical` -> `geometrical`
        """
        from meshio import read
        mesh = read(*args, **kwargs)
        mesh.prune_z_0()

        cell_data = {}
        for key, value in mesh.cell_data.items():
            if key in ('gmsh:physical', 'gmsh:geometrical'):
                key = key.replace('gmsh:', '')

            cell_data[key] = value

        point_data = {}
        for key, value in mesh.point_data.items():
            if key in ('gmsh:physical', 'gmsh:geometrical'):
                key = key.replace('gmsh:', '')

            point_data[key] = value

        return cls(
            mesh.points,
            mesh.cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=mesh.field_data,
            point_sets=mesh.point_sets,
            cell_sets=mesh.cell_sets,
            gmsh_periodic=mesh.gmsh_periodic,
            info=mesh.info,
        )

    def write(self, filename, file_format: str = None, **kwargs):
        """Thin wrapper of `meshio.write` to avoid altering class.

        For gmsh:
        - remaps `physical` -> `gmsh:physical`
        - remaps `geometrical` -> `gmsh:geometrical`

        Parameters
        ----------
        filename : str
            File to write to.
        file_format : str, optional
            Specify file format. By default, this is guessed from the
            extension.
        **kwargs
            Extra keyword arguments passed to `meshio.write`.
        """
        from pathlib import Path

        from meshio import write
        from meshio._helpers import _filetype_from_path

        if file_format is None:
            file_format = _filetype_from_path(Path(filename))

        if file_format.startswith('gmsh'):
            cell_data = {}
            for key, value in self.cell_data.items():
                if key in ('physical', 'geometrical'):
                    key = f'gmsh:{key}'

                cell_data[key] = value

            point_data = {}
            for key, value in self.point_data.items():
                if key in ('physical', 'geometrical'):
                    key = f'gmsh:{key}'

                point_data[key] = value
        else:
            cell_data = self.cell_data
            point_data = self.point_data

        out_mesh = meshio.Mesh(
            self.points,
            self.cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=self.field_data,
            point_sets=self.point_sets,
            cell_sets=self.cell_sets,
            gmsh_periodic=self.gmsh_periodic,
            info=self.info,
        )

        write(filename, mesh=out_mesh, file_format=file_format, **kwargs)
