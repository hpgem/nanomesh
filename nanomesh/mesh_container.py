from __future__ import annotations

from collections import defaultdict
from enum import Enum
from types import MappingProxyType
from typing import Dict, List

import meshio
import numpy as np

from .mesh._base import GenericMesh
from .mesh._mixin import PruneZ0Mixin


class _CellType(Enum):
    NULL = 0
    LINE = 1
    TRIANGLE = 2
    TETRA = 3


class MeshContainer(meshio.Mesh, PruneZ0Mixin):
    """Low-level container for storing mesh data.

    Can contain multiple cell types sharing a set of points.
    It can store different types of cells and associated data.
    :class:`MeshContainer` is based on :class:`meshio.Mesh`
    (https://github.com/nschloe/meshio).

    Parameters
    ----------
    points : numpy.ndarray
        Array storing the mesh points (e.g. vertices)
    cells : list
        List of cell arrays
    point_data : todo
    cell_data : todo
    field_data : dict
        Dictionary mapping field names to cell data values.
    point_sets : todo
    cell_sets : todo
    gmsh_periodic : todo
    info : todo
    """

    def __repr__(self):
        """Canonical string representation."""
        s = super().__repr__().splitlines()
        s[0] = f'<{self.__class__.__name__}>'
        return '\n'.join(s)

    @property
    def number_to_field(self):
        """Mapping from numbers to fields, proxy to
        :attr:`MeshContainer.field_data`."""
        number_to_field = defaultdict(dict)

        for field, (number, dimension) in self.field_data.items():
            dim_name = _CellType(dimension).name.lower()
            number_to_field[dim_name][number] = field

        return MappingProxyType(
            {k: MappingProxyType(v)
             for k, v in number_to_field.items()})

    @property
    def field_to_number(self):
        """Mapping from fields to numbers, proxy to
        :attr:`MeshContainer.field_data`."""
        field_to_number = defaultdict(dict)

        for field, (number, dimension) in self.field_data.items():
            dim_name = _CellType(dimension).name.lower()
            field_to_number[dim_name][field] = number

        return MappingProxyType(
            {k: MappingProxyType(v)
             for k, v in field_to_number.items()})

    def set_field_data(self, cell_type: str, field_data: Dict[int, str]):
        """Update the values in :attr:`MeshContainer.field_data`.

        Parameters
        ----------
        cell_type : str
            Cell type to update the values for.
        field_data : dict
            Dictionary with key-to-number mapping, i.e.
            `field_data={0: 'green', 1: 'blue', 2: 'red'}`
            maps `0` to `green`, etc.
        """
        try:
            input_field_data = dict(self.number_to_field[cell_type])
        except KeyError:
            input_field_data = {}

        input_field_data.update(field_data)

        new_field_data = self.field_data.copy()

        remove_me = []

        for field, (value, field_cell_type) in new_field_data.items():
            if _CellType(field_cell_type) == _CellType[cell_type.upper()]:
                remove_me.append(field)

        for field in remove_me:
            new_field_data.pop(field)

        for value, field in input_field_data.items():
            CELL_TYPE = _CellType[cell_type.upper()].value
            new_field_data[field] = [value, CELL_TYPE]

        self.field_data: Dict[str, List[int]] = new_field_data

    @property
    def cell_types(self):
        """Return cell types in order."""
        return tuple(cell.type for cell in self.cells)

    def set_cell_data(self, cell_type: str, key: str, value: np.ndarray):
        """Set the cell data to the given value.

        Updates :attr:`MeshContainer.cell_data`.

        Parameters
        ----------
        cell_type : str
            Cell type, must be in :attr:`MeshContainer.cell_types`
        key : str
            The key of the value in :attr:`MeshContainer.cell_data`
        value : numpy.ndarray
            Array of values to set
        """
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

        Default to first type :attr:`MeshContainer.cells_dict`.

        Returns
        -------
        cell_type : str
        """
        for type_ in ('tetra', 'triangle', 'line'):
            if type_ in self.cells_dict:
                return type_

        return list(self.cells_dict.keys())[0]

    def get(self, cell_type: str = None):
        """Extract mesh with points/cells of `cell_type`.

        Parameters
        ----------
        cell_type : str, optional
            Element type, such as line, triangle, tetra, etc.

        Returns
        -------
        GenericMesh
            Mesh of the given type
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

        fields = self.field_to_number.get(cell_type, None)

        return GenericMesh(cells=cells,
                           points=points,
                           fields=fields,
                           **cell_data)

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
            These parameters are passed to plotting method.
        """
        cell_types = {cell.type for cell in self.cells}

        if (not cell_type) and (cell_types == {'line', 'triangle'}):
            from .plotting import linetrianglemeshplot
            return linetrianglemeshplot(self, **kwargs)
        else:
            mesh = self.get(cell_type)
            return mesh.plot(**kwargs)

    def plot_mpl(self, cell_type: str = None, **kwargs):
        """Plot data using :mod:`matplotlib`.

        Parameters
        ----------
        cell_type : str, optional
            Cell type to plot.
        **kwargs
            These parameters are passed to plotting method.
        """
        mesh = self.get(cell_type)
        return mesh.plot_mpl(**kwargs)

    def plot_itk(self, cell_type: str = None, **kwargs):
        """Plot data using `itk`.

        Parameters
        ----------
        cell_type : str, optional
            Cell type to plot.
        **kwargs
            These parameters are passed to plotting method.
        """
        mesh = self.get(cell_type)
        return mesh.plot_itk(**kwargs)

    def plot_pyvista(self, cell_type: str = None, **kwargs):
        """Plot data using pyvista.

        Parameters
        ----------
        cell_type : str, optional
            Cell type to plot.
        **kwargs
            These parameters are passed to plotting method.
        """
        mesh = self.get(cell_type)
        return mesh.plot_pyvista(**kwargs)

    @classmethod
    def from_mesh(cls, mesh: GenericMesh):
        """Convert from :class:`nanomesh.mesh.GenericMesh` to
        :class:`MeshContainer`.

        Parameters
        ----------
        mesh : Mesh
            Input mesh, must be a subclass of
            :class:`nanomesh.mesh.GenericMesh`.

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
        """Return instance of :class:`MeshContainer` from triangle results
        dict.

        Parameters
        ----------
        triangle_dict : dict
            Triangle triangulate output dictionary.

        Returns
        -------
        mesh : MeshContainer
        """
        points = triangle_dict['vertices']

        cells = {'triangle': triangle_dict['triangles']}

        cell_data = {}

        try:
            cell_data['physical'] = [
                triangle_dict['triangle_attributes'].squeeze()
            ]

            # Order must match order of cell_data
            cells['line'] = triangle_dict['edges']
            cell_data['physical'].append(
                triangle_dict['edge_markers'].squeeze())
        except KeyError:
            pass

        point_data = {}
        try:
            point_data['physical'] = triangle_dict['vertex_markers'].squeeze()
        except KeyError:
            pass

        mesh = cls(
            points=points,
            cells=cells,
            cell_data=cell_data,
            point_data=point_data,
        )

        mesh.triangle_dict = triangle_dict

        return mesh

    @classmethod
    def read(cls, *args, **kwargs) -> MeshContainer:
        """Wrapper for :func:`meshio.read`.

        For gmsh:

        - remaps `gmsh:physical` -> `physical`
        - remaps `gmsh:geometrical` -> `geometrical`

        Parameters
        ----------
        *args
            These parameters passed to reader
        **kwargs
            These parameters are passed to the reader

        Returns
        -------
        MeshContainer
        """
        from meshio import read
        mesh = read(*args, **kwargs)

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

        ret = cls(
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
        ret.prune_z_0()
        return ret

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
            These parameters are passed to :func:`meshio.write`.
        """
        from pathlib import Path

        from meshio import write
        from meshio._helpers import extension_to_filetypes

        if file_format is None:
            suffix = Path(filename).suffix

            try:
                file_types = extension_to_filetypes[suffix]
                file_format = file_types[0]
            except KeyError:
                raise IOError('Unknown extension, specify file format.')
            except IndexError:
                raise IOError('Specify file format ({file_types}).')

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
