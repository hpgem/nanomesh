from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict

import meshio
import numpy as np
import pyvista as pv

from .._doc import DocFormatterMeta, doc
from ..region_markers import RegionMarkerList


@doc(prefix='Generic mesh class', dim_points='n', dim_cells='j')
class GenericMesh(object, metaclass=DocFormatterMeta):
    """{prefix}.

    Parameters
    ----------
    points : (m, {dim_points}) numpy.ndarray[float]
        Array with points.
    cells : (i, {dim_cells}) numpy.ndarray[int]
        Index array describing the cells of the mesh.
    fields : Dict[str, int]:
        Mapping from field names to labels
    region_markers : RegionMarkerList, optional
        List of region markers used for assigning labels to regions.
        Defaults to an empty list.
    **cell_data
        Additional cell data. Argument must be a 1D numpy array
        matching the number of cells defined by `i`.
    """
    _registry: Dict[int, Any] = {}
    cell_type: str = 'base'

    def __init_subclass__(cls, cell_dim: int, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cell_dim] = cls

    def __new__(cls, points: np.ndarray, cells: np.ndarray, *args, **kwargs):
        cell_dim = cells.shape[1]
        subclass = cls._registry.get(cell_dim, cls)
        return super().__new__(subclass)

    def __init__(self,
                 points: np.ndarray,
                 cells: np.ndarray,
                 *,
                 fields: Dict[str, int] = None,
                 region_markers: RegionMarkerList = None,
                 **cell_data):
        default_key = 'physical'
        if (not cell_data) or (default_key in cell_data):
            self.default_key = default_key
        else:
            self.default_key = list(cell_data.keys())[0]

        self.fields = dict(fields) if fields else {}

        self.points = points
        self.cells = cells
        self.field_to_number = MappingProxyType(self.fields)
        self.region_markers = RegionMarkerList()
        if region_markers:
            self.region_markers.extend(region_markers)
        self.cell_data = cell_data

    def __repr__(self, indent: int = 0):
        """Canonical string representation."""
        region_markers = set(m.name if m.name else m.label
                             for m in self.region_markers)
        s = (
            f'{self.__class__.__name__}(',
            f'    points = {self.points.shape},',
            f'    cells = {self.cells.shape},',
            f'    fields = {tuple(self.field_to_number.keys())},',
            f'    region_markers = {tuple(region_markers)},',
            f'    cell_data = {tuple(self.cell_data.keys())},',
            ')',
        )

        prefix = ' ' * indent
        return f'\n{prefix}'.join(s)

    @property
    def number_to_field(self):
        """Mapping from numbers to fields, proxy to
        :attr:`{classname}.field_to_number`."""
        return MappingProxyType(
            {v: k
             for k, v in self.field_to_number.items()})

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of :func:`meshio.Mesh`."""
        cells = [
            (self.cell_type, self.cells),
        ]

        mesh = meshio.Mesh(self.points, cells)

        for key, value in self.cell_data.items():
            mesh.cell_data[key] = [value]

        return mesh

    @classmethod
    def from_meshio(cls, mesh: 'meshio.Mesh'):
        """Return :class:`{classname}` from meshio object."""
        points = mesh.points
        cells = mesh.cells[0].data
        cell_data = {}

        for key, value in mesh.cell_data.items():
            # PyVista chokes on ':ref' in cell_data
            key = key.replace(':ref', '-ref')
            cell_data[key] = value[0]

        return GenericMesh(points=points, cells=cells, **cell_data)

    def write(self, *args, **kwargs):
        """Simple wrapper around :func:`meshio.write`."""
        self.to_meshio().write(*args, **kwargs)

    @classmethod
    def read(cls, filename, **kwargs):
        """Simple wrapper around :func:`meshio.read`."""
        mesh = meshio.read(filename, **kwargs)
        return cls.from_meshio(mesh)

    def to_pyvista_unstructured_grid(self) -> 'pv.PolyData':
        """Return instance of :class:`pyvista.UnstructuredGrid`.

        References
        ----------
        https://docs.pyvista.org/core/point-grids.html#pv-unstructured-grid-class-methods
        """
        return pv.from_meshio(self.to_meshio())

    def plot(self, **kwargs):
        raise NotImplementedError(
            f'Not implemented for {self.__class__.__name__}, '
            'use one of the specific subclasses.')

    def plot_mpl(self, **kwargs):
        raise NotImplementedError(
            f'Not implemented for {self.__class__.__name__}, '
            'use one of the specific subclasses.')

    def plot_itk(self, **kwargs):
        """Wrapper for :func:`pyvista.plot_itk`.

        Parameters
        ----------
        **kwargs
            These parameters are passed to :func:`pyvista.plot_itk`
        """
        return pv.plot_itk(self.to_meshio(), **kwargs)

    def plot_pyvista(self, **kwargs):
        """Wrapper for :func:`pyvista.plot`.

        Parameters
        ----------
        **kwargs
            These parameters are passed to :func:`pyvista.plot`
        """
        return pv.plot(self.to_meshio(), **kwargs)

    @property
    def cell_centers(self):
        """Return centers of cells (mean of corner points)."""
        return self.points[self.cells].mean(axis=1)

    def get_cell_data(self, key: str, default_value: float = 0) -> np.ndarray:
        """Get cell data with optional default value.

        Parameters
        ----------
        key : str
            Key of the cell data to retrieve.
        default_value : float, optional
            Optional default value (if cell data does not exist)

        Returns
        -------
        numpy.ndarray
        """
        try:
            return self.cell_data[self.default_key]
        except KeyError:
            return np.ones(len(self.cells), dtype=int) * default_value

    @property
    def zero_labels(self) -> np.ndarray:
        """Return zero labels as fallback."""
        return np.zeros(len(self.cells), dtype=int)

    @property
    def labels(self) -> np.ndarray:
        """Shortcut for cell labels."""
        try:
            return self.cell_data[self.default_key]
        except KeyError:
            return self.zero_labels

    @labels.setter
    def labels(self, data: np.ndarray):
        """Shortcut for setting cell labels.

        Updates :attr:`{classname}.cell_data`.
        """
        self.cell_data[self.default_key] = data

    @property
    def dimensions(self) -> int:
        """Return number of dimensions for point data."""
        return self.points.shape[1]

    def reverse_cell_order(self):
        """Reverse order of cells and cell data.

        Updates :attr:`{classname}.cell_data`.
        """
        self.cells = self.cells[::-1]
        for key, data in self.cell_data.items():
            self.cell_data[key] = data[::-1]
