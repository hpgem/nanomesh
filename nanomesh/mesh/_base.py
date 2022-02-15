from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Sequence

import meshio
import numpy as np
import pyvista as pv

from ..region_markers import RegionMarker, RegionMarkerLike

registry: Dict[str, Any] = {}


class BaseMesh:
    cell_type: str = 'base'

    def __init__(self,
                 points: np.ndarray,
                 cells: np.ndarray,
                 fields: Dict[str, int] = None,
                 region_markers: List[RegionMarker] = None,
                 **cell_data):
        """Base class for meshes.

        Parameters
        ----------
        points : (m, n) np.ndarray[float]
            Array with points.
        cells : (i, j) np.ndarray[int]
            Index array describing the cells of the mesh.
        fields : Dict[str, int]:
            Mapping from field names to labels
        region_markers : List[RegionMarker], optional
            List of region markers used for assigning labels to regions.
            Defaults to an empty list.
        **cell_data
            Additional cell data. Argument must be a 1D numpy array
            matching the number of cells defined by `i`.
        """
        default_key = 'physical'
        if (not cell_data) or (default_key in cell_data):
            self.default_key = default_key
        else:
            self.default_key = list(cell_data.keys())[0]

        self.fields = dict(fields) if fields else {}

        self.points = points
        self.cells = cells
        self.field_to_number = MappingProxyType(self.fields)
        self.region_markers = [] if region_markers is None else region_markers
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
        """Mapping from numbers to fields, proxy to `.field_to_number`."""
        return MappingProxyType(
            {v: k
             for k, v in self.field_to_number.items()})

    def add_region_marker(self, region_marker: RegionMarkerLike):
        """Add marker to list of region markers.

        Parameters
        ----------
        region_marker : RegionMarkerLike
            Either a `RegionMarker` object or `(label, point)` tuple,
            where the label must be an `int` and the point a 2- or
            3-element numpy array.
        """
        if not isinstance(region_marker, RegionMarker):
            region_marker = RegionMarker(*region_marker)

        self.region_markers.append(region_marker)

    def add_region_markers(self, region_markers: Sequence[RegionMarkerLike]):
        """Add marker to list of region markers.

        Parameters
        ----------
        region_markers : List[RegionMarkerLike]
            List of region markers passed to `.add_region_marker`.
        """
        for region_marker in region_markers:
            self.add_region_marker(region_marker)

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            (self.cell_type, self.cells),
        ]

        mesh = meshio.Mesh(self.points, cells)

        for key, value in self.cell_data.items():
            mesh.cell_data[key] = [value]

        return mesh

    @classmethod
    def from_meshio(cls, mesh: 'meshio.Mesh'):
        """Return `BaseMesh` from meshio object."""
        points = mesh.points
        cells = mesh.cells[0].data
        cell_data = {}

        for key, value in mesh.cell_data.items():
            # PyVista chokes on ':ref' in cell_data
            key = key.replace(':ref', '-ref')
            cell_data[key] = value[0]

        return BaseMesh.create(points=points, cells=cells, **cell_data)

    @classmethod
    def create(cls, points, cells, **cell_data):
        """Class dispatcher."""
        cell_dimensions = cells.shape[1]
        if cell_dimensions == 2:
            item_class = registry['line']
        elif cell_dimensions == 3:
            item_class = registry['triangle']
        elif cell_dimensions == 4:
            item_class = registry['tetra']
        else:
            item_class = cls
        return item_class(points=points, cells=cells, **cell_data)

    def write(self, *args, **kwargs):
        """Simple wrapper around `meshio.write`."""
        self.to_meshio().write(*args, **kwargs)

    @classmethod
    def read(cls, filename, **kwargs):
        """Simple wrapper around `meshio.read`."""
        mesh = meshio.read(filename, **kwargs)
        return cls.from_meshio(mesh)

    def to_pyvista_unstructured_grid(self) -> 'pv.PolyData':
        """Return instance of `pyvista.UnstructuredGrid`.

        References
        ----------
        https://docs.pyvista.org/core/point-grids.html#pv-unstructured-grid-class-methods
        """
        return pv.from_meshio(self.to_meshio())

    def plot(self, **kwargs):
        raise NotImplementedError

    def plot_mpl(self, **kwargs):
        raise NotImplementedError

    def plot_itk(self, **kwargs):
        """Wrapper for `pyvista.plot_itk`.

        Parameters
        ----------
        **kwargs
            These parameters are passed to `pyvista.plot_itk`
        """
        return pv.plot_itk(self.to_meshio(), **kwargs)

    def plot_pyvista(self, **kwargs):
        """Wrapper for `pyvista.plot`.

        Parameters
        ----------
        **kwargs
            These parameters are passed to `pyvista.plot`
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
        np.ndarray
        """
        try:
            return self.cell_data[self.default_key]
        except KeyError:
            return np.ones(len(self.cells), dtype=int) * default_value

    @property
    def zero_labels(self):
        """Return zero labels as fallback."""
        return np.zeros(len(self.cells), dtype=int)

    @property
    def labels(self):
        """Shortcut for cell labels."""
        try:
            return self.cell_data[self.default_key]
        except KeyError:
            return self.zero_labels

    @labels.setter
    def labels(self, data: np.ndarray):
        """Shortcut for setting cell labels."""
        self.cell_data[self.default_key] = data

    @property
    def dimensions(self):
        """Return number of dimensions for point data."""
        return self.points.shape[1]

    def reverse_cell_order(self):
        """Reverse order of cells and cell data."""
        self.cells = self.cells[::-1]
        for key, data in self.cell_data.items():
            self.cell_data[key] = data[::-1]
