from __future__ import annotations

import numpy as np
import pyvista as pv

from .._doc import doc
from ._base import GenericMesh


@doc(GenericMesh,
     prefix='Data class for tetrahedral meshes',
     dim_points='3',
     dim_cells='4')
class TetraMesh(GenericMesh, cell_dim=4):
    cell_type = 'tetra'

    def to_open3d(self):
        """Return instance of `open3d.geometry.TetraMesh`."""
        import open3d
        return open3d.geometry.TetraMesh(
            vertices=open3d.utility.Vector3dVector(self.points),
            tetras=open3d.utility.Vector4iVector(self.cells))

    @classmethod
    def from_open3d(cls, mesh) -> TetraMesh:
        """Return instance of `TetraMesh` from open3d."""
        points = np.asarray(mesh.vertices)
        cells = np.asarray(mesh.tetras)
        return cls(points=points, cells=cells)

    @classmethod
    def from_pyvista_unstructured_grid(cls, grid: 'pv.UnstructuredGrid'):
        """Return infance of `TetraMesh` from `pyvista.UnstructuredGrid`."""
        assert grid.cells[0] == 4
        cells = grid.cells.reshape(grid.n_cells, 5)[:, 1:]
        points = np.array(grid.points)
        return cls(points=points, cells=cells)

    def plot(self, **kwargs):
        """Shortcut for `.plot_pyvista`."""
        return self.plot_pyvista(**kwargs)

    def plot_pyvista(self, **kwargs):
        """Show grid using `pyvista`.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `pyvista.Plotter().add_mesh`.
        """
        return self.to_pyvista_unstructured_grid().plot(**kwargs)

    def plot_submesh(
        self,
        index: int = None,
        along: str = 'x',
        invert: bool = False,
        show: bool = True,
        backend: str = None,
        **kwargs,
    ):
        """Show submesh using `pyvista`.

        Parameters
        ----------
        index : int, optional
            Index of where to cut the mesh. Shows all tetrahedra
            with cell center < index. Picks the half-way
            point along the axis by default.
        along : str, optional
            Direction along which to cut.
        invert : bool, optional
            Invert the cutting operation, and show all tetrahedra with
            cell center > index.
        show : bool, optional
            If true, show the plot
        **kwargs:
            Keyword arguments passed to `pyvista.Plotter().add_mesh`.

        plotter : `pyvista.Plotter`
            Return plotter instance.
        """
        grid = self.to_pyvista_unstructured_grid()

        # get cell centroids
        cells = grid.cells.reshape(-1, 5)[:, 1:]
        cell_center = grid.points[cells].mean(1)

        # extract cells below index
        axis = 'zyx'.index(along)

        if index is None:
            # pick half-way point
            i, j = axis * 2, axis * 2 + 2
            index = np.mean(grid.bounds[i:j])

        mask = cell_center[:, axis] < index

        if invert:
            mask = ~mask

        cell_ind = mask.nonzero()[0]
        subgrid = grid.extract_cells(cell_ind)

        plotter = pv.Plotter()
        plotter.add_mesh(subgrid, **kwargs)

        if show:
            plotter.show(jupyter_backend=backend)

        return plotter
