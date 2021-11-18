from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista as pv
import scipy
import trimesh

from . import mesh2d, mesh3d
from .mpl.meshplot import _legend_with_triplot_fix

if TYPE_CHECKING:
    import open3d


class BaseMesh:
    _cell_type: str = 'base'

    def __init__(self, points: np.ndarray, cells: np.ndarray, **cell_data):
        self._label_key = 'labels'

        self.points = points
        self.cells = cells
        self.region_markers: List[Tuple[int, np.ndarray]] = []
        self.cell_data = cell_data

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            (self._cell_type, self.cells),
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
            key = key.replace(':ref', 'Ref')
            cell_data[key] = value[0]

        return BaseMesh.create(points=points, cells=cells, **cell_data)

    @classmethod
    def create(cls, points, cells, **cell_data):
        """Class dispatcher."""
        cell_dimensions = cells.shape[1]
        if cell_dimensions == 2:
            item_class = LineMesh
        elif cell_dimensions == 3:
            item_class = TriangleMesh
        elif cell_dimensions == 4:
            item_class = TetraMesh
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
            Extra keyword arguments passed to `pyvista.plot_itk`
        """
        pv.plot_itk(self.to_meshio(), **kwargs)

    def plot_pyvista(self, **kwargs):
        """Wrapper for `pyvista.plot`.

        Parameters
        ----------
        **kwargs
            Extra keyword arguments passed to `pyvista.plot`
        """
        pv.plot(self.to_meshio(), **kwargs)

    @property
    def cell_centers(self):
        """Return centers of cells (mean of corner points)."""
        return self.points[self.cells].mean(axis=1)

    @property
    def zero_labels(self):
        """Return zero labels as fallback."""
        return np.zeros(len(self.cells), dtype=int)

    @property
    def labels(self):
        """Shortcut for cell labels."""
        if not self.cell_data:
            return self.zero_labels

        return self.cell_data[self._label_key]

    @labels.setter
    def labels(self, data: np.array):
        """Shortcut for setting cell labels."""
        self.cell_data[self._label_key] = data

    @property
    def dimensions(self):
        """Return number of dimensions for point data."""
        return self.points.shape[1]


class LineMesh(BaseMesh):
    _cell_type = 'line'

    def plot_mpl(self,
                 ax: plt.Axes = None,
                 label: str = None,
                 fields: dict = None,
                 **kwargs) -> plt.Axes:
        """Simple line mesh plot using `matplotlib`.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to use for plotting.
        label : str, optional
            Label of cell data item to plot.
        **kwargs
            Extra keyword arguments passed to `.mpl.lineplot`

        Returns
        -------
        plt.Axes
        """
        from .mpl.lineplot import lineplot

        if not ax:
            fig, ax = plt.subplots()

        if fields is None:
            fields = {}

        labels = self.cell_data.get(label, self.zero_labels)

        for label in np.unique(labels):
            vert_x, vert_y = self.points.T

            name = fields.get(label, None)

            lineplot(
                ax,
                x=vert_y,
                y=vert_x,
                lines=self.cells,
                mask=labels != label,
                label=name,
            )

        ax.set_title(self._cell_type)
        ax.axis('equal')

        if fields:
            ax.legend()

        return ax


class TriangleMesh(BaseMesh):
    _cell_type = 'triangle'

    def drop_third_dimension(self):
        """Drop third dimension coordinates if present.

        For compatibility, sometimes a column with zeroes is added. This
        method drops that column.
        """
        TOL = 1e-9

        if self.dimensions < 3:
            return

        if not np.all(np.abs(self.points[:, 1]) < TOL):
            raise ValueError(
                'Coordinates in third dimension are not all equal to zero.')

        self.points = self.points[:, 0:2]

    def plot(self, **kwargs):
        """Shortcut for `.plot_mpl`."""
        if self.dimensions == 2:
            self.plot_mpl(**kwargs)
        else:
            self.plot_itk(**kwargs)

    def plot_mpl(self,
                 ax: plt.Axes = None,
                 label: str = None,
                 fields: dict = None,
                 **kwargs) -> plt.Axes:
        """Simple mesh plot using `matplotlib`.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to use for plotting.
        label : str, optional
            Label of cell data item to plot.
        **kwargs
            Extra keyword arguments passed to `ax.triplot`

        Returns
        -------
        plt.Axes
        """
        if not ax:
            fig, ax = plt.subplots()

        if fields is None:
            fields = {}

        labels = self.cell_data.get(label, self.zero_labels)

        for label in np.unique(labels):
            vert_x, vert_y = self.points.T

            name = fields.get(label, None)

            ax.triplot(vert_y,
                       vert_x,
                       triangles=self.cells,
                       mask=labels != label,
                       label=name)

        ax.set_title(self._cell_type)
        ax.axis('equal')

        if fields:
            ax.legend()
            _legend_with_triplot_fix(ax)

        return ax

    def to_trimesh(self) -> 'trimesh.Trimesh':
        """Return instance of `trimesh.Trimesh`."""
        return trimesh.Trimesh(vertices=self.points, faces=self.cells)

    def to_open3d(self) -> 'open3d.geometry.TriangleMesh':
        """Return instance of `open3d.geometry.TriangleMesh`."""
        import open3d
        return open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.points),
            triangles=open3d.utility.Vector3iVector(self.cells))

    def to_polydata(self) -> 'pv.PolyData':
        """Return instance of `pyvista.Polydata`."""
        points = self.points
        cells = self.cells
        # preprend 3 to indicate number of points per cell
        stacked_cells = np.hstack(np.insert(cells, 0, values=3, axis=1))
        return pv.PolyData(points, stacked_cells, n_faces=len(cells))

    @classmethod
    def from_open3d(cls, mesh: 'open3d.geometry.TriangleMesh') -> TriangleMesh:
        """Return instance of `TriangleMesh` from open3d."""
        points = np.asarray(mesh.vertices)
        cells = np.asarray(mesh.triangles)
        return cls(points=points, cells=cells)

    @classmethod
    def from_scipy(cls, mesh: 'scipy.spatial.qhull.Delaunay') -> TriangleMesh:
        """Return instance of `TriangleMesh` from `scipy.spatial.Delaunay`
        object."""
        points = mesh.points
        cells = mesh.simplices
        return cls(points=points, cells=cells)

    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh') -> TriangleMesh:
        """Return instance of `TriangleMesh` from trimesh."""
        return cls(points=mesh.vertices, cells=mesh.faces)

    @classmethod
    def from_triangle_dict(cls, dct: dict) -> TriangleMesh:
        """Return instance of `TriangleMesh` from triangle results dict."""
        points = dct['vertices']
        cells = dct['triangles']

        # TODO: Add this to `.point_data`
        # metadata = {}
        # try:
        #     metadata['is_boundary_point'] = dct['vertex_markers'].squeeze()
        # except KeyError:
        #     pass

        mesh = cls(points=points, cells=cells)

        try:
            mesh.edges = dct['edges']
            mesh.edge_markers = dct['edge_markers']
        except KeyError:
            pass

        return mesh

    def simplify(self, n_cells: int) -> TriangleMesh:
        """Simplify triangular mesh using `open3d`.

        Parameters
        ----------
        n_cells : int
            Simplify mesh until this number of cells is reached.

        Returns
        -------
        TriangleMesh
        """
        mesh_o3d = self.to_open3d()
        simplified_o3d = mesh_o3d.simplify_quadric_decimation(int(n_cells))
        return TriangleMesh.from_open3d(simplified_o3d)

    def simplify_by_point_clustering(self,
                                     voxel_size: float = 1.0) -> TriangleMesh:
        """Simplify mesh geometry using point clustering.

        Parameters
        ----------
        voxel_size : float, optional
            Size of the target voxel within which points are grouped.

        Returns
        -------
        TriangleMesh
        """
        import open3d
        mesh_in = self.to_open3d()
        mesh_smp = mesh_in.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=open3d.geometry.SimplificationContraction.Average)

        return TriangleMesh.from_open3d(mesh_smp)

    def smooth(self, iterations: int = 50) -> TriangleMesh:
        """Smooth mesh using the Taubin filter in `trimesh`.

        The advantage of the Taubin algorithm is that it avoids
        shrinkage of the object.

        Parameters
        ----------
        iterations : int, optional
            Number of smoothing operations to apply

        Returns
        -------
        TriangleMesh
        """
        mesh_tri = self.to_trimesh()
        smoothed_tri = trimesh.smoothing.filter_taubin(mesh_tri,
                                                       iterations=iterations)
        return TriangleMesh.from_trimesh(smoothed_tri)

    def optimize(self,
                 *,
                 method='CVT (block-diagonal)',
                 tol: float = 1.0e-3,
                 max_num_steps: int = 10,
                 **kwargs) -> TriangleMesh:
        """Optimize mesh using `optimesh`.

        Parameters
        ----------
        method : str, optional
            Method name
        tol : float, optional
            Tolerance
        max_num_steps : int, optional
            Maximum number of optimization steps.
        **kwargs
            Arguments to pass to `optimesh.optimize_points_cells`

        Returns
        -------
        TriangleMesh
        """
        import optimesh
        points, cells = optimesh.optimize_points_cells(
            X=self.points,
            cells=self.cells,
            method=method,
            tol=tol,
            max_num_steps=max_num_steps,
            **kwargs,
        )
        return TriangleMesh(points=points, cells=cells)

    def subdivide(self, max_edge: int = 10, iters: int = 10) -> TriangleMesh:
        """Subdivide triangles."""
        from trimesh import remesh
        points, cells = remesh.subdivide(self.points, self.cells)
        return TriangleMesh(points=points, cells=cells)

    def tetrahedralize(self,
                       region_markers: List[Tuple[int, np.ndarray]] = None,
                       **kwargs) -> 'TetraMesh':
        """Tetrahedralize a contour.

        Parameters
        ----------
        region_markers : list, optional
            List of region markers. If not defined, automatically
            generate regions.
        **kwargs
            Keyword arguments passed to `nanomesh.tetgen.tetrahedralize`.

        Returns
        -------
        TetraMesh
        """
        import tempfile

        if region_markers is None:
            region_markers = []

        region_markers.extend(self.region_markers)

        from nanomesh import tetgen
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'nanomesh.smesh')
            tetgen.write_smesh(path, self, region_markers=region_markers)
            tetgen.tetrahedralize(path, **kwargs)
            ele_path = path.with_suffix('.1.ele')
            return TetraMesh.read(ele_path)

    def pad(self, **kwargs) -> TriangleMesh:
        """Pad a mesh.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `nanomesh.mesh2d.helpers.pad`
        """
        return mesh2d.pad(self, **kwargs)

    def pad3d(self, **kwargs) -> TriangleMesh:
        """Pad a 3d mesh.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `nanomesh.mesh3d.helpers.pad`
        """
        return mesh3d.pad(self, **kwargs)


class TetraMesh(BaseMesh):
    _cell_type = 'tetra'

    def to_open3d(self):
        """Return instance of `open3d.geometry.TetraMesh`."""
        import open3d
        return open3d.geometry.TetraMesh(
            vertices=open3d.utility.Vector3dVector(self.points),
            tetras=open3d.utility.Vector4iVector(self.cells))

    @classmethod
    def from_open3d(cls, mesh):
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
        self.plot_pyvista(**kwargs)

    def plot_pyvista(self, **kwargs):
        """Show grid using `pyvista`.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `pyvista.Plotter().add_mesh`.
        """
        self.to_pyvista_unstructured_grid().plot(**kwargs)

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
