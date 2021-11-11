from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import meshio
import numpy as np
import open3d
import pyvista as pv
import scipy
import trimesh
from trimesh import remesh

from . import mesh2d, mesh3d


class MeshContainer:
    _element_type: str = ''

    def __init__(self, points: np.ndarray, cells: np.ndarray, **metadata):
        self._label_key = 'labels'

        self.points = points
        self.cells = cells
        self.region_markers: List[Tuple[int, np.ndarray]] = []

        metadata.setdefault(self._label_key,
                            np.zeros(len(self.cells), dtype=int))
        self.metadata = metadata

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            (self._element_type, self.cells),
        ]

        mesh = meshio.Mesh(self.points, cells)

        for key, value in self.metadata.items():
            mesh.cell_data[key] = [value]

        return mesh

    @classmethod
    def from_meshio(cls, mesh: 'meshio.Mesh'):
        """Return `MeshContainer` from meshio object."""
        points = mesh.points
        cells = mesh.cells[0].data
        metadata = {}

        for key, value in mesh.cell_data.items():
            # PyVista chokes on ':ref' in metadata
            key = key.replace(':ref', 'Ref')
            metadata[key] = value[0]

        return MeshContainer.create(points=points, cells=cells, **metadata)

    @classmethod
    def create(cls, points, cells, **metadata):
        """Class dispatcher."""
        n = cells.shape[1]
        if n == 3:
            item_class = TriangleMesh
        elif n == 4:
            item_class = TetraMesh
        else:
            item_class = cls
        return item_class(points=points, cells=cells, **metadata)

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

    def plot_itk(self):
        """Wrapper for `pyvista.plot_itk`."""
        pv.plot_itk(self.to_meshio())

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
    def labels(self):
        """Shortcut for cell labels."""
        return self.metadata[self._label_key]

    @labels.setter
    def labels(self, data: np.array):
        """Shortcut for setting cell labels."""
        self.metadata[self._label_key] = data

    @property
    def unique_labels(self):
        """Return unique labels."""
        return np.unique(self.labels)


class TriangleMesh(MeshContainer):
    _element_type = 'triangle'

    def drop_third_dimension(self):
        """Drop third dimension coordinates if present.

        For compatibility, sometimes a column with zeroes is added. This
        method drops that column.
        """
        has_third_dimension = self.points.shape[1] == 3
        if has_third_dimension:
            self.points = self.points[:, 0:2]

    def plot(self, ax: plt.Axes = None) -> plt.Axes:
        """Simple mesh plot using `matplotlib`.

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Axes to use for plotting.

        Returns
        -------
        ax : matplotlib.Axes
        """
        if not ax:
            fig, ax = plt.subplots()

        for label in self.unique_labels:
            vert_x, vert_y = self.points.T
            ax.triplot(vert_y,
                       vert_x,
                       triangles=self.cells,
                       mask=self.labels != label,
                       label=label)

        ax.axis('equal')

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
        """Return instance of `TriangleMesh` from trimesh results dict."""
        points = dct['vertices']
        cells = dct['triangles']
        return cls(points=points, cells=cells)

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


class TetraMesh(MeshContainer):
    _element_type = 'tetra'

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
