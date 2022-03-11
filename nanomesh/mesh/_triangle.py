from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy

from .._doc import doc
from .._tetgen_wrapper import tetrahedralize
from ._mesh import Mesh
from ._mixin import PruneZ0Mixin

if TYPE_CHECKING:
    import open3d

    from nanomesh import MeshContainer


@doc(Mesh,
     prefix='Data class for triangle meshes',
     dim_points='2 or 3',
     dim_cells='3')
class TriangleMesh(Mesh, PruneZ0Mixin, cell_dim=3):
    cell_type = 'triangle'

    def plot(self, **kwargs):
        """Shortcut for :meth:`TriangleMesh.plot_mpl` or
        :meth:`TriangleMesh.plot_itk` depending on dimensions."""
        if self.dimensions == 2:
            return self.plot_mpl(**kwargs)
        else:
            return self.plot_itk(**kwargs)

    def plot_mpl(self, *args, **kwargs) -> plt.Axes:
        """Simple triangle mesh plot using :mod::mod:`matplotlib`. See
        :func:`plotting.trianglemeshplot` for details.

        Parameters
        ----------
        *args
            Arguments passed to :func:`plotting.trianglemeshplot`
        **kwargs
            Keyword arguments passed to :func:`plotting.trianglemeshplot`

        Returns
        -------
        plt.Axes
        """
        from ..plotting import trianglemeshplot
        return trianglemeshplot(self, *args, **kwargs)

    def to_trimesh(self):
        """Return instance of :class:`trimesh.Trimesh`."""
        import trimesh
        return trimesh.Trimesh(vertices=self.points, faces=self.cells)

    def to_open3d(self) -> 'open3d.geometry.TriangleMesh':
        """Return instance of :class:`open3d.geometry.TriangleMesh`."""
        import open3d
        return open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.points),
            triangles=open3d.utility.Vector3iVector(self.cells))

    def to_polydata(self) -> 'pv.PolyData':
        """Return instance of :class:`pyvista.Polydata`."""
        points = self.points
        cells = self.cells
        # preprend 3 to indicate number of points per cell
        stacked_cells = np.hstack(np.insert(cells, 0, values=3,
                                            axis=1))  # type: ignore
        return pv.PolyData(points, stacked_cells, n_faces=len(cells))

    @classmethod
    def from_open3d(cls, mesh: 'open3d.geometry.TriangleMesh') -> TriangleMesh:
        """Return instance of :class:`TriangleMesh` from open3d."""
        points = np.asarray(mesh.vertices)
        cells = np.asarray(mesh.triangles)
        return cls(points=points, cells=cells)

    @classmethod
    def from_scipy(cls, mesh: 'scipy.spatial.qhull.Delaunay') -> TriangleMesh:
        """Return instance of :class:`TriangleMesh` from
        :class:`scipy.spatial.Delaunay` object."""
        points = mesh.points
        cells = mesh.simplices
        return cls(points=points, cells=cells)

    @classmethod
    def from_trimesh(cls, mesh) -> TriangleMesh:
        """Return instance of :class:`TriangleMesh` from :mod:`trimesh`."""
        return cls(points=mesh.vertices, cells=mesh.faces)

    @classmethod
    def from_triangle_dict(cls, dct: dict) -> TriangleMesh:
        """Return instance of :class:`TriangleMesh` from triangle results
        dict."""
        from .mesh_container import MeshContainer
        mesh = MeshContainer.from_triangle_dict(dct)
        return mesh.get('triangle')

    def optimize(self,
                 *,
                 method='CVT (block-diagonal)',
                 tol: float = 1.0e-3,
                 max_num_steps: int = 10,
                 **kwargs) -> TriangleMesh:
        """Optimize mesh using :mod:`optimesh`.

        Parameters
        ----------
        method : str, optional
            Method name
        tol : float, optional
            Tolerance
        max_num_steps : int, optional
            Maximum number of optimization steps.
        **kwargs
            Arguments to pass to :func:`optimesh.optimize_points_cells`

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

    @doc(tetrahedralize,
         prefix='Tetrahedralize mesh using :func:`tetrahedralize`')
    def tetrahedralize(self, **kwargs) -> 'MeshContainer':
        mesh = tetrahedralize(self, **kwargs)
        return mesh
