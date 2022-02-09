from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy

from ._base import BaseMesh
from ._mixin import PruneZ0Mixin
from ._tetra import TetraMesh

if TYPE_CHECKING:
    import open3d


class TriangleMesh(BaseMesh, PruneZ0Mixin):
    cell_type = 'triangle'

    def plot(self, **kwargs):
        """Shortcut for `.plot_mpl` or `.plot_itk` depending on dimensions."""
        if self.dimensions == 2:
            return self.plot_mpl(**kwargs)
        else:
            return self.plot_itk(**kwargs)

    def plot_mpl(self, *args, **kwargs) -> plt.Axes:
        """Simple triangle mesh plot using `matplotlib`. See
        `.plotting.trianglemeshplot` for details.

        Parameters
        ----------
        *args
            Arguments passed to `.plotting.trianglemeshplot`
        **kwargs
            Keyword arguments passed to `.plotting.trianglemeshplot`

        Returns
        -------
        plt.Axes
        """
        from ..plotting import trianglemeshplot
        return trianglemeshplot(self, *args, **kwargs)

    def to_trimesh(self):
        """Return instance of `trimesh.Trimesh`."""
        import trimesh
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
        stacked_cells = np.hstack(np.insert(cells, 0, values=3,
                                            axis=1))  # type: ignore
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
    def from_trimesh(cls, mesh) -> TriangleMesh:
        """Return instance of `TriangleMesh` from trimesh."""
        return cls(points=mesh.vertices, cells=mesh.faces)

    @classmethod
    def from_triangle_dict(cls, dct: dict) -> TriangleMesh:
        """Return instance of `TriangleMesh` from triangle results dict."""
        from .mesh_container import MeshContainer
        mesh = MeshContainer.from_triangle_dict(dct)
        return mesh.get('triangle')

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

    def tetrahedralize(self, **kwargs) -> 'TetraMesh':
        """Tetrahedralize a contour.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `nanomesh.tetrahedralize`.

        Returns
        -------
        mesh : TetraMesh
            Tetrahedralized mesh.
        """
        from .._tetgen_wrapper import tetrahedralize
        mesh = tetrahedralize(self, **kwargs)
        return mesh
