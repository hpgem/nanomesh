from dataclasses import dataclass

import meshio
import numpy as np
import open3d
import pyvista as pv
import trimesh
from trimesh import remesh


@dataclass
class BaseMeshContainer:
    vertices: np.ndarray
    faces: np.ndarray
    labels: np.ndarray = None  # face markers

    def write(self, **kwargs):
        """Simple wrapper around `meshio.write`."""
        self.to_meshio().write(**kwargs)

    def plot_itk(self):
        """Wrapper for `pyvista.plot_itk`."""
        pv.plot_itk(self.to_meshio())

    @property
    def face_centers(self):
        """Return centers of faces (mean of vertices)."""
        return self.vertices[self.faces].mean(axis=1)


class TwoDMeshContainer(BaseMeshContainer):
    def to_trimesh(self) -> 'trimesh.Trimesh':
        """Return instance of `trimesh.Trimesh`."""
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            ('triangle', self.faces),
        ]

        mesh = meshio.Mesh(self.vertices, cells)

        if self.labels is not None:
            mesh.cell_data['labels'] = [self.labels]

        return mesh

    def to_open3d(self) -> 'open3d.geometry.TriangleMesh':
        """Return instance of `open3d.geometry.TriangleMesh`."""
        import open3d
        return open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            triangles=open3d.utility.Vector3iVector(self.faces))

    @classmethod
    def from_open3d(
            cls, mesh: 'open3d.geometry.TriangleMesh') -> 'TwoDMeshContainer':
        """Return instance of `TwoDMeshContainer` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh') -> 'TwoDMeshContainer':
        """Return instance of `TwoDMeshContainer` from open3d."""
        return cls(vertices=mesh.vertices, faces=mesh.faces)

    @classmethod
    def from_triangle_dict(cls, dct: dict) -> 'TwoDMeshContainer':
        """Return instance of `TwoDMeshContainer` from trimesh results dict."""
        vertices = dct['vertices']
        faces = dct['triangles']
        labels = dct['vertex_markers'].reshape(-1)
        return cls(vertices=vertices, faces=faces, labels=labels)


class SurfaceMeshContainer(BaseMeshContainer):
    def to_trimesh(self) -> 'trimesh.Trimesh':
        """Return instance of `trimesh.Trimesh`."""
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            ('triangle', self.faces),
        ]

        mesh = meshio.Mesh(self.vertices, cells)
        return mesh

    def to_open3d(self) -> 'open3d.geometry.TriangleMesh':
        """Return instance of `open3d.geometry.TriangleMesh`."""
        import open3d
        return open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            triangles=open3d.utility.Vector3iVector(self.faces))

    def to_polydata(self) -> 'pv.PolyData':
        """Return instance of `pyvista.Polydata`."""
        vertices = self.vertices
        faces = self.faces
        # preprend 3 to indicate number of points per face
        stacked_faces = np.hstack(np.insert(faces, 0, values=3, axis=1))
        return pv.PolyData(vertices, stacked_faces, n_faces=len(faces))

    @classmethod
    def from_open3d(
            cls,
            mesh: 'open3d.geometry.TriangleMesh') -> 'SurfaceMeshContainer':
        """Return instance of `SurfaceMeshContainer` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh') -> 'SurfaceMeshContainer':
        """Return instance of `SurfaceMeshContainer` from open3d."""
        return cls(vertices=mesh.vertices, faces=mesh.faces)

    def simplify(self, n_faces: int) -> 'SurfaceMeshContainer':
        """Simplify triangular mesh using `open3d`.

        Parameters
        ----------
        n_faces : int
            Simplify mesh until this number of faces is reached.

        Returns
        -------
        SurfaceMeshContainer
        """
        mesh_o3d = self.to_open3d()
        simplified_o3d = mesh_o3d.simplify_quadric_decimation(int(n_faces))
        return SurfaceMeshContainer.from_open3d(simplified_o3d)

    def simplify_by_vertex_clustering(self,
                                      voxel_size: float = 1.0
                                      ) -> 'SurfaceMeshContainer':
        """Simplify mesh geometry using vertex clustering.

        Parameters
        ----------
        voxel_size : float, optional
            Size of the target voxel within which vertices are grouped.

        Returns
        -------
        SurfaceMeshContainer
        """
        mesh_in = self.to_open3d()
        mesh_smp = mesh_in.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=open3d.geometry.SimplificationContraction.Average)

        return SurfaceMeshContainer.from_open3d(mesh_smp)

    def smooth(self, iterations: int = 50) -> 'SurfaceMeshContainer':
        """Smooth mesh using the Taubin filter in `trimesh`.

        The advantage of the Taubin algorithm is that it avoids
        shrinkage of the object.

        Parameters
        ----------
        iterations : int, optional
            Number of smoothing operations to apply

        Returns
        -------
        SurfaceMeshContainer
        """
        mesh_tri = self.to_trimesh()
        smoothed_tri = trimesh.smoothing.filter_taubin(mesh_tri,
                                                       iterations=iterations)
        return SurfaceMeshContainer.from_trimesh(smoothed_tri)

    def optimize(self,
                 *,
                 method='CVT (block-diagonal)',
                 tol: float = 1.0e-3,
                 max_num_steps: int = 10,
                 **kwargs) -> 'SurfaceMeshContainer':
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
        SurfaceMeshContainer
        """
        import optimesh
        verts, faces = optimesh.optimize_points_cells(
            X=self.vertices,
            cells=self.faces,
            method=method,
            tol=tol,
            max_num_steps=max_num_steps,
            **kwargs,
        )
        return SurfaceMeshContainer(vertices=verts, faces=faces)

    def subdivide(self,
                  max_edge: int = 10,
                  iters: int = 10) -> 'SurfaceMeshContainer':
        """Subdivide triangles until the maximum edge size is reached.

        Parameters
        ----------
        max_edge : int, optional
            Max triangle edge distance.
        iters : int, optional
            Maximum number of iterations of iterations.
        """
        verts, faces = remesh.subdivide_to_size(self.vertices,
                                                self.faces,
                                                max_edge=max_edge,
                                                max_iter=10)
        return SurfaceMeshContainer(vertices=verts, faces=faces)

    def tetrahedralize(self, **kwargs) -> 'VolumeMeshContainer':
        """Tetrahedralize a contour.

        Parameters
        ----------
        label : int
            Label of the contour
        **kwargs
            Keyword arguments passed to `tetgen.TetGen`

        Returns
        -------
        VolumeMeshContainer
        """
        import tetgen
        kwargs.setdefault('order', 1)
        polydata = self.to_polydata()
        tet = tetgen.TetGen(polydata)
        tet.tetrahedralize(**kwargs)
        grid = tet.grid
        return VolumeMeshContainer.from_pyvista_unstructured_grid(grid)


class VolumeMeshContainer(BaseMeshContainer):
    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            ('tetra', self.faces),
        ]

        mesh = meshio.Mesh(self.vertices, cells)
        return mesh

    def to_open3d(self):
        """Return instance of `open3d.geometry.TetraMesh`."""
        import open3d
        return open3d.geometry.TetraMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            tetras=open3d.utility.Vector4iVector(self.faces))

    def to_pyvista_unstructured_grid(self) -> 'pv.PolyData':
        """Return instance of `pyvista.UnstructuredGrid`."""
        return pv.from_meshio(self.to_meshio())

    @classmethod
    def from_open3d(cls, mesh):
        """Return instance of `VolumeMeshContainer` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.tetras)
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_pyvista_unstructured_grid(cls, grid: 'pv.UnstructuredGrid'):
        """Return infance of `VolumeMeshContainer` from
        `pyvista.UnstructuredGrid`."""
        assert grid.cells[0] == 4
        faces = grid.cells.reshape(grid.n_cells, 5)[:, 1:]
        vertices = np.array(grid.points)
        return cls(vertices=vertices, faces=faces)

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
        **kwargs:
            Keyword arguments passed to `pyvista.Plotter().add_mesh`.
        """
        kwargs.setdefault('color', 'lightgray')

        plotter = pv.Plotter()

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

        plotter.add_mesh(subgrid, **kwargs)
        plotter.show()
