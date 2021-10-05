from pathlib import Path

import matplotlib.pyplot as plt
import meshio
import numpy as np
import open3d
import pyvista as pv
import scipy
import trimesh
from trimesh import remesh


class MeshContainer:
    _element_type: str = ''

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, **metadata):
        self._label_key = 'labels'

        self.vertices = vertices
        self.faces = faces

        metadata.setdefault(self._label_key,
                            np.zeros(len(self.faces), dtype=int))
        self.metadata = metadata

    def to_meshio(self) -> 'meshio.Mesh':
        """Return instance of `meshio.Mesh`."""
        cells = [
            (self._element_type, self.faces),
        ]

        mesh = meshio.Mesh(self.vertices, cells)

        for key, value in self.metadata.items():
            mesh.cell_data[key] = [value]

        return mesh

    @classmethod
    def from_meshio(cls, mesh: 'meshio.Mesh'):
        """Return `MeshContainer` from meshio object."""
        vertices = mesh.points
        faces = mesh.cells[0].data
        metadata = {}

        for key, value in mesh.cell_data.items():
            # PyVista chokes on ':ref' in metadata
            key = key.replace(':ref', 'Ref')
            metadata[key] = value[0]

        return MeshContainer.create(vertices=vertices, faces=faces, **metadata)

    @classmethod
    def create(cls, vertices, faces, **metadata):
        """Class dispatcher."""
        n = faces.shape[1]
        if n == 3:
            item_class = TriangleMesh
        elif n == 4:
            item_class = TetraMesh
        else:
            item_class = cls
        return item_class(vertices=vertices, faces=faces, **metadata)

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

    @property
    def face_centers(self):
        """Return centers of faces (mean of vertices)."""
        return self.vertices[self.faces].mean(axis=1)

    @property
    def labels(self):
        """Shortcut for face labels."""
        return self.metadata[self._label_key]

    @labels.setter
    def labels(self, data: np.array):
        """Shortcut for setting face labels."""
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
        has_third_dimension = self.vertices.shape[1] == 3
        if has_third_dimension:
            self.vertices = self.vertices[:, 0:2]

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
            vert_x, vert_y = self.vertices.T
            ax.triplot(vert_y,
                       vert_x,
                       triangles=self.faces,
                       mask=self.labels != label,
                       label=label)

        ax.axis('equal')

        return ax

    def to_trimesh(self) -> 'trimesh.Trimesh':
        """Return instance of `trimesh.Trimesh`."""
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

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
    def from_open3d(cls,
                    mesh: 'open3d.geometry.TriangleMesh') -> 'TriangleMesh':
        """Return instance of `TriangleMesh` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_scipy(cls,
                   mesh: 'scipy.spatial.qhull.Delaunay') -> 'TriangleMesh':
        """Return instance of `TriangleMesh` from `scipy.spatial.Delaunay`
        object."""
        vertices = mesh.points
        faces = mesh.simplices
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh') -> 'TriangleMesh':
        """Return instance of `TriangleMesh` from trimesh."""
        return cls(vertices=mesh.vertices, faces=mesh.faces)

    @classmethod
    def from_triangle_dict(cls, dct: dict) -> 'TriangleMesh':
        """Return instance of `TriangleMesh` from trimesh results dict."""
        vertices = dct['vertices']
        faces = dct['triangles']
        vertex_markers = dct['vertex_markers'].reshape(-1)
        return cls(vertices=vertices,
                   faces=faces,
                   vertex_markers=vertex_markers)

    def simplify(self, n_faces: int) -> 'TriangleMesh':
        """Simplify triangular mesh using `open3d`.

        Parameters
        ----------
        n_faces : int
            Simplify mesh until this number of faces is reached.

        Returns
        -------
        TriangleMesh
        """
        mesh_o3d = self.to_open3d()
        simplified_o3d = mesh_o3d.simplify_quadric_decimation(int(n_faces))
        return TriangleMesh.from_open3d(simplified_o3d)

    def simplify_by_vertex_clustering(self,
                                      voxel_size: float = 1.0
                                      ) -> 'TriangleMesh':
        """Simplify mesh geometry using vertex clustering.

        Parameters
        ----------
        voxel_size : float, optional
            Size of the target voxel within which vertices are grouped.

        Returns
        -------
        TriangleMesh
        """
        mesh_in = self.to_open3d()
        mesh_smp = mesh_in.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=open3d.geometry.SimplificationContraction.Average)

        return TriangleMesh.from_open3d(mesh_smp)

    def smooth(self, iterations: int = 50) -> 'TriangleMesh':
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
                 **kwargs) -> 'TriangleMesh':
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
        verts, faces = optimesh.optimize_points_cells(
            X=self.vertices,
            cells=self.faces,
            method=method,
            tol=tol,
            max_num_steps=max_num_steps,
            **kwargs,
        )
        return TriangleMesh(vertices=verts, faces=faces)

    def subdivide(self, max_edge: int = 10, iters: int = 10) -> 'TriangleMesh':
        """Subdivide triangles."""
        verts, faces = remesh.subdivide(self.vertices, self.faces)
        return TriangleMesh(vertices=verts, faces=faces)

    def tetrahedralize(self,
                       region_markers: dict = None,
                       **kwargs) -> 'TetraMesh':
        """Tetrahedralize a contour.

        Parameters
        ----------
        region_markers : dict, optional
            Dictionary of region markers. If not defined, automatically
            generate regions.
        **kwargs
            Keyword arguments passed to `nanomesh.tetgen.tetrahedralize`.

        Returns
        -------
        TetraMesh
        """
        import tempfile

        if region_markers is None:
            region_markers = {}

        from nanomesh import tetgen
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'nanomesh.smesh')
            tetgen.write_smesh(path, self, region_markers=region_markers)
            tetgen.tetrahedralize(path, **kwargs)
            ele_path = path.with_suffix('.1.ele')
            return TetraMesh.read(ele_path)

    def pad(self, **kwargs) -> 'TriangleMesh':
        """Pad a mesh.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to `nanomesh.mesh_utils.pad`
        """
        from nanomesh.mesh_utils import pad
        return pad(self, **kwargs)


class TetraMesh(MeshContainer):
    _element_type = 'tetra'

    def to_open3d(self):
        """Return instance of `open3d.geometry.TetraMesh`."""
        import open3d
        return open3d.geometry.TetraMesh(
            vertices=open3d.utility.Vector3dVector(self.vertices),
            tetras=open3d.utility.Vector4iVector(self.faces))

    @classmethod
    def from_open3d(cls, mesh):
        """Return instance of `TetraMesh` from open3d."""
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.tetras)
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def from_pyvista_unstructured_grid(cls, grid: 'pv.UnstructuredGrid'):
        """Return infance of `TetraMesh` from `pyvista.UnstructuredGrid`."""
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
        show: bool = True,
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
            plotter.show()

        return plotter
