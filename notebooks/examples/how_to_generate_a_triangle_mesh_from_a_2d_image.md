---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: nanomesh
    language: python
    name: nanomesh
---

```python
%load_ext autoreload
%autoreload 2
%config InlineBackend.rc = {'figure.figsize': (10,6)}
%matplotlib inline
```

## Generate a 2D triangular mesh

This notebook shows how to mesh a 2D image:

1. Load and visualize a 2D data image
2. Generate a 2D triangle mesh
3. Visualize the mesh
4. Relabel the regions
5. Export the mesh to other formats


### Load and vizualize the data

This example uses generated sample data from `nanomesh.data`.

If you want to use your own data, any (2D) numpy array can be passed to into a [`Image`](https://nanomesh.readthedocs.io/en/latest/nanomesh.plane.html#nanomesh.plane.Plane) object. Data stored as `.npy` can be loaded using `Image.load()`.

```python
from nanomesh import Image
from nanomesh.data import binary_blobs2d

data = binary_blobs2d(seed=12345)

plane = Image(data)
plane.show()
```

### Generate mesh


Meshes are generated using the `Mesher2D` class. Meshing consists of two steps:

1. Contour finding (using the [`find_contours`](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours) function
2. Triangulation (using the [`triangle`](https://rufat.be/triangle/) library)

Contour finding uses the [marching cubes algorithm](https://en.wikipedia.org/wiki/Marching_cubes) to wrap all the pores in a polygon. `max_edge_dist=5` splits up long edges in the contour, so that no two points are further than 5 pixels apart. `level` is directly passed to `find_contours` and specifies the level at which the contour is generated. In this case, we set it to the threshold value determined above.

```python tags=[]
from nanomesh import Mesher2D

mesher = Mesher2D(plane)
mesher.generate_contour(max_edge_dist=3)

mesher.plot_contour()
```

The next step is to use the contours to initialize triangulation.

Triangulation options can be specified through the `opts` keyword argument. This example uses `q30` to generate a quality mesh with angles > 30°, and `a100` to set a maximum triangle size of 100 pixels. For more options, see [here](https://rufat.be/triangle/API.html#triangle.triangulate).

```python
mesh = mesher.triangulate(opts='q30a100')
```

Triangulation returns a `MeshContainer` dataclass that can be used for various operations, for example comparing it with the original image:

```python
plane.compare_with_mesh(mesh)
```

### Region markers

By default, regions are split up in the *background* and *features*. Feature regions are grouped by the label 1.

```python
mesher.contour.region_markers
```

To label regions sequentially, set `group_regions=False`:

```python
mesher = Mesher2D(plane)
mesher.generate_contour(max_edge_dist=3, group_regions=False)

mesher.plot_contour()
```

Notice that each feature has been given a unique name:

```python
mesher.contour.region_markers
```

These labels will assigned to each triangle in the corresponding region after triangulation. These are stored in `mesh.cell_data` of the `MeshContainer`. This container stores a single set of points, and both the line segments (`LineMesh`) and triangles (`TriangleMesh`). To extract the triangle cells only, use `MeshContainer.get('triangle')`. this returns a class that is simpler to work with.

The cell below shows how to use this to access the cell data for the triangle cells in the mesh.

```python
mesh = mesher.triangulate(opts='q30a100')
mesh.get('triangle').cell_data
```

### Field data

Field data can be used to associate names with the values in the cell data. These are shown in the legend of mesh data (i.e. in the plots above). The field data is stored in the `.field_data` attribute. Because the data are somewhat difficult to use in this state, the properties `.field_to_number` and `.number_to_field` can be used to access the mapping per cell type.

```python
mesh.number_to_field
```

To update the values, you can update `.field_data` directory, or use `.set_field_data`. Note that field names are shared between cell types. For example, to relabel the cells data:



```python
fields = {}
for k, v in mesh.number_to_field['triangle'].items():
    v = v.replace('background', 'Silicon')
    v = v.replace('feature', 'Pore')
    fields[k] = v

mesh.set_field_data('triangle', fields)
mesh.number_to_field
```

Plotting the mesh now shows the fields in the legend. Note that the fields are also saved when exported to a format that supports them (e.g. *gmsh*).

```python
mesh.plot(lw=1, color_map={0: 'lightgray'}, legend='floating')
```

### Interoperability

The `MeshContainer` object can also be used to convert to various other library formats, such as:

- [`trimesh.Trimesh`](https://trimsh.org/trimesh.base.html#trimesh.base.Trimesh)
- [`pyvista.UnstructuredGrid`](https://docs.pyvista.org/examples/00-load/create-unstructured-surface.html)
- [`meshio.Mesh`](https://docs.pyvista.org/examples/00-load/create-unstructured-surface.html)

First, we must extract the triangle data:

```python
triangle_mesh = mesh.get('triangle')

pv_mesh = triangle_mesh.to_pyvista_unstructured_grid()
trimesh_mesh = triangle_mesh.to_trimesh()
meshio_mesh = triangle_mesh.to_meshio()
```

To save the data, use the `.write` method. This is essentially a very thin wrapper around `meshio`, equivalent to `meshio.write(...)`.

```python
mesh.write('out.msh', file_format='gmsh22', binary=False)
```
