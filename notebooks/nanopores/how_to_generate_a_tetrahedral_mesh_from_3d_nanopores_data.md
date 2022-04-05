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

## Generate a 3D tetrahedral mesh

This notebook shows how to mesh a 3D volume:

1. Load and visualize a volume
2. Apply image filters and segment image
3. Generate a 3D surface mesh
4. Visualize and export the mesh to other formats

```python
import pyvista as pv
from skimage import filters
import numpy as np
```

### Load and vizualize the data

This example uses nanopore sample data from `nanomesh.data`.

If you want to use your own data, any numpy array can be passed to into a [`Image`](https://nanomesh.readthedocs.io/en/latest/nanomesh.volume.html#nanomesh.volume.Volume) object. Data stored as `.npy` can be loaded using `Image.load()`.

```python
from nanomesh import Volume
from nanomesh.data import nanopores3d

data = nanopores3d()

vol = Volume(data)
vol.show_slice()
```

For this example, select a subvolume using `.select_subvolume` and downscale the image to keep the cpu times in check.

```python
from skimage.transform import rescale

subvol = vol.select_subvolume(
    ys=(0, 100),
    xs=(0, 100),
).apply(rescale, scale=0.5)
subvol.show_slice()
```

Nanomesh makes use of [`itkwidgets`](https://github.com/InsightSoftwareConsortium/itkwidgets) to render the volumes.

```python
subvol.show()
```

### Filter and segment the data

Image segmentation is a way to label the pixels of different regions of interest in an image. In this example, we are interested in separating the bulk material (Si) from the nanopores. In the image, the Si is bright, and the pores are dark.

First, we apply a [`gaussian filter`](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian) to smooth out some of the image noise to get a cleaner segmentation.

```python
subvol_gauss = subvol.gaussian(sigma=1)
subvol_gauss.show_slice(x=12)
```

`scikit-image` contains a useful function to [try all threshold finders on the data](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.try_all_threshold). These methods analyse the contrast histogram and try to find the optimal value to separate which parts of the image belong to each domain.

Since the function only works on a single slice, we first select a slice using the `.select_plane` method.

```python
from skimage import filters

plane = subvol_gauss.select_plane(x=12)
plane.try_all_threshold(figsize=(5, 10))
```

We will use the `li` method, because it gives nice separation.

The threshold value is used to segment the image using [`np.digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html#numpy-digitize).

```python
subvol_seg = subvol_gauss.binary_digitize(threshold='minimum')
subvol_seg = subvol_seg.invert_contrast()
subvol_seg.show_slice()
```

### Generate 3d tetragonal mesh

Meshes can be generated using the `Mesher` class. Meshing consists of two steps:

1. Contour finding (using the [`marching_cubes`](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes) function
2. Triangulation (using the [`tetgen`](https://tetgen.pyvista.org/) library)

`Mesher` requires a segmented image. `generate_contour()` wraps all domains of the image corresponding to that label. Here, 1 corresponds to the bulk (Si) material.

Meshing options are defined in the [tetgen documentation](http://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html#sec35). These can be specified using the `opts` parameter. The default options are `opts='-pAq1.2`:

- `-A`: Assigns attributes to tetrahedra in different regions.
- `-p`: Tetrahedralizes a piecewise linear complex (PLC).
- `-q`: Refines mesh (to improve mesh quality).

Also useful:

- `-a`: Applies a maximum tetrahedron volume constraint. Don't make `-a` too small, or the algorithm will take a very long time to complete. If this parameter is left out, the triangles will keep growing without limit.

```python
%%time

from nanomesh import Mesher

mesher = Mesher(subvol_seg)
mesher.generate_contour()
mesh = mesher.tetrahedralize(opts='-pAq')
```

Tetrahedralization returns a `TetraMesh` dataclass that can be used for various operations, for example showing the result using `itkwidgets`:

```python
mesh.plot_pyvista(jupyter_backend='static', show_edges=True)
```

### Using region markers

By default, the region attributes are assigned automatically by `tetgen`. Tetrahedra in each enclosed region will be assigned a new label sequentially.

Region markers are used to assign attributes to tetrahedra in different regions. After tetrahedralization, the region markers will 'flood' the regions up to the defined boundaries. The elements of the resulting mesh are marked according to the region they belong to (`tetras.metadata['tetgenRef']`.

You can view the existing region markers by looking at the `.region_markers` attribute on the contour.

```python
mesher.contour.region_markers
```

It is possible to set your own attributes using the `region_markers` parameter. These are a list of `RegionMarker` objects stored in a `RegionMarkerList` container. You can define your own, or you can use the methods on the `mesher.contour.region_markers` attribute.

For example, to relabel the pores sequentially:

```python
mesher.contour.region_markers = mesher.contour.region_markers.label_sequentially(
    2, fmt_name='pore{}')
mesher.contour.region_markers
```

Then re-run tetrahedralization.

```python
%%time
import numpy as np

mesh = mesher.tetrahedralize(opts='-pAq')

for label in mesher.contour.region_markers.labels:
    num = np.sum(mesh.cell_data['tetgen:ref'] == label)
    print(f'{num} tetrahedra with attribute `{label}`')
```

Note that all connected regions have a different label now.

```python
mesh.plot_pyvista(jupyter_backend='static', show_edges=True)
```

### Mesh evaluation

The mesh can be evaluated using the `metrics` module. This example shows how to calculate all metrics and plot them on a section through the generated mesh.

```python
from nanomesh import metrics

tetra_mesh = mesh.get('tetra')

metrics_dict = metrics.calculate_all_metrics(tetra_mesh, inplace=True)
metrics_dict
```

Using the `.plot_submesh()` method, any array that is present in the metadata can be plotted. `plot_submesh()` is flexible, in that it can show a slice through the mesh as defined using `index`, `along`, and `invert`. Extra keyword arguments, such as `show_edges` and `lighting` are passed on to [`Plotter.add_mesh()`](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.add_mesh.html?highlight=add_mesh).

```python
tetra_mesh.plot_submesh(
    along='x',
    index=15,
    scalars='min_angle',
    show_edges=True,
    lighting=True,
    backend='static',
)
```

### Interoperability

The `TetraMesh` object can also be used to convert to various other library formats, such as:

- [`trimesh.open3d`](http://www.open3d.org/docs/release/python_api/open3d.geometry.TetraMesh.html#open3d.geometry.TetraMesh)
- [`pyvista.UnstructuredGrid`](https://docs.pyvista.org/examples/00-load/create-unstructured-surface.html)
- [`meshio.Mesh`](https://docs.pyvista.org/examples/00-load/create-unstructured-surface.html)


To save the data, use the `.write` method. This is essentially a very thin wrapper around `meshio`, equivalent to `meshio_mesh.write(...)`.

```python
tetra_mesh.write('volume_mesh.msh', file_format='gmsh22', binary=False)
```
