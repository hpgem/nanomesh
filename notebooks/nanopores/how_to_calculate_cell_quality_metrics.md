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

## Calculate mesh quality metrics

This notebook reads a mesh and plots different quality indicators:

- Minimum/maximum angle
- Ratio min/max edge length
- Ratio circumscribed to inscribed circle (largest circle fitting inside vs smallest circle fitting around a triangle)

The indicators are plotted on the mesh and as a histogram.

You can use your own mesh by supplying it to `MeshContainer.read()`.

```python
from nanomesh import metrics
from nanomesh import MeshContainer
```

```python
mesh = MeshContainer.read('out.msh')
triangle_mesh = mesh.get('triangle')

triangle_mesh.plot()
```

### Metrics

Quality metrics are available through the `metrics` submodule, for example to access the area for each face:

```python
metrics.area(triangle_mesh)
```

### Minumum and maximum cell angles

`nanomesh.metrics` includes convenience functions for plotting histograms and colored 2d meshes. The `ax` object can be re-used to overlay the mesh triangles.

```python
plot_kwargs = {
    'linewidth': 1,
    'show_labels': ('Pore', ),
    'colors': ('tab:orange', ),
    'flip_xy': False,
    'legend': 'all',
}

metrics.histogram(triangle_mesh, metric='min_angle')
ax = metrics.plot2d(triangle_mesh, metric='min_angle')
triangle_mesh.plot_mpl(ax, **plot_kwargs)
```

```python
metrics.histogram(triangle_mesh, metric='max_angle')
ax = metrics.plot2d(triangle_mesh, metric='max_angle')
triangle_mesh.plot_mpl(ax, **plot_kwargs)
```

### Ratio between radii

Another useful metric is the ratio between the inner and outer radius. For more info, see this [link](https://www.geogebra.org/m/VRE3Dyrn).

```python
metrics.histogram(triangle_mesh, metric='radius_ratio')
ax = metrics.plot2d(triangle_mesh, metric='radius_ratio')
triangle_mesh.plot_mpl(ax, **plot_kwargs)
```

### Ratio between longest and shortest edge

```python
metrics.histogram(triangle_mesh, metric='max_min_edge_ratio')
ax = metrics.plot2d(triangle_mesh, metric='max_min_edge_ratio')
triangle_mesh.plot_mpl(ax, **plot_kwargs)
```

### Calculate and export all metrics

This way they can be viewed in another program like Paraview.

```python
metrics.calculate_all_metrics(triangle_mesh, inplace=True)
triangle_mesh.write("mesh_quality.msh", file_format='gmsh22', binary=False)
```
