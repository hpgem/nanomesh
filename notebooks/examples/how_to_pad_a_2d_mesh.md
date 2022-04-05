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

## Pad a 2D mesh

This notebook shows how to pad a 2D mesh. Note that padding a 2d mesh is done on the contour (before triangulation).


### Generating some data

This cell generates a simple 2d plane.

If you want to use your own data, any numpy array can be passed to into a [`Image`](https://nanomesh.readthedocs.io/en/latest/nanomesh.plane.html#nanomesh.volume.Plane) object. Data stored as `.npy` can be loaded using `Image.load()`.

```python
from nanomesh.data import binary_blobs2d
from nanomesh import Image

data = binary_blobs2d(length=100, seed=42)

plane = Image(data)
plane.show()
```

### Generating the contour

The first step in mesh generation is to generate the contour.

```python
from nanomesh import Mesher2D

mesher = Mesher2D(plane)
mesher.generate_contour()
mesher.plot_contour(legend='fields')
```

### Padding different sides

The mesh can be padded using a similar API as 3d meshes. Each side (top/bottom, left/right) can be padded. A width must be specified.

Regions are labeled with a number. If no label is given, an arbitrary number is assigned. This is used to identify different regions in the mesh.

Padded areas can be given a name. Regions with the same name are assigned the same number.

```python
mesher.pad_contour(side='left', width=30, name='Left side')
mesher.pad_contour(side='right', width=40, name='Right side')
mesher.pad_contour(side='top', width=20, label=11)
mesher.pad_contour(side='bottom', width=50, label=11)
mesher.plot_contour(legend='fields')
```

### Generate triagonal mesh

Finally, generate the triagonal mesh.

Note that the legend specifies the name of the region if available in the `.fields` attribute.

```python
mesh = mesher.triangulate(opts='pAq30a100e')
mesh.plot(legend='floating', hide_labels=(0, ), linewidth=1)
```

### Labelling outer boundaries

The outer boundaries can be labeled using the `LineMesh.label_boundaries` method.

The default `.cell_data` key is `'physical'`. This can be overridden using the `key='...'` parameter. To label the top and bottom boundaries, use the `top`, `bottom` parameters.

```python
line_mesh = mesh.get('line')

line_mesh.label_boundaries(left='outer left', right='outer right')

# transfer labels back to MeshContainer
mesh.set_cell_data('line', 'physical', line_mesh.cell_data['physical'])
mesh.set_field_data('line', line_mesh.number_to_field)

mesh.plot(legend='floating', hide_labels=(0, ), linewidth=1)
```

### Padding left / right sides

The width, mesh quality, and label assigned to this this area can be defined.

This example shows how to double pad the left and right sides with different triangle sizes for each step.

```python
mesher = Mesher2D(plane)
mesher.generate_contour()

mesher.pad_contour(side='left', width=20, label=30, name='inner pad')
mesher.pad_contour(side='left', width=40, label=40, name='outer pad')

mesher.pad_contour(side='right', width=20, label=30, name='inner pad')
mesher.pad_contour(side='right', width=40, label=40, name='outer pad')

padded_mesh = mesher.triangulate(opts='pAq30a100e')

padded_mesh.plot('triangle', legend='fields')
```

### Spiral mesh

This pattern is infinitely extensible. The example below shows the flexibility of the method.

```python
from itertools import cycle
import numpy as np

mesher = Mesher2D(plane)
mesher.generate_contour()

choices = ('left', 'bottom', 'right', 'top')

for i, side in zip(range(1, 50), cycle(choices)):
    name = 'ABCDE'[i % 5]
    mesher.pad_contour(side=side, width=i, name=name)

spiral_mesh = mesher.triangulate(opts='pAq30a200e')

spiral_mesh.plot(legend='floating', hide_labels=(0, ), linewidth=0.5)
```
