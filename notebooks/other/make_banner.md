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

## Generate banner

This banner uses Nanomesh to generate the banner for Nanomesh.

```python
from nanomesh import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
```

Load the source image.

```python
banner_o = io.imread(r'source_text_o.png')
plane = Image(rgb2gray(banner_o))

seg = plane.binary_digitize(threshold=0.5)
seg.show()
```

### Contour finding

```python tags=[]
from nanomesh import Mesher2D

mesher = Mesher2D(seg.image)
mesher.generate_contour(max_edge_dist=10)
mesher.plot_contour()
```

### Create the mesh

And compare with original image.

```python
mesh = mesher.triangulate(opts='pq30a2500')
seg.compare_with_mesh(mesh)
```

### Create the banner using matplotlib

```python
plt.rcParams['image.cmap'] = 'gist_rainbow'

banner_no_o = io.imread(r'source_text_no_o.png')

tri_mesh = mesh.get('triangle')

points = tri_mesh.points
triangles = tri_mesh.cells
labels = tri_mesh.labels

x, y = points.T[::-1]

fig, ax = plt.subplots(figsize=(8, 2))
fig.tight_layout(pad=0)

ax.imshow(banner_no_o)
ax.axis('off')
ax.margins(0)

colors = np.arange(len(triangles))
np.random.shuffle(colors)  # mix up the colors
mask_o = (labels == 1)
ax.tripcolor(x, y, triangles=triangles, mask=mask_o, facecolors=colors)
ax.triplot(x, y, triangles=triangles, mask=mask_o, color='black', lw=0.5)

mask_rest = (labels == 2)
ax.triplot(x, y, triangles=triangles, mask=mask_rest, lw=0.5, alpha=0.8)

plt.savefig('banner.png', bbox_inches='tight')
```
