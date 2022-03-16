---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: nanomesh
    language: python
    name: nanomesh
---

## Poisson equation with getfem

This is an adaptation of the tutorial linked [here](https://getfem-examples.readthedocs.io/en/latest/demo_unit_disk.html).


### Setup getfem

First, we must setup the path to the python module ([link](https://getfem.org/python/pygf.html#introduction)), so that getfem can be used in our Nanomesh environment.

We import getfem and generate a mesh to test if it works.

```python
import sys
sys.path.append('../../../getfem/interface/src/python/')

import getfem
m = getfem.Mesh('cartesian', range(0, 3), range(0,3))
m
```

### Poisson's equation

```python
import getfem as gf
import numpy as np

center = [0.0, 0.0]
radius = 1.0

mo = gf.MesherObject("ball", center, radius)

h = 0.1
K = 2
mesh = gf.Mesh("generate", mo, h, K)

outer_faces = mesh.outer_faces()
OUTER_BOUND = 1
mesh.set_region(OUTER_BOUND, outer_faces)

sl = gf.Slice(("none",), mesh, 1)

mfu = gf.MeshFem(mesh, 1)

elements_degree = 2
mfu.set_classical_fem(elements_degree)

mim = gf.MeshIm(mesh, pow(elements_degree, 2))

md = gf.Model("real")
md.add_fem_variable("u", mfu)

md.add_Laplacian_brick(mim, "u")

F = 1.0
md.add_fem_data("F", mfu)

md.set_variable("F", np.repeat(F, mfu.nbdof()))

md.add_source_term_brick(mim, "u", "F")

md.add_Dirichlet_condition_with_multipliers(mim, "u", elements_degree - 1, OUTER_BOUND)

md.solve()

# visualize
U = md.variable("u")

sl.export_to_vtk("u.vtk", "ascii", mfu, U, "U")
```

### Display result

```python
import pyvista as pv

p = pv.Plotter()
m = pv.read("u.vtk")
contours = m.contour()
p.add_mesh(m, show_edges=False)
p.add_mesh(contours, color="black", line_width=1)
p.add_mesh(m.contour(8).extract_largest(), opacity=0.1)
pts = m.points
p.show(window_size=[384, 384], cpos="xy")
```
