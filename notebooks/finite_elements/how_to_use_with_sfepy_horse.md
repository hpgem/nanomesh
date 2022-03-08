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

```python
%load_ext autoreload
%autoreload 2
%config InlineBackend.rc = {'figure.figsize': (10,6)}
%matplotlib inline
```

## Interfacing with SfePy

This example recreates one of the SfePy examples using data generated with Nanomesh.

The example is based on:
https://sfepy.org/doc-devel/examples/diffusion-laplace_fluid_2d.html


### Clean up old files

```python
from pathlib import Path

for fn in ('phi.vtk', 'regions.vtk', 'citroen.msh'):
    try:
        Path(fn).unlink()
    except:
        pass
    else:
        print(f'{fn} deleted')
```

### Load data

```python
from skimage.data import horse
from nanomesh import Image
from scipy import ndimage as ndi

data = horse()

plane = Image(~data)
plane = plane.apply(ndi.binary_fill_holes).astype(int)

plane.show(title=str(plane))
```

```python
from nanomesh import Mesher

mesher = Mesher(plane)
mesher.generate_contour(precision=5, max_edge_dist=25)
mesher.pad_contour(side='left', width=200, name='Left')
mesher.pad_contour(side='right', width=200, name='Right')
mesher.pad_contour(side='top', width=75, name='Top')
mesher.plot_contour()


nanomesh_mesh = mesher.triangulate(opts='pAq30a300')
nanomesh_mesh.plot()
```

### Purge feature

```python
triangles = nanomesh_mesh.get('triangle')
triangles.plot()
idx = triangles.cell_data['physical'] != 2
triangles.cell_data['physical'] = triangles.cell_data['physical'][idx]
triangles.cells = triangles.cells[idx]

triangles.plot()
```

```python
import numpy as np

triangles.points = np.flip(triangles.points, axis=1)
triangles.points[:,1] = -triangles.points[:,1]
triangles.plot()
```

### Set up sfepy config

Noe that the mesh hook uses the mesh container loaded using Nanomesh.

```python
from sfepy.discrete.fem.meshio import UserMeshIO
from sfepy.discrete.fem import Mesh


def mesh_hook(mesh, mode):
    if mode == 'read':
        points = triangles.points

        cells = triangles.cells
        cell_data = triangles.cell_data['physical']
        cell_description = ['2_3']

        mesh = Mesh.from_data('triangle', points, None,
                                      [cells], [cell_data], cell_description)
        return mesh

xmin, ymin = triangles.points.min(axis=0)
xmax, ymax = triangles.points.max(axis=0)

class mod:
    __file__ = 'nanomesh'  # dummy value
    filename_mesh = UserMeshIO(mesh_hook)

    # 2D vector defining far field velocity
    v0 = np.array([
        [-1.0],
        [0.0],
    ])

    materials = {
        'm': (
            {
                'v0': v0
            },
        ),
    }

    regions = {
        'Omega': 'all',
        'Gamma_Left': (f'vertices in (x < {xmin+0.1})', 'facet'),
        'Gamma_Right': (f'vertices in (x > {xmax-0.1})', 'facet'),
        'Gamma_Top': (f'vertices in (y > {ymax-0.1})', 'facet'),
        'Gamma_Bottom': (f'vertices in (y < {ymax+0.1})', 'facet'),
        'Vertex': ('vertex in r.Gamma_Left', 'vertex'),
    }

    fields = {
        'u': ('real', 1, 'Omega', 1),
    }

    variables = {
        'phi': ('unknown field', 'u', 0),
        'psi': ('test field', 'u', 'phi'),
    }

    # these EBCS prevent the matrix from being singular, see description
    ebcs = {
        'fix': ('Vertex', {'phi.0': 0.0}),
    }

    integrals = {
        'i': 2,
    }

    equations = {
        'Laplace equation':
            """dw_laplace.i.Omega( psi, phi )
             = dw_surface_ndot.i.Gamma_Left( m.v0, psi )
             + dw_surface_ndot.i.Gamma_Right( m.v0, psi )
             + dw_surface_ndot.i.Gamma_Top( m.v0, psi )
             + dw_surface_ndot.i.Gamma_Bottom( m.v0, psi )"""
    }

    solvers = {
        'ls': ('ls.scipy_direct', {}),
        'newton': ('nls.newton', {
            'i_max': 5,
            'eps_a': 1e-16,
        }),
    }

from sfepy.base.conf import ProblemConf

conf = ProblemConf.from_module(mod)
```

### Reading mesh into sfepy data

```python
from sfepy.discrete.fem import Mesh
trunk = conf.filename_mesh.get_filename_trunk()
_ = Mesh(trunk)
mesh = conf.filename_mesh.read(_)
# mesh = io.read(mesh, omit_facets=omit_facets)
mesh._set_shape_info()
```

### Solving the PDE with FEM

```python
from sfepy.applications import solve_pde

problem, variables = solve_pde(conf)
```

### Plot with Mayavi

Note that this will open a new window.

```python
from sfepy.postprocess.viewer import Viewer

out = 'phi.vtk'

problem.save_state(out, variables)

view = Viewer(out)
view(rel_scaling=2, is_scalar_bar=True,
     is_wireframe=True, colormap='viridis')
```

### Flow plot using pyvista

```python
from resview import pv_plot

class options:
    step = 0
    view_2d = True
    position_vector = None
    fields_map = []
    fields = [
        ('phi', 'p0'),
        ('phi', 't100:p0'),
             ]
    opacity = 1.
    show_edges = False
    warp = None
    factor = 1.0
    outline = False
    color_map = None
    show_scalar_bars = False
    show_labels = False

import pyvista as pv

plotter = pv.Plotter()
plotter = pv_plot([out], options=options, plotter=plotter, use_cache=False)
plotter.view_xy()
plotter.show(jupyter_backend='static')
```

### Loading the data back into Nanomesh

```python
from nanomesh import MeshContainer

meshc = MeshContainer.read(out)
meshc.plot('triangle')
```
