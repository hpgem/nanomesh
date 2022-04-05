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

## Fluid dynamics with SfePy

In this notebook we recreate one of the [SfePy examples](https://sfepy.org/doc-devel/examples/) using a mesh generated with Nanomesh.

The [original example](https://sfepy.org/doc-devel/examples/diffusion-laplace_fluid_2d.html), describes a Laplace equation that models the flow of “dry water” around an obstacle shaped like a Citroen CX. [Fluid dynamics](https://en.wikipedia.org/wiki/Fluid_dynamics) are commonly used to model air flow around an object. We don't have an image of a car, but let's see how far we get with modeling aero-dynamics of a [horse](https://en.wikipedia.org/wiki/Horse). :-)

Prerequisites:
- [Sfepy](https://sfepy.org/doc-devel/installation.html)
- [Mayavi](https://docs.enthought.com/mayavi/mayavi/installation.html) (optional, for one of the plots)



### Load data

This example uses the [skimage horse sample data](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.horse).

The data are inverted and small gaps in the tail are filled.

```python
from skimage.data import horse
from nanomesh import Image
from scipy import ndimage as ndi

data = horse()

plane = Image(~data)
plane = plane.apply(ndi.binary_fill_holes).astype(int)

plane.show(title=plane)
```

### Generating the mesh

The shape of the object is simplified to reduce the number of triangles. Note that the bbox was expanded to leave some head/tail room for the partial derivatives.

```python
from nanomesh import Mesher

mesher = Mesher(plane)
mesher.bbox = [[  -75,   -200],
        [327,   -200],
        [327, 599],
        [  -75, 599]]
mesher.generate_contour(precision=2, max_edge_dist=15)
mesher.plot_contour()

nanomesh_mesh = mesher.triangulate(opts='pAq30a300')
nanomesh_mesh.plot()
```

### Prepare mesh for SfePy

In the next cell we extract the triangle mesh and prepare the mesh for SfePy.

1. Remove triangles representing the horse using the `Mesh.remove_cells()` method.
2. Flip and rotate the coordinates. This ensures the mesh has the correct orientation.

```python
import numpy as np

triangles = nanomesh_mesh.get('triangle')
triangles.remove_cells(label=2, key='physical')

triangles.points = np.flip(triangles.points, axis=1)
triangles.points[:,1] = -triangles.points[:,1]

triangles.plot()
```

### Running SfePy the easy way

At this stage, the data can also be saved to SfePy-supported data type, and run using the command-line options.

```python
triangles.write('horse.vtk')
```

### Running Sfepy the interactive way

To run SfePy in the Jupyter notebook, we need to set up the config interactively.

The next cell sets up the config for SfePy.

The `problem_desc` class essentially defines what is known as the [**problem description file**](https://sfepy.org/doc-devel/users_guide.html#problem-description-file).

The mesh from Nanomesh is converted using the `mesh_hook`.

```python
from sfepy.discrete.fem.meshio import UserMeshIO
from sfepy.discrete.fem import Mesh

def mesh_hook(mesh, mode):
    if mode == 'read':
        points = triangles.points

        cells = triangles.cells
        cell_data = triangles.cell_data['physical']
        cell_description = ['2_3']

        mesh = Mesh.from_data(
            'triangle',
            points,
            None,  
            [cells],
            [cell_data],
            cell_description
        )
        return mesh

xmin, ymin = triangles.points.min(axis=0)
xmax, ymax = triangles.points.max(axis=0)

class problem_desc:
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

conf = ProblemConf.from_module(problem_desc)
```

### Bonus: Accessings SfePy mesh type

Now that the config has been defined, the next cell contains a little snippet to load the SfePy mesh from the config.

```python
from sfepy.discrete.fem import Mesh
trunk = conf.filename_mesh.get_filename_trunk()
mesh = conf.filename_mesh.read(Mesh(trunk))
mesh._set_shape_info()
mesh
```

### Solving the PDE with FEM

Solving the partial differential equations with SfePy is straightforward:

```python
from sfepy.applications import solve_pde

problem, variables = solve_pde(conf)
```

### Plot with Mayavi

The data can be stored to a file, and then displayed in Mayavi.

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

To view the streamlines, we use the `pv_plot` function from Sfepy. This uses `pyvista` as the renderer.

This is normally a command-line tool, so we must mimick the plotting options.

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


plotter = pv_plot([out], options=options, use_cache=False)
plotter.view_xy()
plotter.show(jupyter_backend='static')
```

### Loading the data back into Nanomesh

The data can be loaded back into Nanomesh. Note that we must flip back the coordinates to get the correct orientation.

```python
from nanomesh import MeshContainer

mesh_container = MeshContainer.read(out)

mesh_container.points[:,1] = -mesh_container.points[:,1]
mesh_container.points = np.flip(mesh_container.points, axis=1)

mesh_container.plot('triangle')
```
