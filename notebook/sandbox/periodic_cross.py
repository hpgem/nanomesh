import numpy as np
import pygalmesh
import pyvista as pv


class Cross(pygalmesh.DomainBase):
    def __init__(self):
        super().__init__()

        self.npts = 11
        self.x = np.linspace(0, 1, self.npts)
        self.data = np.zeros((self.npts, self.npts, self.npts))
        idx = (self.x > 0.25) & (self.x < 0.75)
        all_idx = list(range(self.npts))
        self.data[np.ix_(idx, idx, all_idx)] = 1.
        self.data[np.ix_(idx, all_idx, idx)] = 1.
        self.data[np.ix_(all_idx, idx, idx)] = 1.

    def eval(self, x):
        ix = np.argmin(np.abs(x[0] - self.x))
        iy = np.argmin(np.abs(x[1] - self.x))
        iz = np.argmin(np.abs(x[2] - self.x))
        if self.data[ix, iy, iz] == 1:
            return -1.
        else:
            return 0.


mesh = pygalmesh.generate_periodic_mesh(
    Cross(),
    [0, 0, 0, 1, 1, 1],
    max_cell_circumradius=0.05,
    min_facet_angle=30,
    max_radius_surface_delaunay_ball=0.05,
    max_facet_distance=0.025,
    max_circumradius_edge_ratio=2.0,
    number_of_copies_in_output=1,
    make_periodic=True,
    # odt=True,
    # lloyd=True,
    verbose=False,
)

pv.plot_itk(mesh)
