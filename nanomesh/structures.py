import math

import numpy as np
import pygalmesh

from nanomesh.generator import Generator

# The actual dimension of the cells are 480 nm x 680 nm x 480 nm
# However we specify xdim=ydim=zdim because the perdiodic mesher can only
# handle cubic domains
# dimension of the domain used by CGAL to mesh the data
XDIM, YDIM, ZDIM = 0.48, 0.48, 0.48


class Pore3D(pygalmesh.DomainBase):
    def __init__(self):
        super().__init__()

        # instantiate the generator
        self.gen = Generator(a=680, c=480, radius=0.24 * 680)

        # Possible rotation/transformation of the coordinate system
        theta = 0.5 * math.pi
        c = math.cos(theta)
        s = math.sin(theta)
        trans = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ])

        # numbre of voxel in each direction
        self.size = [48, 68, 48]

        # size of the voxel (in nm)
        self.res = [10, 10, 10]

        # generate the data cube
        self.data = self.gen.generate_vect(
            self.size,
            self.res,
            transform=trans,
            bin_val=[0., 1.],
        )

        # size used during the meshing by the orcale
        # must be a perfect cube
        self.x = np.linspace(0, XDIM, self.data.shape[0])
        self.y = np.linspace(0, YDIM, self.data.shape[1])
        self.z = np.linspace(0, ZDIM, self.data.shape[2])

    def eval(self, x):
        """Orale function used during the meshing."""

        # get the indices
        ix = np.digitize(x[0] % XDIM, self.x) - 1
        iy = np.digitize(x[1] % YDIM, self.y) - 1
        iz = np.digitize(x[2] % ZDIM, self.z) - 1

        # evaluate the data cube
        if self.data[ix, iy, iz] == 0:
            out = -1.
        else:
            out = 1.
        return out


class FullCube(pygalmesh.DomainBase):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        return -1
