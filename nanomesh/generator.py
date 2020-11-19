
import logging
import numpy as np
import math

# Pores in X direction are at
# y,z= (0,0) (a,0) (0,c) (a,c) and (a/2,c/2)

# Pores in Z direction are at
# x,y = (0,a/4) (c,a/4) and (c/2,3a/4)


# Generator for a theoretical structure
class Generator(object):

    def __init__(self, a, c, r):
        # Long crystal axis, usually a = sqrt(c)
        self.a = a
        self.c = c
        # Pore radius
        self.r = r

    def generate(self, sizes, resolution, transform=None):
        result = np.zeros(sizes)

        # Invert all the transform
        if transform is not None:
            transform = np.linalg.inv(transform)

        for ix, iy, iz in np.ndindex(result.shape):
            r = np.asarray([ix * resolution[0], iy * resolution[1], iz * resolution[2]])
            if transform is not None:
                r = transform.dot(r)
            # The pores in Z direction need to be offset by either +a/4 or -a/4
            if self.check_pore(r[2], r[1]) or self.check_pore(r[0], r[1] + self.a/4):
                result[ix, iy, iz] = 1.0
        return result

    def check_pore(self, xz, y):
        # Check for a pore in the xy or zy plane.
        # This places pores at the corners
        # xz,y = (0,0), (0,a), (c,0), (c,a)
        # and the centre (xz,y) = (c/2,a/2)

        xzr = xz % self.c
        yr = y % self.a

        r2 = self.r * self.r

        xzr2 = xzr * xzr
        yr2 = yr * yr

        xzrc2 = (self.c - xzr) * (self.c - xzr)
        yra2 = (self.a - yr) * (self.a - yr)

        if yr2 + xzr2 < r2 or yra2 + xzr2 < r2 or yr2 + xzrc2 < r2 or yra2 + xzrc2 < r2:
            return True
        else:
            xzrh = xzr - self.c / 2
            yrh = yr - self.a / 2
            return yrh * yrh + xzrh * xzrh < r2
