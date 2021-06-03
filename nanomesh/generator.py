from typing import List

import numpy as np

# Pores in X direction are at
# y,z= (0,0) (a,0) (0,c) (a,c) and (a/2,c/2)

# Pores in Z direction are at
# x,y = (0,a/4) (c,a/4) and (c/2,3a/4)


# Generator for a theoretical structure
class Generator(object):
    """Generator for densities of theoretical structures."""
    def __init__(self, a: float, c: float, radius: float):
        """Initializer with specific dimensions.

        Parameters
        ----------
        a : float
            The size of the long axis (typically 680 nm)
        c : float
            The size of the short axis, all structures have `c = a/sqrt(2)`
            (typically about 481nm)
        radius : float
            The radius of the pores (typically `0.20 < r/a < 0.24`) is assumed
            to be less than `min(c/2, a/4)`
        """
        # Long crystal axis, usually `a = sqrt(c)`
        self.a = a
        self.c = c

        # Pore radius
        self.r = radius

    def generate(self,
                 sizes: List[int],
                 resolution: List[float],
                 transform=None,
                 bin_val: List[float] = [0., 1.]) -> np.ndarray:
        """Generate a volume image of the structure.

        Parameters
        ----------
        sizes : List[int]
            The size (3 integers) of the resulting volume in voxels
        resolution : List[float]
            The resolution of each voxel (xray images are at 10nm or 20nm)
        transform : None, optional
            Optional 3D transformation matrix to map from the coordinate
            system of the structure to the coordinate system of the volume.
            It should have determinant of +-1 for the resolution to remain
            correct.
        bin_val : List[float, float], optional
            Binary values (low / high)

        Returns
        -------
        np.ndarray
            A ndarray of size sizes filled with either 1 (air) or 0 (silicon)
        """
        result = bin_val[0] * np.ones(sizes)

        # Invert all the transform
        if transform is not None:
            transform = np.linalg.inv(transform)

        for ix, iy, iz in np.ndindex(result.shape):
            r = np.asarray(
                [ix * resolution[0], iy * resolution[1], iz * resolution[2]])
            if transform is not None:
                r = transform.dot(r)
            # The pores in Z direction need to be offset by either +a/4 or -a/4
            if self.check_pore(r[2], r[1]) or self.check_pore(
                    r[0], r[1] + self.a / 4):
                result[ix, iy, iz] = bin_val[1]
        return result

    def generate_vect(self,
                      sizes: List[int],
                      resolution: List[float],
                      transform=None,
                      bin_val=[0., 1.]) -> np.ndarray:
        """Generate a volume image of the structure. Vectorized version of
        `.generate`.

        Parameters
        ----------
        sizes : List[int]
            The size (3 integers) of the resulting volume in voxels
        resolution : List[float]
            The resolution of each voxel (xray images are at 10nm or 20nm)
        transform : None, optional
            Optional 3D transformation matrix to map from the coordinate
            system of the structure to the coordinate system of the volume.
            It should have determinant of +-1 for the resolution to remain
            correct.
        bin_val : List[float, float], optional
            Binary values (low / high)

        Returns
        -------
        np.ndarray
            A ndarray of size sizes filled with either 1 (air) or 0 (silicon)
        """
        result = bin_val[0] * np.ones(sizes)

        # Invert all the transform
        if transform is not None:
            transform = np.linalg.inv(transform)

        r = np.array(list(np.ndindex(result.shape)))
        r = np.array(resolution) * r

        if transform is not None:
            r = r @ transform.T

        shift = np.array([0, +self.a / 4])
        cond = self.check_pore_vect(
            r[:, [2, 1]]) + self.check_pore_vect(r[:, [0, 1]] + shift)
        cond = cond.reshape(*sizes)
        result[cond] = bin_val[1]

        return result

    def check_pore(self, xz: float, y: float) -> bool:
        """Helper function to determine whether a 2D coordinate is inside a
        pore.

        This method checks for a pore in the `xy` or `zy` plane.
        This places pores at the corners, i.e.
        `xz,y = (0,0), (0,a), (c,0), (c,a)`, and the centre
        `(xz,y) = (c/2,a/2)`.

        Parameters
        ----------
        xz : float
            The coordinate along the axis of length c (COPS coordinates X
            and Z)
        y : float
            The coordinate along the axis of length a (COPS coordinate Y)

        Returns
        -------
        bool
            True if the given coordinate is within an air pore.
        """
        # Compute the coordinate within the regular unit cell.
        xzr = xz % self.c
        yr = y % self.a

        r2 = self.r * self.r

        xzr2 = xzr * xzr
        yr2 = yr * yr

        xzrc2 = (self.c - xzr) * (self.c - xzr)
        yra2 = (self.a - yr) * (self.a - yr)

        if ((yr2 + xzr2 < r2 or yra2 + xzr2 < r2) or (yr2 + xzrc2 < r2)
                or (yra2 + xzrc2 < r2)):
            return True
        else:
            xzrh = xzr - self.c / 2
            yrh = yr - self.a / 2
            return yrh * yrh + xzrh * xzrh < r2

    def check_pore_vect(self, data: np.ndarray) -> np.ndarray:
        """Vectorized version of `.check_pore`.

        Parameters
        ----------
        data : np.ndarray
            2-column numpy containing the coordinates (xz, y).

        Returns
        -------
        np.ndarray
            Boolean array with the same length as `data`.
        """
        xz = data[:, 0]
        y = data[:, 1]

        # Compute the coordinate within the regular unit cell.
        xzr = xz % self.c
        yr = y % self.a

        r2 = self.r * self.r

        xzr2 = xzr * xzr
        yr2 = yr * yr

        xzrc2 = (self.c - xzr) * (self.c - xzr)
        yra2 = (self.a - yr) * (self.a - yr)

        cond = ((yr2 + xzr2) < r2) + (yra2 + xzr2 < r2) + (
            yr2 + xzrc2 < r2) + (yra2 + xzrc2 < r2)

        xzrh = xzr - self.c / 2
        yrh = yr - self.a / 2
        cond = cond + (yrh * yrh + xzrh * xzrh < r2)

        return cond
