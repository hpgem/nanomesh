from __future__ import annotations

import numpy as np


class PruneZ0Mixin:

    def prune_z_0(self):
        """Drop third dimension (z) coordinates if present and all values are
        equal to 0 (within tolerance).

        For compatibility, sometimes a column with zeroes is added, for
        example when exporting to gmsh2.2 format. This method drops that
        column.
        """
        TOL = 1e-9

        is_3_dimensional = self.points.shape[1] == 3
        if not is_3_dimensional:
            return

        z_close_to_0 = np.all(np.abs(self.points[:, 2]) < TOL)
        if z_close_to_0:
            self.points = self.points[:, 0:2]
