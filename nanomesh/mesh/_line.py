from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np

from .._doc import doc
from ._mesh import Mesh

if TYPE_CHECKING:
    from ..mesh_container import MeshContainer


@doc(Mesh,
     prefix='Data class for line meshes',
     dim_points='2 or 3',
     dim_cells='2')
class LineMesh(Mesh, cell_dim=2):
    cell_type = 'line'

    def plot_mpl(self, *args, **kwargs) -> plt.Axes:
        """Simple line mesh plot using :mod:`matplotlib`.

        Shortcut for :func:`plotting.linemeshplot`.

        Parameters
        ----------
        *args
            Arguments passed to :func:`plotting.linemeshplot`
        **kwargs
            Keyword arguments passed to :func:`plotting.linemeshplot`

        Returns
        -------
        plt.Axes
        """
        from ..plotting import linemeshplot
        return linemeshplot(self, *args, **kwargs)

    @doc(plot_mpl)
    def plot(self, *args, **kwargs):
        return self.plot_mpl(*args, **kwargs)

    def label_boundaries(self,
                         left: Optional[int | str] = None,
                         right: Optional[int | str] = None,
                         top: Optional[int | str] = None,
                         bottom: Optional[int | str] = None,
                         key: str = None):
        """Labels the boundaries of the mesh with the given value.

        Parameters
        ----------
        left : int | str, optional
            Labels left boundary segments with the given value. If a string
            is passed, the :attr:`LineMesh.fields` attribute is updated with
            the field / value pair.
        right : int | str, optional
            Same as above.
        top : int | str, optional
            Same as above.
        bottom : int | str, optional
            Same as above.
        key : str, optional
            Key of the :attr:`LineMesh.cell_data` dictionary to update.
            Defaults to :attr:`LineMesh.default_key`.
        """
        if not key:
            key = self.default_key

        for side, f_bound, col in (
            (left, np.min, 1),
            (right, np.max, 1),
            (top, np.max, 0),
            (bottom, np.min, 0),
        ):
            if not side:
                continue

            bound = f_bound(self.points)
            idx = np.argwhere(self.points[:, col] == bound)
            side_idx = np.nonzero(np.all(np.isin(self.cells, idx), axis=1))

            if isinstance(side, str):
                int_label = max(self.cell_data[key].max(), 1) + 1
                self.fields[side] = int_label
            else:
                int_label = side

            self.cell_data[key][side_idx] = int_label

    @doc(prefix='Triangulate mesh using :func:`triangulate`')
    def triangulate(self, opts: str = 'pq30Aa100', **kwargs) -> MeshContainer:
        from .._triangle_wrapper import triangulate
        return triangulate(self, opts=opts, **kwargs)
