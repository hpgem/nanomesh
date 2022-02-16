import numpy as np
from matplotlib.widgets import PolygonSelector


class PolygonSelectorWithSnapping(PolygonSelector):
    """Select a polygon region of an axes with snapping to points.

    For usage details see :class:`matplotlib.widgets.PolygonSelector`

    Parameters
    ----------
    snap_to : (n,2) numpy.ndarray
        List of points to snap to .
    *args : list
        Arguments passed to :class:`matplotlib.widgets.PolygonSelector`.
    **kwargs
        Keyword arguments passed to
        :class:`matplotlib.widgets.PolygonSelector`.
        The parent axes for the widget.
    """

    def __init__(
        self,
        *args,
        snap_to: np.ndarray = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.snapped = False
        self._snap_to_points = snap_to
        self.SNAPPING = (snap_to is not None)

    def _release(self, event):
        """Button release event handler."""
        # Release active tool handle.
        if self._active_handle_idx >= 0:
            self._active_handle_idx = -1

        # Complete the polygon.
        elif (len(self._xs) > 3 and self._xs[-1] == self._xs[0]
              and self._ys[-1] == self._ys[0]):
            self._selection_completed = True

        elif self.snapped is not None:
            self._xs.insert(-1, self._xs[-1])
            self._ys.insert(-1, self._ys[-1])

        # Place new vertex.
        elif (not self._selection_completed and 'move_all' not in self.state
              and 'move_vertex' not in self.state):
            self._xs.insert(-1, event.xdata)
            self._ys.insert(-1, event.ydata)

        if self._selection_completed:
            self.onselect(self.verts)

    def _onmove(self, event):
        """Cursor move event handler."""
        # Move the active vertex (ToolHandle).
        if self._active_handle_idx >= 0:
            idx = self._active_handle_idx
            self._xs[idx], self._ys[idx] = event.xdata, event.ydata
            # Also update the end of the polygon line if the first vertex is
            # the active handle and the polygon is completed.
            if idx == 0 and self._selection_completed:
                self._xs[-1], self._ys[-1] = event.xdata, event.ydata

        # Move all vertices.
        elif 'move_all' in self.state and self.eventpress:
            dx = event.xdata - self.eventpress.xdata
            dy = event.ydata - self.eventpress.ydata
            for k in range(len(self._xs)):
                self._xs[k] = self._xs_at_press[k] + dx
                self._ys[k] = self._ys_at_press[k] + dy

        # Do nothing if completed or waiting for a move.
        elif (self._selection_completed or 'move_vertex' in self.state
              or 'move_all' in self.state):
            return

        # Position pending vertex.
        else:
            # Calculate distance to the start vertex.
            x0, y0 = self.line.get_transform().transform(
                (self._xs[0], self._ys[0]))
            v0_dist = np.hypot(x0 - event.x, y0 - event.y)

            if self.SNAPPING:
                # Calculate nearest point to snap to.
                self._tf_snap_to = self.line.get_transform().transform(
                    self._snap_to_points)
                vn_dists = np.hypot(self._tf_snap_to[:, 0] - event.x,
                                    self._tf_snap_to[:, 1] - event.y)
                vn_ind = np.argmin(vn_dists)

            # Lock on to the start vertex if near it and ready to complete.
            if len(self._xs) > 3 and v0_dist < self.vertex_select_radius:
                self._xs[-1], self._ys[-1] = self._xs[0], self._ys[0]

            # Lock on to pre-defined point if near it
            elif self.SNAPPING and (vn_dists[vn_ind] <
                                    self.vertex_select_radius):
                self._xs[-1], self._ys[-1] = self._snap_to_points[vn_ind]
                self._snapped = True

            else:
                self._xs[-1], self._ys[-1] = event.xdata, event.ydata
                self._snapped = False

        self._draw_polygon()
