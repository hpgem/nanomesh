import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista as pv


class Metric:
    """Factory function for metrics using `pyvista`.

    Parameters
    ----------
    metric : str
        Metric to calculate. For a full list,
        see `pvmesh.compute_cell_quality`.
    """
    def __init__(self, metric: str):
        super().__init__()
        self.metric = metric
        self.__doc__ = ('Compute the {metric} cell quality metric for the '
                        'given data.')

    def __call__(self, mesh: meshio.Mesh) -> np.ndarray:
        pvmesh = pv.from_meshio(mesh)
        ret = pvmesh.compute_cell_quality(self.metric)
        quality = ret.cell_arrays['CellQuality']
        return np.array(quality)


# Functions are available, but give undefined values
# aspect_beta = Metric('aspect_beta')
# aspect_gamma = Metric('aspect_gamma')
# collapse_ratio = Metric('collapse_ratio')
# diagonal = Metric('diagonal')
# dimension = Metric('dimension')
# distortion = Metric('distortion')
# jacobian = Metric('jacobian')
# max_aspect_frobenius = Metric('max_aspect_frobenius')
# max_edge_ratio = Metric('max_edge_ratio')
# med_aspect_frobenius = Metric('med_aspect_frobenius')
# oddy = Metric('oddy')
# relative_size_squared = Metric('relative_size_squared')
# shape_and_size = Metric('shape_and_size')
# shear = Metric('shear')
# shear_and_size = Metric('shear_and_size')
# skew = Metric('skew')
# stretch = Metric('stretch')
# taper = Metric('taper')
# volume = Metric('volume')
# warpage = Metric('warpage')
area = Metric('area')
aspect_frobenius = Metric('aspect_frobenius')
aspect_ratio = Metric('aspect_ratio')
condition = Metric('condition')
max_angle = Metric('max_angle')
min_angle = Metric('min_angle')
radius_ratio = Metric('radius_ratio')
scaled_jacobian = Metric('scaled_jacobian')
shape = Metric('shape')


def max_min_edge_ratio(mesh: meshio.Mesh) -> np.ndarray:
    """Compute the min/max cell edge ratio for the given data."""
    cell_points = mesh.points[mesh.cells[0].data]
    diff = cell_points - np.roll(cell_points, shift=1, axis=1)
    lengths = np.linalg.norm(diff, axis=2)
    return lengths.max(axis=1) / lengths.min(axis=1)


_func_dispatch = {
    'area': area,
    'aspect_frobenius': aspect_frobenius,
    'aspect_ratio': aspect_ratio,
    'condition': condition,
    'max_angle': max_angle,
    'min_angle': min_angle,
    'radius_ratio': radius_ratio,
    'scaled_jacobian': scaled_jacobian,
    'shape': shape,
    'max_min_edge_ratio': max_min_edge_ratio,
}


def calculate_all_metrics(mesh: meshio.Mesh, inplace: bool = False) -> dict:
    """Calculate all available metrics.

    Parameters
    ----------
    mesh : meshio.Mesh
        Input mesh
    inplace : bool, optional
        Updates the `cell_data` attribute on the mesh with the metrics.

    Returns
    -------
    dict
        Return a dict
    """
    dct = {}
    for metric, func in _func_dispatch.items():
        quality = func(mesh)  # type: ignore
        dct[metric] = quality

        if inplace:
            mesh.cell_data[metric] = [quality]

    return dct


def histogram(
    mesh: meshio.Mesh,
    *,
    metric: str,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a mesh plot with the faces are colored by the face quality.

    Parameters
    ----------
    mesh : meshio.Mesh
        Input mesh
    metric : str
        Metric to calculate.
    ax : `matplotlib.Axes`
        If specified, `ax` will be used to create the subplot.
    vmin, vmax : int, float
        Set the lower/upper boundary for the color value.
    cmap : str
        Set the color map.
    **kwargs
        Keyword arguments passed on to `ax.hist`.

    Returns
    -------
    ax : `matplotlib.Axes`
    """
    kwargs.setdefault('bins', 50)
    kwargs.setdefault('density', True)
    kwargs.setdefault('rwidth', 0.8)

    func = _func_dispatch[metric]
    quality = func(mesh)  # type: ignore

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(quality, **kwargs)
    ax.set_title(metric)
    return ax


def plot2d(
    mesh: meshio.Mesh,
    *,
    metric: str,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a mesh plot with the faces are colored by the face quality.

    Parameters
    ----------
    mesh : meshio.Mesh
        Input mesh
    metric : str
        Metric to calculate.
    ax : `matplotlib.Axes`
        If specified, `ax` will be used to create the subplot.
    vmin, vmax : int, float
        Set the lower/upper boundary for the color value.
    cmap : str
        Set the color map.
    **kwargs
        Keyword arguments passed on to `ax.tripcolor`.

    Returns
    -------
    ax : `matplotlib.Axes`
    """
    func = _func_dispatch[metric]
    quality = func(mesh)  # type: ignore

    fig, ax = plt.subplots()

    x, y, _ = mesh.points.T
    faces = mesh.cells[0].data

    ax.set_title(metric)

    tpc = ax.tripcolor(x, y, quality, triangles=faces, **kwargs)
    ax.figure.colorbar(tpc)
    ax.axis('equal')
    return ax
