from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .mesh_container import MeshContainer


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

    def __call__(self, mesh: MeshContainer) -> np.ndarray:
        grid = mesh.to_pyvista_unstructured_grid()
        ret = grid.compute_cell_quality(self.metric)
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
distortion = Metric('distortion')
max_angle = Metric('max_angle')
min_angle = Metric('min_angle')
radius_ratio = Metric('radius_ratio')
relative_size_squared = Metric('relative_size_squared')
scaled_jacobian = Metric('scaled_jacobian')
shape = Metric('shape')
shape_and_size = Metric('shape_and_size')


def max_min_edge_ratio(mesh: MeshContainer) -> np.ndarray:
    """Place holder, updated dynamically."""
    cell_points = mesh.points[mesh.cells]
    diff = cell_points - np.roll(cell_points, shift=1, axis=1)
    lengths = np.linalg.norm(diff, axis=2)
    return lengths.max(axis=1) / lengths.min(axis=1)


@dataclass
class MetricDescriptor:
    name: str
    description: str
    units: str
    func: Callable
    optimal: Optional[Tuple[float, float]] = None
    range: Optional[Tuple[float, float]] = None

    @property
    def label(self):
        """Return label with units."""
        string = self.name
        if units := self.units:
            string += f' ({units})'
        return string


# coreform.com/cubit_help/mesh_generation/mesh_quality_assessment/triangular_metrics.htm
# vtk.org/doc/nightly/html/classvtkMeshQuality.html#aefa3db78933a64e68c2718cf83eac3c5
# www.feflow.info/html/help73/feflow/09_Parameters/Auxiliary_Data/condition_number.html
_metric_dispatch = {
    'area':
    MetricDescriptor(
        name='Triangle area',
        description='Calculate the area of a triangle.',
        units='px^2',
        optimal=None,
        range=(0, np.inf),
        func=area,
    ),
    'aspect_frobenius':
    MetricDescriptor(
        name='Frobenius aspect',
        description=(
            'Calculate the Frobenius condition number of the '
            'transformation matrix from an equilateral triangle to a triangle.'
        ),
        units='',
        optimal=None,
        range=None,
        func=aspect_frobenius,
    ),
    'aspect_ratio':
    MetricDescriptor(
        name='Aspect ratio',
        description='Calculate the aspect ratio of a triangle.',
        units='',
        optimal=None,
        range=None,
        func=aspect_ratio,
    ),
    'condition':
    MetricDescriptor(
        name='Condition number',
        description='Calculate the condition number of a triangle.',
        units='',
        optimal=(1, 1.3),
        range=(1, np.inf),
        func=condition,
    ),
    'max_angle':
    MetricDescriptor(
        name='Maximum angle',
        description=(
            'Calculate the maximal (nonoriented) angle of a triangle.'),
        units='degrees',
        optimal=(60, 90),
        range=(60, 180),
        func=max_angle,
    ),
    'min_angle':
    MetricDescriptor(
        name='Minimum angle',
        description=(
            'Calculate the minimal (nonoriented) angle of a triangle.'),
        units='degrees',
        optimal=(30, 60),
        range=(0, 60),
        func=min_angle,
    ),
    'radius_ratio':
    MetricDescriptor(
        name='Radius ratio',
        description=(
            'Calculate the radius ratio of a triangle. The radius ratio of a '
            'triangle $t$ is: $\frac{R}{2r}$, where $R$ and $r$ respectively '
            'denote the circumradius and the inradius of $t$.'),
        units='',
        optimal=(1.0, 2.0),
        range=(1, np.inf),
        func=radius_ratio,
    ),
    'scaled_jacobian':
    MetricDescriptor(
        name='Scaled Jacobian',
        description='Calculate the scaled Jacobian of a triangle.',
        units='',
        optimal=(0.2, 1.0),
        range=(-1, 1),
        func=scaled_jacobian,
    ),
    'shape':
    MetricDescriptor(
        name='Shape',
        description='Calculate the shape of a triangle.',
        units='',
        optimal=(0.25, 1.0),
        range=(0, 1),
        func=shape,
    ),
    'relative_size_squared':
    MetricDescriptor(
        name='Relative size',
        description='Calculate the square of the relative size of a triangle.',
        units='',
        optimal=(0.25, 1.0),
        range=(0, 1),
        func=relative_size_squared,
    ),
    'shape_and_size':
    MetricDescriptor(
        name='Shape and size',
        description=(
            'Calculate the product of shape and relative size of a triangle.'),
        units='',
        optimal=(0.25, 1.0),
        range=(0, 1),
        func=shape_and_size,
    ),
    'distortion':
    MetricDescriptor(
        name='Distortion',
        description='Calculate the distortion of a triangle.',
        units='px^2',
        optimal=(0.6, 1.0),
        range=(0, 1),
        func=distortion,
    ),
    'max_min_edge_ratio':
    MetricDescriptor(
        name='Ratio max/min edge',
        description=('Calculate the ratio between the longest '
                     'and shortest edge lengths of a triangle.'),
        units='',
        optimal=(1.0, 2.0),
        range=(1, np.inf),
        func=max_min_edge_ratio,
    ),
}

# patch docstrings
for descriptor in _metric_dispatch.values():
    descriptor.func.__doc__ = f"""{descriptor.description}

Parameters
----------
mesh : MeshContainer
    Input mesh

Returns
-------
quality : np.ndarray
    Array with cell qualities.
    """


def calculate_all_metrics(mesh: MeshContainer, inplace: bool = False) -> dict:
    """Calculate all available metrics.

    Parameters
    ----------
    mesh : MeshContainer
        Input mesh
    inplace : bool, optional
        Updates the `metadata` attribute on the mesh with the metrics.

    Returns
    -------
    metrics : dict
        Return a dict with all the metrics
    """
    metrics = {}
    for metric, descriptor in _metric_dispatch.items():
        quality = descriptor.func(mesh)  # type: ignore
        metrics[metric] = quality

        if inplace:
            mesh.metadata[metric] = quality

    return metrics


def histogram(
    mesh: MeshContainer,
    *,
    metric: str,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a mesh plot with the cells are colored by the cell quality.

    Parameters
    ----------
    mesh : MeshContainer
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
    kwargs.setdefault('rwidth', 0.8)

    descriptor = _metric_dispatch[metric]
    quality = descriptor.func(mesh)  # type: ignore

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(quality, **kwargs)
    ax.set_title(f'Histogram of {descriptor.name.lower()}')

    if optimal_range := descriptor.optimal:
        ax.axvspan(*optimal_range, color='limegreen', zorder=0, alpha=0.1)

    xlabel = descriptor.label
    ax.set_xlabel(xlabel)

    ylabel = 'probability' if kwargs.get('density') else 'frequency'
    ax.set_ylabel(ylabel)

    return ax


def plot2d(
    mesh: MeshContainer,
    *,
    metric: str,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a mesh plot with the cells are colored by the cell quality.

    Parameters
    ----------
    mesh : MeshContainer
        Input mesh
    metric : str
        Metric to calculate.
    ax : `matplotlib.Axes`
        If specified, `ax` will be used to create the subplot.
    vmin, vmax : int, float
        Set the lower/upper boundary for the color value.
        Defaults to the 1st and 99th percentile, respectively.
    cmap : str
        Set the color map.
    **kwargs
        Keyword arguments passed on to `ax.tripcolor`.

    Returns
    -------
    ax : `matplotlib.Axes`
    """
    descriptor = _metric_dispatch[metric]
    quality = descriptor.func(mesh)  # type: ignore

    kwargs.setdefault('vmin', np.percentile(quality, 1))
    kwargs.setdefault('vmax', np.percentile(quality, 99))

    fig, ax = plt.subplots()

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]

    cells = mesh.cells

    tpc = ax.tripcolor(x, y, quality, triangles=cells, **kwargs)
    ax.figure.colorbar(tpc)
    ax.axis('scaled')

    ax.set_title(f'Triplot of {descriptor.name.lower()}')

    return ax
