import itertools

import numpy as np
from bokeh.models import LinearColorMapper
from bokeh.palettes import Category10_10 as catpalette
from bokeh.palettes import Viridis256 as palette
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

from nanomesh import Image, metrics


def image_plot(data):
    if isinstance(data, Image):
        data = data.image

    fig = figure(tooltips=[('x', '$x'), ('y', '$y'), ('value', '@image')],
                 match_aspect=True,
                 border_fill_alpha=0.0,
                 height=300)

    fig.x_range.range_padding = fig.y_range.range_padding = 0

    dw, dh = data.shape

    # must give a vector of image data for image parameter
    fig.image(image=[data],
              x=0,
              y=0,
              dw=dw,
              dh=dh,
              palette='Viridis11',
              level='image')
    fig.grid.grid_line_width = 0.5
    return fig


def contour_mesh(mesher):
    fig = figure(title='Line Mesh',
                 tooltips=[('x', '$x'), ('y', '$y'), ('value', '@image')],
                 match_aspect=True,
                 border_fill_alpha=0.0)

    c = mesher.contour

    vert_x, vert_y = c.points.T

    colors = itertools.cycle(catpalette)

    for i in np.unique(c.cell_data['segment_markers']):
        color = next(colors)
        mask = c.cell_data['segment_markers'] != i

        if mask is not None:
            cells = c.cells[~mask.squeeze()]

        lines_x = (vert_x[cells] + 0.5).tolist()
        lines_y = (vert_y[cells] + 0.5).tolist()

        fig.multi_line(lines_y,
                       lines_x,
                       legend_label=f'{i}',
                       line_width=5,
                       color=color)

    fig.hover.point_policy = 'follow_mouse'

    # image
    d = mesher.image

    fig.x_range.range_padding = fig.y_range.range_padding = 0

    dw, dh = d.shape
    # must give a vector of image data for image parameter
    fig.image(image=[d],
              x=0,
              y=0,
              dw=dw,
              dh=dh,
              palette='Viridis11',
              level='image')
    fig.grid.grid_line_width = 0.0
    return fig


def get_meshplot(mesh):
    fig = figure(title='Meshplot',
                 tooltips=[('Physical', '@physical'), ('(x, y)', '($x, $y)'),
                           ('index', '$index')],
                 x_axis_label='x',
                 y_axis_label='y',
                 match_aspect=True)

    color_mapper = LinearColorMapper(palette=palette)

    xs = mesh.points[mesh.cells][:, :, 0]
    ys = mesh.points[mesh.cells][:, :, 1]
    cs = mesh.cell_data['physical']

    data = {
        'x': xs.tolist(),
        'y': ys.tolist(),
        'physical':
        tuple(mesh.number_to_field.get(val, str(val)) for val in cs)
    }

    factors = tuple(reversed(mesh.fields))

    fig.patches(
        'x',
        'y',
        source=data,
        # line_color='white',
        fill_color=factor_cmap('physical', palette=catpalette,
                               factors=factors),
        fill_alpha=0.7,
        line_width=1)
    fig.hover.point_policy = 'follow_mouse'

    return fig


def get_metric_hist(mesh, metric):
    metric_vals = getattr(metrics, metric)(mesh)

    hist, edges = np.histogram(metric_vals, bins=50)

    fig = figure(
        title=f'Histogram of triangle {metric}',
        x_axis_label=f'Triangle {metric}',
        y_axis_label='Frequency',
        tooltips=[
            ('value', '@top'),
        ],
    )
    fig.quad(top=hist,
             bottom=0,
             left=edges[:-1],
             right=edges[1:],
             fill_color='navy',
             line_color='white',
             alpha=0.5)
    fig.hover.point_policy = 'follow_mouse'

    return fig


def get_metric_2dplot(mesh, metric):
    metric_vals = getattr(metrics, metric)(mesh)

    color_mapper = LinearColorMapper(palette=palette)

    fig = figure(title=f'Triplot of triangle {metric}',
                 tooltips=[('Physical', '@physical'),
                           (metric.capitalize(), f'@{metric}'),
                           ('(x, y)', '($x, $y)')],
                 x_axis_label='x',
                 y_axis_label='y',
                 match_aspect=True)

    xs = mesh.points[mesh.cells][:, :, 0]
    ys = mesh.points[mesh.cells][:, :, 1]

    cs = mesh.cell_data['physical']

    data = {
        'x': xs.tolist(),
        'y': ys.tolist(),
        'label': cs.tolist(),
        metric: metric_vals.tolist(),
    }

    fig.patches('x',
                'y',
                source=data,
                fill_color={
                    'field': metric,
                    'transform': color_mapper
                },
                line_width=1)
    fig.hover.point_policy = 'follow_mouse'

    return fig
