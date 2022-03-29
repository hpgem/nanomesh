import numpy as np
import streamlit as st
from bokeh.models import LinearColorMapper
from bokeh.palettes import Category10_10 as catpalette
from bokeh.palettes import Viridis256 as palette
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

from nanomesh import Plane, metrics
from nanomesh.data import binary_blobs2d

st.title('Nanomesh - 2D Meshing')

# TODO
# - More image input methods
# - Apply image processing on plane
# - Find more dynamic way of plotting histograms/2d data (plotly, altair?)

with st.sidebar:
    st.header('Load data')

    data_choice = st.radio('Data',
                           index=0,
                           options=('Synthetic data', 'Custom data'))

    data_choice

    if data_choice == 'Synthetic data':
        seed = st.number_input('Seed', value=1234)
        length = st.number_input('Length', value=50)
        blob_size_fraction = st.slider('Blob size fraction',
                                       value=0.3,
                                       step=0.05)
        volume_fraction = st.slider('Volume fraction', value=0.2, step=0.05)

        data = binary_blobs2d(seed=seed,
                              length=length,
                              blob_size_fraction=blob_size_fraction,
                              volume_fraction=volume_fraction)

    else:
        uploaded_file = st.file_uploader('Upload data file in numpy format')

        if not uploaded_file:
            st.stop()

        @st.cache
        def load_data(data_file):
            return np.load(data_file)

        data = load_data(uploaded_file)

    plane = Plane(data)

    d = plane.image

    p = figure(tooltips=[('x', '$x'), ('y', '$y'), ('value', '@image')],
               match_aspect=True,
               border_fill_alpha=0.0)
    p.x_range.range_padding = p.y_range.range_padding = 0

    dw, dh = d.shape
    # must give a vector of image data for image parameter
    p.image(image=[d],
            x=0,
            y=0,
            dw=dw,
            dh=dh,
            palette='Viridis11',
            level='image')
    p.grid.grid_line_width = 0.5

    st.bokeh_chart(p, use_container_width=True)

st.header('Generate mesh')


@st.cache
def generate_contour(precision=None, max_edge_dist=None):
    from nanomesh import Mesher
    mesher = Mesher(plane)
    mesher.generate_contour(precision=precision, max_edge_dist=max_edge_dist)
    return mesher


@st.cache
def triangulate(mesher, opts=None):
    mesh_container = mesher.triangulate(opts=opts)
    return mesh_container


col1, col2 = st.columns(2)
with col1:
    precision = st.number_input('precision',
                                min_value=0.0,
                                step=0.1,
                                value=1.0)
with col2:
    max_edge_dist = st.number_input('Max edge distance',
                                    min_value=0.0,
                                    step=0.5,
                                    value=5.0)

mesher = generate_contour(precision=precision, max_edge_dist=max_edge_dist)


def contour_mesh(mesher):
    p = figure(title='Line Mesh',
               tooltips=[('x', '$x'), ('y', '$y'), ('value', '@image')],
               match_aspect=True,
               border_fill_alpha=0.0)

    c = mesher.contour

    vert_x, vert_y = c.points.T

    import itertools
    colors = itertools.cycle(catpalette)

    for i in np.unique(c.cell_data['segment_markers']):
        color = next(colors)
        mask = c.cell_data['segment_markers'] != i

        if mask is not None:
            cells = c.cells[~mask.squeeze()]

        lines_x = (vert_x[cells] + 0.5).tolist()
        lines_y = (vert_y[cells] + 0.5).tolist()

        p.multi_line(lines_y,
                     lines_x,
                     legend_label=f'{i}',
                     line_width=5,
                     color=color)

    p.hover.point_policy = 'follow_mouse'

    # image
    d = mesher.image

    p.x_range.range_padding = p.y_range.range_padding = 0

    dw, dh = d.shape
    # must give a vector of image data for image parameter
    p.image(image=[d],
            x=0,
            y=0,
            dw=dw,
            dh=dh,
            palette='Viridis11',
            level='image')
    p.grid.grid_line_width = 0.0
    return p


p = contour_mesh(mesher)
st.bokeh_chart(p, use_container_width=True)

opts = st.text_input('Triangulation options', value='q30a5')

mesh_container = triangulate(mesher, opts=opts)
triangle_mesh = mesh_container.get('triangle')


def get_meshplot(mesh):
    p = figure(title='Meshplot',
               tooltips=[('Physical', '@physical'), ('(x, y)', '($x, $y)'),
                         ('index', '$index')],
               x_axis_label='x',
               y_axis_label='y',
               match_aspect=True)

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

    p.patches('x',
              'y',
              source=data,
              fill_color=None,
              line_color=factor_cmap('physical',
                                     palette=catpalette,
                                     factors=factors),
              line_width=2)
    p.hover.point_policy = 'follow_mouse'

    return p


p = get_meshplot(triangle_mesh)
st.bokeh_chart(p)

st.header('Mesh metrics')

metrics_list = st.multiselect('Metrics to plot',
                              default='area',
                              options=list(metrics._metric_dispatch))


def get_metric_hist(mesh, metric):
    metric_vals = getattr(metrics, metric)(mesh)

    hist, edges = np.histogram(metric_vals, bins=50)

    p = figure(
        title=f'Histogram of triangle {metric}',
        x_axis_label=f'Triangle {metric}',
        y_axis_label='Frequency',
        tooltips=[
            ('value', '@top'),
        ],
    )
    p.quad(top=hist,
           bottom=0,
           left=edges[:-1],
           right=edges[1:],
           fill_color='navy',
           line_color='white',
           alpha=0.5)
    p.hover.point_policy = 'follow_mouse'

    return p


def get_metric_2dplot(mesh, metric):
    metric_vals = getattr(metrics, metric)(mesh)

    color_mapper = LinearColorMapper(palette=palette)

    p = figure(title=f'Triplot of triangle {metric}',
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

    p.patches('x',
              'y',
              source=data,
              fill_color={
                  'field': metric,
                  'transform': color_mapper
              },
              line_width=1)
    p.hover.point_policy = 'follow_mouse'

    return p


for metric in metrics_list:
    p = get_metric_hist(triangle_mesh, metric)
    st.bokeh_chart(p, use_container_width=True)

    p = get_metric_2dplot(triangle_mesh, metric)
    st.bokeh_chart(p, use_container_width=True)

# Embedding pyvista / vtk
# https://discuss.streamlit.io/t/is-it-possible-plot-3d-mesh-file-or-to-add-the-vtk-pyvista-window-inline-streamlit/4164/7

st.sidebar.header('Export mesh')

import atexit
import tempfile
from pathlib import Path

import meshio


@st.cache
def convert_mesh(mesh, filename, file_format):
    tempdir = tempfile.TemporaryDirectory()
    temp_path = Path(tempdir.name) / filename
    mesh.write(temp_path, file_format=file_format)
    atexit.register(tempdir.cleanup)
    return temp_path


filetypes = {}
for ext, fmts in meshio._helpers.extension_to_filetypes.items():
    for fmt in fmts:
        filetypes[f'{fmt} ({ext})'] = (fmt, ext)

file_stem = 'mesh'
file_format = st.sidebar.selectbox('Export to',
                                   index=0,
                                   options=[None] + list(filetypes))

if not file_format:
    st.stop()

fmt, ext = filetypes[file_format]
filename = file_stem + ext

if file_format:
    temp_path = convert_mesh(triangle_mesh, filename, fmt)

    with open(temp_path, 'rb') as file:
        btn = st.sidebar.download_button(
            label='Download',
            data=file,
            file_name=filename,
        )
