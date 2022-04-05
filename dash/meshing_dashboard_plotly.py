import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
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

    fig = px.imshow(d, color_continuous_scale='Viridis')
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(margin=dict(l=5, r=5, t=0, b=0), )
    st.plotly_chart(fig, use_container_width=True)

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


@st.cache
def contour_mesh(mesher):
    # image
    d = mesher.image
    fig = px.imshow(d, color_continuous_scale='Viridis')
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(margin=dict(l=5, r=5, t=0, b=0), )

    c = mesher.contour

    vert_x, vert_y = c.points.T

    import itertools
    colors = itertools.cycle(catpalette)

    for i in np.unique(c.cell_data['segment_markers']):
        color = next(colors)
        mask = c.cell_data['segment_markers'] != i

        if mask is not None:
            cells = c.cells[~mask.squeeze()]

        lines_x = vert_x[cells]
        lines_y = vert_y[cells]

        lines_x = np.insert(lines_x, 2, np.nan, axis=1).ravel()
        lines_y = np.insert(lines_y, 2, np.nan, axis=1).ravel()

        hoverinfo = ''

        fig.add_trace(
            go.Scatter(
                x=lines_y,
                y=lines_x,
                name=f'{i}',
                mode='lines',
                fill='toself',
                showlegend=False,
                # hovertemplate=hoverinfo,
                hoveron='points+fills'))

    return fig


fig = contour_mesh(mesher)
st.plotly_chart(fig, use_container_width=True)

opts = st.text_input('Triangulation options', value='q30a5')

mesh_container = triangulate(mesher, opts=opts)
triangle_mesh = mesh_container.get('triangle')


@st.cache
def get_meshplot(mesh):
    fig = go.Figure()

    vert_x, vert_y = mesh.points.T

    for i in np.unique(mesh.cell_data['physical']):
        mask = mesh.cell_data['physical'] != i

        if mask is not None:
            cells = mesh.cells[~mask.squeeze()]

        lines_x = vert_x[cells]
        lines_y = vert_y[cells]

        lines_x = np.insert(lines_x, 3, lines_x[:, 0], axis=1)
        lines_y = np.insert(lines_y, 3, lines_y[:, 0], axis=1)

        lines_x = np.insert(lines_x, 4, np.nan, axis=1).ravel()
        lines_y = np.insert(lines_y, 4, np.nan, axis=1).ravel()

        fig.add_trace(go.Scatter(x=lines_x, y=lines_y, fill='toself'))

    fig.update_yaxes(scaleanchor='x', scaleratio=1)

    return fig


fig = get_meshplot(triangle_mesh)
st.plotly_chart(fig)

st.header('Mesh metrics')

metrics_list = st.multiselect('Metrics to plot',
                              default='area',
                              options=list(metrics._metric_dispatch))


@st.cache
def get_metric_2dplot(mesh, metric):
    metric_vals = getattr(metrics, metric)(mesh)

    fig = go.Figure()

    vert_x, vert_y = mesh.points.T
    vert_z = np.zeros_like(vert_x)

    cell_i, cell_j, cell_k = mesh.cells.T

    color = metric_vals
    color = mesh.cell_data['physical']

    fig = go.Figure(
        go.Mesh3d(
            x=vert_x,
            y=vert_y,
            z=vert_z,
            i=cell_i,
            j=cell_j,
            k=cell_k,
            intensity=metric_vals,
            intensitymode='cell',
        ))

    return fig


@st.cache
def get_metric_hist(mesh, metric):
    metric_vals = getattr(metrics, metric)(mesh)

    fig = px.histogram(metric_vals, nbins=50)
    fig.update_layout(bargap=0.2)
    return fig


for metric in metrics_list:
    fig = get_metric_hist(triangle_mesh, metric)
    st.plotly_chart(fig, use_container_width=True)

    fig = get_metric_2dplot(triangle_mesh, metric)
    st.plotly_chart(fig, use_container_width=True)

    metric_vals = getattr(metrics, metric)(triangle_mesh)
    hist_values, hist_edges = np.histogram(metric_vals, bins=50)
    st.bar_chart(hist_values)

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
