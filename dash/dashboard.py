import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from nanomesh import Plane, metrics
from nanomesh.data import binary_blobs2d

st.title('Nanomesh - 2D Meshing')

# - [x] uploaded
# - [ ] select from submodule
# - [ ] image from web
# - [ ] png

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

    # blobs = binary_blobs(length=100, volume_fraction=0.25, seed=2102)
    # plane = Image(blobs)

    fig, ax = plt.subplots()
    ax = plane.show()
    st.pyplot(ax.figure)

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

fig, ax = plt.subplots()
ax = mesher.plot_contour(legend='floating', show_region_markers=False)
st.pyplot(ax.figure)

opts = st.text_input('Triangulation options', value='q30a5')

mesh_container = triangulate(mesher, opts=opts)
triangle_mesh = mesh_container.get('triangle')

fig, ax = plt.subplots()
ax = triangle_mesh.plot()
st.pyplot(ax.figure)

metrics_list = st.multiselect('Metrics to plot',
                              default='area',
                              options=list(metrics._metric_dispatch))

st.header('Mesh metrics')


@st.cache
def get_metric_fig(mesh, metric):
    ax = metrics.histogram(mesh, metric=metric)
    return ax.figure


for metric in metrics_list:
    col1, col2 = st.columns(2)

    with col1:
        ax = metrics.histogram(triangle_mesh, metric=metric)
        # fig = get_metric_fig(triangle_mesh, metric)
        st.pyplot(ax.figure)

    with col2:
        ax = metrics.plot2d(triangle_mesh, metric=metric)
        st.pyplot(ax.figure)

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
