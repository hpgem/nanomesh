import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from skimage.data import binary_blobs

from nanomesh import Plane, metrics

# - [x] uploaded
# - [ ] select from submodule
# - [ ] image from web
# - [ ] png

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
plane.show(ax=ax)
st.pyplot(fig)


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


precision = st.slider('precision',
                      min_value=0.0,
                      max_value=10.0,
                      step=0.2,
                      value=1.0)
max_edge_dist = st.slider('max_edge_dist',
                          min_value=0.0,
                          max_value=25.0,
                          step=0.5,
                          value=5.0)

mesher = generate_contour(precision=precision, max_edge_dist=max_edge_dist)

fig, ax = plt.subplots()
mesher.plot_contour(legend='floating', show_region_markers=False, ax=ax)
st.pyplot(fig)

opts = st.text_input('Triangulation options', value='q30a5')

mesh_container = triangulate(mesher, opts=opts)
triangle_mesh = mesh_container.get('triangle')

fig, ax = plt.subplots()
triangle_mesh.plot(ax=ax)
st.pyplot(fig)

metrics_list = st.multiselect('Metrics to plot',
                              default='area',
                              options=list(metrics._metric_dispatch))


@st.cache
def get_metric_fig(mesh, metric):
    fig, ax = plt.subplots()
    metrics.histogram(mesh, metric=metric, ax=ax)
    return fig


for metric in metrics_list:
    fig, ax = plt.subplots()
    metrics.histogram(triangle_mesh, metric=metric, ax=ax)
    # fig = get_metric_fig(triangle_mesh, metric)
    st.pyplot(fig)

# Embedding pyvista / vtk
# https://discuss.streamlit.io/t/is-it-possible-plot-3d-mesh-file-or-to-add-the-vtk-pyvista-window-inline-streamlit/4164/7

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
file_format = st.selectbox('Export to',
                           index=0,
                           options=[None] + list(filetypes))

if not file_format:
    st.stop()

fmt, ext = filetypes[file_format]
filename = file_stem + ext

if file_format:
    temp_path = convert_mesh(triangle_mesh, filename, fmt)

    with open(temp_path, 'rb') as file:
        btn = st.download_button(
            label='Download',
            data=file,
            file_name=filename,
        )
