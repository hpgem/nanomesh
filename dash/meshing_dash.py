import atexit
import tempfile
from pathlib import Path

import meshio
import numpy as np
import streamlit as st
from bokeh_plots import (contour_mesh, get_meshplot, get_metric_2dplot,
                         get_metric_hist, image_plot)
from skimage import io
from skimage.color import rgb2gray

from nanomesh import Mesher, Plane, metrics
from nanomesh.data import binary_blobs2d, nanopores
from nanomesh.image._image import _threshold_dispatch

st.set_page_config(page_title='Nanomesh dashboard',
                   page_icon='🔺',
                   initial_sidebar_state='expanded')

st.title('Nanomesh - 2D Meshing')
st.write('Upload your own data or use the example data to generate a mesh!')


@st.experimental_memo
def load_data(data_file):
    return np.load(data_file)


@st.experimental_memo
def load_binary_data(**kwargs):
    return binary_blobs2d(**kwargs)


@st.cache
def load_data_from_url(url):
    return rgb2gray(io.imread(url))


@st.cache
def data_are_binary(plane):
    return set(np.unique(plane.image)) == {0, 1}


@st.cache
def generate_contour(precision=None, max_edge_dist=None):
    mesher = Mesher(plane)
    mesher.generate_contour(precision=precision, max_edge_dist=max_edge_dist)
    return mesher


@st.cache
def triangulate(mesher, opts=None):
    mesh_container = mesher.triangulate(opts=opts)
    return mesh_container


with st.sidebar:
    st.header('Input data')

    CHOICE_1 = 'Use synthetic data (blobs)'
    CHOICE_2 = 'Use example data (nanopores)'
    CHOICE_3 = 'Use your own image'
    CHOICE_4 = 'Use URL'

    data_choice = st.radio('Choose data source',
                           index=0,
                           options=(CHOICE_1, CHOICE_2, CHOICE_3))

    if data_choice == CHOICE_1:
        seed = st.number_input('Seed', value=1234)
        length = st.number_input('Length', value=50, step=10)
        blob_size_fraction = st.slider('Blob size fraction',
                                       value=0.3,
                                       step=0.05)
        volume_fraction = st.slider('Volume fraction', value=0.2, step=0.05)

        data = load_binary_data(seed=seed,
                                length=length,
                                blob_size_fraction=blob_size_fraction,
                                volume_fraction=volume_fraction)

    if data_choice == CHOICE_2:
        data = nanopores()

    elif data_choice == CHOICE_3:
        uploaded_file = st.file_uploader('Upload data file in numpy format')

        if not uploaded_file:
            st.stop()

        data = load_data(uploaded_file)

    elif data_choice == CHOICE_4:
        url = st.text_input('Link to image (url)')

        if not url:
            st.stop()

        data = load_data_from_url(url)

    plane = Plane(data)

    fig = image_plot(plane)
    st.bokeh_chart(fig, use_container_width=True)


@st.cache
def do_gaussian(plane, **kwargs):
    return plane.gaussian(**kwargs)


@st.cache
def do_digitize(plane, **kwargs):
    return plane.digitize(**kwargs)


if not data_are_binary(plane):
    st.header('Image processing')

    with st.container():
        c1, c2 = st.columns((0.3, 0.7))

        with c1:
            gaussian_blur = st.checkbox('Apply gaussian blur', value=False)
            sigma = st.number_input('Sigma', value=5)

        if gaussian_blur:
            plane = do_gaussian(plane, sigma=sigma)

        with c1:
            threshold_method = st.selectbox('Select thresholding method',
                                            index=1,
                                            options=list(_threshold_dispatch))
            threshold_value = plane.threshold(threshold_method)
            st.metric('Threshold value',
                      f'{threshold_value:.3f}',
                      delta=None,
                      delta_color='normal')

        with c1:
            segment = st.checkbox('Segment', value=False)
            invert = st.checkbox('Invert contrast', value=False)

        if segment:
            plane = do_digitize(plane, bins=[threshold_value])

        if invert:
            plane = plane.invert_contrast()

        with c2:
            fig = image_plot(plane)
            st.bokeh_chart(fig, use_container_width=True)

if not data_are_binary(plane):
    st.error('Data are not binary. Segment first.')
    st.stop()

st.header('Contour finding')

c1, c2 = st.columns((0.3, 0.7))
with c1:
    precision = st.number_input('precision',
                                min_value=0.0,
                                step=0.1,
                                value=1.0)

    max_edge_dist = st.number_input('Max edge distance',
                                    min_value=0.0,
                                    step=0.5,
                                    value=5.0)

mesher = generate_contour(precision=precision, max_edge_dist=max_edge_dist)

with c2:
    fig = contour_mesh(mesher)
    st.bokeh_chart(fig, use_container_width=True)

st.header('Triangulation')

opts = st.text_input(
    'Triangulation options',
    value='q30a5',
    help=('For more information, check out the [triangle documentation]'
          '(https://rufat.be/triangle/API.html#triangle.triangulate).'))

with st.spinner('Triangulating...'):
    mesh_container = triangulate(mesher, opts=opts)
    triangle_mesh = mesh_container.get('triangle')

fig = get_meshplot(triangle_mesh)
st.bokeh_chart(fig)

st.header('Mesh metrics')

metrics_list = st.multiselect('Select which metrics to plot',
                              default='area',
                              options=list(metrics._metric_dispatch))

for metric in metrics_list:
    fig = get_metric_hist(triangle_mesh, metric)
    st.bokeh_chart(fig, use_container_width=True)

    fig = get_metric_2dplot(triangle_mesh, metric)
    st.bokeh_chart(fig, use_container_width=True)

st.sidebar.header('Export mesh')


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
file_format = st.sidebar.selectbox('Choose format to export data to',
                                   index=0,
                                   options=[None, *filetypes])

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
