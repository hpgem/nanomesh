import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from bokeh_plots import get_meshplot, get_metric_2dplot, get_metric_hist

from nanomesh import MeshContainer, data, metrics

st.title('Nanomesh - mæshboard')
st.write('Upload your own mesh or use the example data to generate metrics!')


def load_mesh(data_file):
    suffix = Path(data_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(dir='.', suffix=suffix, delete=False)
    tmp.write(data_file.getbuffer())
    tmp.close()
    mesh = MeshContainer.read(tmp.name).get('triangle')
    os.unlink(tmp.name)

    return mesh


def load_example_mesh(opts):
    return data.blob_mesh2d(opts=opts).get('triangle')


meshes = []

opts_choices = (
    'q30a10',
    'a10',
    'q30',
    'q30a5',
)

with st.sidebar:

    if st.checkbox(f'Use example data'):
        meshes = [load_example_mesh(opts=opts) for opts in opts_choices]
    else:
        meshes = []

        for n in range(4):
            uploaded_file = st.file_uploader(f'Upload your mesh',
                                             key=f'upload_{n}')
            if uploaded_file:
                mesh = load_mesh(uploaded_file)
                meshes.append(mesh)
            else:
                st.stop()

            if not st.checkbox('Upload another?', key=f'cb_upload_{n}'):
                break

    if not meshes:
        st.stop()

    st.header('Metrics')

    metrics_list = st.multiselect('Select which metrics to plot',
                                  default='area',
                                  options=list(metrics._metric_dispatch))

dfs = []

with st.expander('Meshes'):

    for mesh in meshes:
        c1, c2 = st.columns(2)
        with c1:
            fig = get_meshplot(mesh)
            st.bokeh_chart(fig, use_container_width=True)
        with c2:
            st.text(mesh)

        df = pd.DataFrame(metrics.calculate_all_metrics(mesh))
        dfs.append(df)

import altair as alt

LEVEL = 'mesh_num'
df = pd.concat({i: df for i, df in enumerate(dfs)}, names=(LEVEL, ))
df = df.reset_index(level=LEVEL)

c = alt.Chart(df).mark_bar(opacity=0.3, binSpacing=1).encode(
    x=alt.X('area:Q', bin=alt.Bin(maxbins=50)),
    y=alt.Y('count()', stack=None),  # zero, center, normalize
    color=f'{LEVEL}:N',
    column=f'{LEVEL}:N',
    tooltip=['count(area)'])

c.properties(title='area', )

st.altair_chart(c)

st.stop()

# df = metrics.calculate_all_metrics(meshes[0])

# for metric in metrics_list:
#     fig = get_metric_hist(mesh, metric)
#     st.bokeh_chart(fig, use_container_width=True)

#     fig = get_metric_2dplot(mesh, metric)
#     st.bokeh_chart(fig, use_container_width=True)

# from IPython import embed
# embed()

# for metric in metrics:
# plot metric 2D
# bar chart with metric values

# for multiple meshes
# 2d plots side by side
# bar charts overlap
