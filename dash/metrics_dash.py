import os
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from bokeh_plots import get_meshplot

from nanomesh import MeshContainer, data, metrics

st.title('Nanomesh - meshboard')
st.write('Upload your own mesh or use the example data to generate metrics!')


def load_mesh(data_file):
    suffix = Path(data_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(dir='.', suffix=suffix, delete=False)
    tmp.write(data_file.getbuffer())
    tmp.close()
    mesh = MeshContainer.read(tmp.name).get('triangle')
    os.unlink(tmp.name)

    return mesh


@st.cache
def load_example_mesh(opts):
    print('loading', opts)
    return data.blob_mesh2d(opts=opts).get('triangle')


meshes = []

opts_choices = (
    'q30a10',
    'a10',
    'q30',
    'q30a5',
)

with st.sidebar:

    if st.checkbox('Use example data'):
        meshes = [load_example_mesh(opts=opts) for opts in opts_choices]
    else:
        st.text('Upload up to 4 meshes')
        meshes = []

        for n in range(4):
            st.header(f'Mesh #{n+1}')
            uploaded_file = st.file_uploader('Upload your mesh',
                                             key=f'upload_{n}')
            if uploaded_file:
                mesh = load_mesh(uploaded_file)
                meshes.append(mesh)
            else:
                break

    if not meshes:
        st.stop()

    st.header('Metrics')

    metrics_list = st.multiselect('Select which metrics to plot',
                                  default='area',
                                  options=list(metrics._metric_dispatch))

dfs = []

with st.expander('Meshes'):

    for i, mesh in enumerate(meshes):
        st.header(f'Mesh #{i+1}')

        c1, c2 = st.columns(2)
        with c1:
            fig = get_meshplot(mesh)
            st.bokeh_chart(fig, use_container_width=True)
        with c2:
            st.text(mesh)

        df = pd.DataFrame(metrics.calculate_all_metrics(mesh))
        dfs.append(df)

LEVEL = 'Mesh index'
df = pd.concat({i: df for i, df in enumerate(dfs)}, names=(LEVEL, ))
df = df.reset_index(level=LEVEL)

for metric in metrics_list:
    st.header(metric)
    c = alt.Chart(df).mark_bar(opacity=0.3, binSpacing=1).encode(
        x=alt.X(f'{metric}:Q', bin=alt.Bin(maxbins=50)),
        y=alt.Y('count()', stack=None),  # zero, center, normalize
        color=f'{LEVEL}:N',
        column=f'{LEVEL}:N',
        tooltip=[f'count({metric})'])

    c.properties(title=f'{metric}', )

    st.altair_chart(c)
