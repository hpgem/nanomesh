from pathlib import Path

import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from nanomesh import metrics
from nanomesh.mesh_container import MeshContainer


@pytest.fixture
def sample_mesh():
    expected_fn = Path(__file__).parent / 'segmented_mesh_2d.msh'
    mesh = MeshContainer.read(expected_fn)

    sample_mesh = mesh.get('triangle')

    return sample_mesh


@pytest.mark.parametrize('inplace', (False, True))
def test_metrics(sample_mesh, inplace):
    ret = metrics.calculate_all_metrics(sample_mesh, inplace=inplace)

    for metric, output in ret.items():
        assert isinstance(metric, str)
        assert metric in metrics._metric_dispatch

        if inplace:
            assert metric in sample_mesh.cell_data

        assert isinstance(output, np.ndarray)
        assert len(output) == len(sample_mesh.cells)

    sample_mesh.cell_data.clear()


@image_comparison(
    baseline_images=['test_metrics_histogram'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_histogram(sample_mesh):
    metrics.histogram(sample_mesh, metric='area')


@image_comparison(
    baseline_images=['test_metrics_plot2d'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_plot2d(sample_mesh):
    metrics.plot2d(sample_mesh, metric='area')
