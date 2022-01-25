import numpy as np
import pytest
from helpers import image_comparison2

from nanomesh import metrics


@pytest.mark.parametrize('inplace', (False, True))
def test_metrics(sample_triangle_mesh, inplace):
    ret = metrics.calculate_all_metrics(sample_triangle_mesh, inplace=inplace)

    for metric, output in ret.items():
        assert isinstance(metric, str)
        assert metric in metrics._metric_dispatch

        if inplace:
            assert metric in sample_triangle_mesh.cell_data

        assert isinstance(output, np.ndarray)
        assert len(output) == len(sample_triangle_mesh.cells)

    sample_triangle_mesh.cell_data.clear()


@image_comparison2(baseline_images=['test_metrics_histogram'])
def test_histogram(sample_triangle_mesh):
    metrics.histogram(sample_triangle_mesh, metric='area')


@image_comparison2(baseline_images=['test_metrics_plot2d'])
def test_plot2d(sample_triangle_mesh):
    metrics.plot2d(sample_triangle_mesh, metric='area')
