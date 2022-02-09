import numpy as np
import pytest
from helpers import image_comparison2

from nanomesh import Mesher2D
from nanomesh.image2mesh import compare_mesh_with_image
from nanomesh.mesh import TriangleMesh
from nanomesh.plotting import linetrianglemeshplot


@image_comparison2(baseline_images=['line_mesh'])
def test_line_mesh_plot(line_tri_mesh):
    lines = line_tri_mesh.get('line')
    lines.plot_mpl()


@image_comparison2(baseline_images=['linetrianglemeshplot'])
def test_line_triangle_mesh_plot(line_tri_mesh):
    linetrianglemeshplot(line_tri_mesh)


@image_comparison2(baseline_images=['triangle_mesh'])
def test_triangle_mesh_plot(line_tri_mesh):
    lines = line_tri_mesh.get('triangle')
    lines.plot_mpl()


@pytest.mark.xfail(pytest.OS_DOES_NOT_MATCH_DATA_GEN,
                   raises=AssertionError,
                   reason=('https://github.com/hpgem/nanomesh/issues/144'))
@image_comparison2(baseline_images=[
    'contour_plot_fields', 'contour_plot_floating', 'contour_plot_all'
])
def test_contour_plot(segmented_image):
    np.random.seed(1234)  # for region marker coords
    mesher = Mesher2D(segmented_image)
    mesher.generate_contour(max_contour_dist=5, level=0.5)

    mesher.plot_contour(legend='fields')
    mesher.plot_contour(legend='floating')
    mesher.plot_contour(legend='all')


@image_comparison2(baseline_images=['compare_mesh_with_image'])
def test_compare_with_mesh():
    image = np.zeros([5, 5])

    points = np.array([
        [0, 0],
        [0, 2],
        [2, 2],
        [2, 0],
        [3, 3],
        [3, 4],
        [4, 4],
        [4, 3],
    ])
    cells = np.array([
        [0, 1, 2],
        [0, 3, 2],
        [4, 5, 6],
        [4, 7, 6],
    ])

    # 1: small square, 0: big square
    labels = np.array([0, 0, 1, 1])
    mesh = TriangleMesh(points=points, cells=cells, labels=labels)

    compare_mesh_with_image(image, mesh, legend='all')
