import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from nanomesh import Mesher2D
from nanomesh.mesh import TriangleMesh
from nanomesh.mesh2d import compare_mesh_with_image
from nanomesh.mpl.meshplot import plot_line_triangle


@image_comparison(
    baseline_images=['line_mesh'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_line_mesh_plot(mesh_square2d):
    lines = mesh_square2d.get('line')
    lines.plot_mpl(label='data')


@image_comparison(
    baseline_images=['line_triangle_plot'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_line_triangle_mesh_plot(mesh_square2d):
    plot_line_triangle(mesh_square2d, label='data')


@image_comparison(
    baseline_images=['triangle_mesh'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_triangle_mesh_plot(mesh_square2d):
    lines = mesh_square2d.get('triangle')
    lines.plot_mpl(label='data')


@pytest.mark.xfail(pytest.OS_MATCHES_DATA_GEN,
                   raises=AssertionError,
                   reason=('https://github.com/hpgem/nanomesh/issues/144'))
@image_comparison(
    baseline_images=['contour_plot'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_contour_plot(segmented_image):
    np.random.seed(1234)  # for region marker coords
    mesher = Mesher2D(segmented_image)
    mesher.generate_contour(max_contour_dist=5, level=0.5)
    mesher.show_contour()


@image_comparison(
    baseline_images=['compare_mesh_with_image'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
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

    compare_mesh_with_image(image, mesh)
