import numpy as np
from matplotlib.testing.decorators import image_comparison

from nanomesh.mesh import TriangleMesh
from nanomesh.mesh2d import compare_mesh_with_image


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
