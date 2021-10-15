import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from nanomesh.mesh_container import TriangleMesh
from nanomesh.mesh_utils import simple_triangulate
from nanomesh.utils import SliceViewer, requires, show_image


@pytest.mark.parametrize('condition,expected', ((True, True), (False, None)))
def test_requires(condition, expected):
    """Test `requires` functionality."""
    @requires(condition=condition, message=f'condition: {condition}')
    def func():
        return True

    assert func() == expected


def test_SliceViewer_fails():
    """Test `utils.SliceViewer fails`."""
    data = np.arange(27).reshape(3, 3, 3)
    sv = SliceViewer(data, update_delay=0)

    with pytest.raises(TypeError):  # argument index missing
        sv.update(along='x')

    with pytest.raises(KeyError):
        sv.update(index=0, along='FAIL')

    plt.close()


@pytest.mark.parametrize('along,index,slice', (
    ('x', 0, np.s_[:, :, 0]),
    ('y', 1, np.s_[:, 1, :]),
    ('z', 2, np.s_[2, :, :]),
))
def test_SliceViewer(along, index, slice):
    """Test `utils.SliceViewer`."""
    data = np.arange(27).reshape(3, 3, 3)
    sv = SliceViewer(data, update_delay=0)

    sv.update(along=along, index=index)
    arr = sv.im.get_array()

    np.testing.assert_array_equal(arr, data[slice])

    plt.close()


@image_comparison(
    baseline_images=['show_image'],
    remove_text=True,
    extensions=['png'],
    savefig_kwarg={'bbox_inches': 'tight'},
)
def test_show_image():
    """Test `utils.show_image`"""
    data = np.arange(25).reshape(5, 5)
    show_image(data, dpi=80, title='TESTING')


@pytest.fixture
def simple_mesh():
    return TriangleMesh(points=np.array([
        [0., 0.],
        [0., 1.],
        [1., 1.],
        [1., 0.],
    ]),
                        cells=np.array([[1, 0, 3], [3, 2, 1]]))


def test_simple_triangulate(simple_mesh):
    """Test simple mesh creation."""
    points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    mesh = simple_triangulate(points, opts='q30a1')

    np.testing.assert_equal(mesh.points, simple_mesh.points)
    np.testing.assert_equal(mesh.cells, simple_mesh.cells)
    np.testing.assert_equal(mesh.labels, simple_mesh.labels)


@pytest.mark.parametrize('side,shape', (
    ('left', (0, -1, 1, 1)),
    ('right', (0, 0, 1, 2)),
    ('top', (0, 0, 2, 1)),
    ('bottom', (-1, 0, 1, 1)),
))
def test_pad_side(simple_mesh, side, shape):
    """Test `side` parameter for `pad`."""
    res = simple_mesh.pad(side=side, width=1, opts='q30a1')

    assert isinstance(res, TriangleMesh)

    res_shape = np.hstack([
        res.points.min(axis=0),
        res.points.max(axis=0),
    ])

    np.testing.assert_equal(res_shape, shape)


@pytest.mark.parametrize('width', (0.0, 0.5, 1, np.pi, 100))
def test_pad_width(simple_mesh, width):
    """Test `width` parameter for `pad`."""
    res = simple_mesh.pad(side='left', width=width, opts='q30a1')

    assert isinstance(res, TriangleMesh)

    res_shape = np.hstack([
        res.points.min(axis=0),
        res.points.max(axis=0),
    ])

    np.testing.assert_equal(res_shape, (0, -width, 1, 1))


@pytest.mark.parametrize('label,expected_labels', (
    (None, (0, 0, 1, 1)),
    (0, (0, 0, 0, 0)),
    (2, (0, 0, 2, 2)),
    (np.pi, (0, 0, np.pi, np.pi)),
))
def test_pad_label(simple_mesh, label, expected_labels):
    """Test `label` parameter for `pad`."""
    res = simple_mesh.pad(side='left', width=1, opts='q30a1', label=label)

    assert isinstance(res, TriangleMesh)

    np.testing.assert_equal(res.labels, expected_labels)
