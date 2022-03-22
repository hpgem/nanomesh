import pytest

from nanomesh import data


@pytest.mark.parametrize('func,ndim', (
    (data.binary_blobs2d, 2),
    (data.binary_blobs3d, 3),
))
def test_binaryblobsxd(func, ndim):
    length = 10
    arr = func(length=length)
    assert arr.ndim == ndim
    assert arr.shape == (length, ) * ndim
    assert arr.dtype == int
    assert arr.min() == 0
    assert arr.max() == 1


@pytest.mark.parametrize('func,shape', (
    (data.nanopores, (200, 200)),
    (data.nanopores_gradient, (700, 190)),
    (data.nanopores3d, (200, 200, 200)),
))
def test_nanopores_data(func, shape):
    arr = func()
    assert arr.shape == shape


def test_blob_mesh_2d():
    mesh = data.blob_mesh2d()
    assert {'triangle', 'line'} == set(mesh.cells_dict)


def test_blob_mesh_3d():
    mesh = data.blob_mesh3d()
    assert {'tetra'} == set(mesh.cells_dict)
