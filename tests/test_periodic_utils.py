import meshio

from nanomesh.periodic_utils import insert_periodic_info


def test_insert_periodic_info():
    """Test insertion of periodic info."""
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ]
    cells = [
        ('triangle', [[0, 1, 2], [1, 3, 2]]),
        ('quad', [[1, 4, 5, 3]]),
    ]

    mesh = meshio.Mesh(
        points,
        cells,
        point_data={'T': [0.3, -1.2, 0.5, 0.7, 0.0, -3.0]},
        cell_data={'a': [[0.1, 0.2], [0.4]]},
    )

    new_mesh = insert_periodic_info(mesh, [0, 0, 0, 1.0, 1.0, 0.0])

    assert isinstance(new_mesh, meshio.Mesh)
    assert hasattr(new_mesh, 'gmsh_periodic')
    assert mesh.gmsh_periodic == [
        [
            6,
            (0, 0),
            None,
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [1, 4],
                [3, 5],
            ],
        ],
    ]
