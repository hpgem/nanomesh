import meshio
import pyvista as pv


def tetrahedra_to_mesh(points, faces, mask=None):
    """Convert list of tetrahedra and mask to `meshio.Mesh`"""
    if mask is not None:
        faces = faces[mask]

    cells = [
        ('tetra', faces),
    ]

    mesh = meshio.Mesh(points, cells)
    mesh.remove_orphaned_nodes()

    return mesh


def triangles_to_mesh(points, faces, mask=None):
    """Convert list of triangles and mask to `meshio.Mesh`"""
    if mask is not None:
        faces = faces[mask]

    cells = [
        ('triangle', faces),
    ]

    mesh = meshio.Mesh(points, cells)
    mesh.remove_orphaned_nodes()

    return mesh


def meshio_to_polydata(mesh):
    return pv.from_meshio(mesh)


def meshio_to_trimesh(mesh):
    raise NotImplementedError


def trimesh_to_meshio(mesh):
    raise NotImplementedError


def trimesh_to_polydata(mesh):
    raise NotImplementedError


def polydata_to_meshio(mesh):
    raise NotImplementedError


def polydata_to_trimesh(mesh):
    raise NotImplementedError
