from typing import List

import meshio
import numpy as np


def insert_periodic_info(mesh: meshio.Mesh,
                         boundary: np.ndarray) -> meshio.Mesh:
    """Insert periodic info in the mesh.

    Parameters
    ----------
    mesh : meshio.Mesh
        Input mesh.
    boundary : np.ndarray
        Boundary array consisting of [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns
    -------
    meshio.Mesh
        Mesh with periodic info inserted.
    """

    mesh.remove_orphaned_nodes()
    map_index = map_boundary_points(mesh, np.array(boundary))
    mesh.gmsh_periodic = [[len(map_index), (0, 0), None, list(map_index)]]

    return mesh


def map_boundary_points(mesh: meshio.Mesh, boundary: np.ndarray) -> list:
    """Create a mapping between the boundary points.

    Parameters
    ----------
    mesh : meshio.Mesh
        Input mesh.
    boundary : np.array
        Boundary array consisting of [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns
    -------
    map_index : list
        Mapping of the boundary as a list

    Raises:
        ValueError: If multiple points are found
    """

    idx_out_domain = get_index_out_domain(mesh.points, boundary)
    map_index = []
    for idx in idx_out_domain:
        idx_match = get_index_match(mesh.points, idx, boundary)

        if len(idx_match) == 0:
            print(' Warning : No mapping point found at ', idx,
                  mesh.points[idx])
            continue

        if len(idx_match) > 1:
            raise ValueError('Multiple mapping points found at ', idx,
                             mesh.points[idx])

        map_index.append([idx_match[0], idx])
    return map_index


def get_index_out_domain(points: np.ndarray,
                         boundary: np.ndarray) -> np.ndarray:
    """get the index of the points that are outside the domain defined by
    boundary.

    Parameters
    ----------
    points : np.array
        Coordinates of the vertices
    boundary : np.array
        Boundary array consisting of [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    xyz0 = boundary[:3]
    xyz1 = boundary[3:]

    idx_out_domain: List[np.ndarray] = []
    for ix, x in enumerate(xyz0):
        idx_out_domain.extend(list(np.where(points[:, ix] <= x)[0]))

    for ix, x in enumerate(xyz1):
        idx_out_domain.extend(list(np.where(points[:, ix] >= x)[0]))

    return np.unique(idx_out_domain)


def get_trans_vect(point, boundary: np.ndarray) -> np.ndarray:
    """Get a translation vector to bring an outside point inside the domain.

    Parameters
    ----------
    point : np.array
        Position of the vertex
    boundary : np.array
        Boundary array consisting of [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns
    -------
    trans : (3,) np.ndarray
        Translation vector
    """

    xyz0 = boundary[:3]
    xyz1 = boundary[3:]
    length = xyz1 - xyz0

    trans = np.zeros(3)
    trans += (point < xyz0).astype('int') * length
    trans -= (point > xyz1).astype('int') * length

    return trans


def get_index_match(points, idx_root, boundary, eps=1E-4):
    """Get the index of the matching point of the idx0 point.

    Parameters
    ----------
    points : np.ndarray
        List of coordinates
    idx_root : ?
        Description
    boundary : np.array
        Boundary array consisting of [xmin, ymin, zmin, xmax, ymax, zmax]
    eps : float, optional
        Description

    Returns
    -------
    np.ndarray
        Description
    """
    trans = get_trans_vect(points[idx_root], boundary)
    return np.where(
        np.linalg.norm(points - (points[idx_root] + trans), axis=1) < eps)[0]
