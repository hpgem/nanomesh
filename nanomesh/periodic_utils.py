import pygalmesh
import meshio
import numpy as np

def insert_periodic_info(mesh, boundary):
    """insert periodic info in the mesh."""

    mesh.remove_orphaned_nodes()
    map_index = map_boundary_points(mesh, np.array(boundary))
    mesh.gmsh_periodic = [[len(map_index), (0, 0), None, list(map_index)]]

    return mesh

def map_boundary_points(mesh, boundary):
    """create a mapping between the boundary points.
    
    Args:
        mesh (meshio.mesh) : an instance of a meshio mesh
        boudnary (np.array) : boundary [xmin, ymin, zmin, xmax, ymax, zmax]
    """

    idx_out_domain = get_index_out_domain(mesh.points, boundary)
    map_index = []
    for idx in idx_out_domain:
        idx_match = get_index_match(mesh.points, idx, boundary)
        if len(idx_match) != 1:
            raise ValueError('error in mapping sorry')
        map_index .append([idx_match[0], idx])
    return map_index

def get_index_out_domain(points, boundary):
    """get the index of the points that are outside the domain define by boundary.
    
    Args:
        points (np.array) : positions of the vertices
        boudnary (np.array) : boundary [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    xyz0 = boundary[:3]
    xyz1 = boundary[3:]

    idx_out_domain = []
    for ix, x in enumerate(xyz0):
        idx_out_domain = idx_out_domain + list(np.where(points[:,ix] < x)[0])

    for ix, x in enumerate(xyz1):
        idx_out_domain = idx_out_domain + list(np.where(points[:,ix] > x)[0])

    return np.unique(idx_out_domain)

def get_trans_vect(point, boundary):
    """get a translation vector to bring an outside point inside the domain
    
    Args:
        point (np.array) : positions of the vertexx
        boudnary : boundary [xmin, ymin, zmin, xmax, ymax, zmax]
    """

    xyz0 = boundary[:3]
    xyz1 = boundary[3:]
    length = xyz1 - xyz0

    trans = np.zeros(3)
    trans += (point<xyz0).astype('int') * length
    trans -= (point>xyz1).astype('int') * length

    return trans

def get_index_match(pts, idx_root, boundary):
    """Get the index of the matching point of the idx0 point."""
    trans = get_trans_vect(pts[idx_root], boundary)
    return np.where(np.linalg.norm(pts - (pts[idx_root]+trans), axis=1)<1E-6)[0]