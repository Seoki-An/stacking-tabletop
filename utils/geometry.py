import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull, HalfspaceIntersection
from typing import List, Tuple


def compute_polytope_vertices(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Arguments:
        A = np.array(shape=(nineq, ndim))
        b = np.array(shape=(nineq, 1)) or np.array(shape=(nineq))
        A, b defines linear inequalities Ax <= b
    Returns:
        intersections = np.array(shape=(nvert, ndim))
    """
    assert A.ndim == 2, "the shape of A should be (n, m)"
    if b.ndim == 1:
        halfspaces = np.hstack([A, -b.reshape(-1, 1)])
    else:
        halfspaces = np.hstack([A, -b])

    norm_vector = np.reshape(np.linalg.norm(A, axis=1), (-1, 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((A, norm_vector))
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    if res.status == 0 and res.x is not None and res.x[-1] > 0:
        interior_point = res.x[:-1]
        try:
            intersect = HalfspaceIntersection(halfspaces, interior_point)
            return intersect.intersections
        except:
            return None
    else:
        return None


def compute_polytope_halfspaces(vertices: np.ndarray):
    """
    Arguments:
        vertices = np.array(shape=(nverts, ndim))
    Returns:
        A = np.array(shape=(nineq, ndim))
        b = np.array(shape=(nineq))
        A, b defines linear inequalities Ax <= b
        which represents a convex hull formed by the input vertices
    """
    assert (
        vertices.ndim == 2 and vertices.shape[-1] == 3
    ), "input vertices array's shape should be (n, 3)"

    hull = ConvexHull(vertices)
    Ab = hull.equations
    return (Ab[:, :3], -Ab[:, -1]), hull.volume


def compute_intersection(
    halfList1: List[Tuple[np.ndarray, np.ndarray]],
    halfList2: List[Tuple[np.ndarray, np.ndarray]],
) -> List[np.ndarray]:
    """
    Arguments:
        halfList = list(half spaces defined by tuple A, b defining convex by Ax <= b)
    Returns:
        verticesList = intersection vertices defined by halfList1, halfList2
    """
    verticesList = []
    for half1 in halfList1:
        for half2 in halfList2:
            A1, b1 = half1
            A2, b2 = half2
            A = np.vstack([A1, A2])
            b = np.hstack([b1.flatten(), b2.flatten()])
            vertices = compute_polytope_vertices(A, b)
            if vertices is not None:
                verticesList.append(vertices)

    return verticesList


def half_to_vertices(halfList: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
    verticesList = []
    for half in halfList:
        A, b = half
        vertices = compute_polytope_vertices(A, b)
        if vertices is not None:
            verticesList.append(vertices)
    return verticesList


def vertices_to_mesh(vertices: List[np.ndarray]) -> List[pv.PolyData]:
    """
    Arguments:
        vertices = list(vertices of polytope) or numpy array
        polydata = pyvista polydata object
    """
    if type(vertices) == list:
        mesh_list = []
        for V in vertices:
            polytope = pv.PolyData(V)
            polytope = polytope.delaunay_3d(alpha=5.0)
            polytope = polytope.extract_surface()
            mesh_list.append(polytope)

        return mesh_list

    else:
        polytope = pv.PolyData(vertices)
        polytope = polytope.delaunay_3d(alpha=5.0)
        polytope = polytope.extract_surface()

        return polytope


def vertices_to_hull(
    vertices: List[np.ndarray], calc_vol: bool = False
) -> Tuple[pv.PolyData, float]:
    """
    Arguments:
        vertices = list(numpy.array(size=(N, 3))) or numpy.array(size=(N, 3))
        polydata = pyvista polydata object
    """
    if type(vertices) == list:
        poly_list = []
        volume_list = []
        for V in vertices:
            hull = ConvexHull(V)

            simplices = np.zeros(hull.simplices.shape, dtype=np.int32)
            for idx in range(hull.vertices.shape[0]):
                simplices[hull.simplices == hull.vertices[idx]] = idx

            faces = np.column_stack(
                (
                    3 * np.ones((len(simplices), 1), dtype=np.int32),
                    simplices,
                )
            ).flatten()
            poly = pv.PolyData(hull.points[hull.vertices], faces)
            poly.compute_normals(
                point_normals=True,
                cell_normals=False,
                split_vertices=False,
                consistent_normals=True,
                auto_orient_normals=True,
                inplace=True,
            )

            poly_list.append(poly)
            volume_list.append(hull.volume)
        if calc_vol:
            return poly_list, volume_list
        else:
            return poly_list
    else:
        hull = ConvexHull(vertices)
        simplices = np.zeros(hull.simplices.shape, dtype=np.int32)
        for idx in range(hull.vertices.shape[0]):
            simplices[hull.simplices == hull.vertices[idx]] = idx

        faces = np.column_stack(
            (
                3 * np.ones((len(simplices), 1), dtype=np.int32),
                simplices,
            )
        ).flatten()
        poly = pv.PolyData(hull.points[hull.vertices], faces)
        poly.compute_normals(
            point_normals=True,
            cell_normals=False,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=True,
            inplace=True,
        )

        if calc_vol:
            return poly, hull.volume
        else:
            return poly


def compute_point_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:

    n_vertices = len(vertices)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
    point_normals = np.zeros((n_vertices, 3), dtype=np.float64)

    for i in range(3):
        np.add.at(point_normals, faces[:, i], face_normals)

    point_normals /= np.linalg.norm(point_normals, axis=1, keepdims=True) + 1e-12

    assert len(np.unique(faces.reshape(-1))) == len(
        vertices
    ), f"v in faces {len(np.unique(faces.reshape(-1)))} != v {len(vertices)}"
    return point_normals


def rigidTrans_half(
    pose: np.ndarray, A: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arguments:
        pose = [position, quaternion] numpy vector
        A = np.array(shape=(n,3))
        b = np.array(shape=(n,1)) or np.array(shape=(n))
    Return:
        output = (A_, b_)
        tuple of transformed half spaces (A_, b_)
    - Apply rigidbody transform to half spaces spaces
    """
    position = pose[:3]
    R = Rotation.from_quat(pose[3:])
    A_ = R.apply(A)
    if b.ndim == 1:
        b = b.reshape((-1, 1))
    b_ = b + A_ @ position[:, None]

    return A_, b_


def rigidTrans_vertex(pose: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Arguments:
        pose = [position, quaternion] numpy vector
        vertices = N X 3 matrix
    Return:
        transformed vertices
    - Apply rigidbody transform to vertices
    """
    position = pose[:3].copy()
    R = Rotation.from_quat(pose[3:].copy())
    vertices_new = R.apply(vertices.copy()) + position[None, :]
    return vertices_new


def pose_to_transformation_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Arguments:
        pose = [position, quaternion] numpy vector
    Return:
        transformation matrix = 4 X 4 matrix
    - Convert pose to transformation matrix
    """
    position = pose[:3].copy()
    R = Rotation.from_quat(pose[3:]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def pyvista_to_o3d(poly: pv.PolyData) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(poly.points)
    faces = poly.faces.reshape((-1, 4))[:, 1:4]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh


def fix_normals_outward(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    center = vertices.mean(axis=0)

    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)

    flipped = 0

    for i, tri in enumerate(triangles):
        v0 = vertices[tri[0]]
        n = normals[i]

        if np.dot(n, v0 - center) < 0:
            triangles[i] = [tri[0], tri[2], tri[1]]
            flipped += 1

    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    return mesh
