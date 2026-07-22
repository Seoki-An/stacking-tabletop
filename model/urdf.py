import copy
from typing import Dict

import numpy as np
import open3d as o3d
import inrol_urdf_parser as urdf


def apply_nonuniform_scale(
    mesh: o3d.geometry.TriangleMesh,
    scale: tuple = (1.0, 1.0, 1.0),
    center: tuple = None,
):
    """
    mesh: Open3D mesh
    scale: (sx, sy, sz)
    center: scaling reference center
    """
    sx, sy, sz = scale
    if center is None:
        center = mesh.get_center()
    center = np.asarray(center)

    T1 = np.eye(4)
    T1[:3, 3] = -center
    S = np.eye(4)
    S[0, 0] = sx
    S[1, 1] = sy
    S[2, 2] = sz  # scale
    T2 = np.eye(4)
    T2[:3, 3] = center

    M = T2 @ S @ T1
    mesh.transform(M)


def update_urdf_mesh(
    model: urdf.URDF, link_meshes: Dict[str, o3d.geometry.TriangleMesh], q: np.ndarray
) -> Dict[str, o3d.geometry.TriangleMesh]:
    model.SetState(q)
    transformed_meshes = {}
    for name, mesh in link_meshes.items():
        mesh = copy.deepcopy(mesh)
        link_frame = np.eye(4)
        link_frame[:3, :3] = model.GetLink(name).geoms[0].GetRotation()
        link_frame[:3, -1] = model.GetLink(name).geoms[0].GetPosition()
        mesh = mesh.transform(link_frame)
        transformed_meshes[name] = mesh

    return transformed_meshes
