#!/usr/bin/env python3
import os
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import inrol_urdf_parser as urdf
from diffsimpy import diffsim

from model import update_urdf_mesh


def simulate_lidar(
    model: urdf.URDF,
    model_meshes: Dict[str, o3d.geometry.TriangleMesh],
    scene_meshes: Dict[str, o3d.geometry.TriangleMesh],
    fov: List[float],
    lidar_name: str,
    q: np.ndarray,
    visualize: bool = False,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )

    o3d_device = o3d.core.Device("CPU:0")
    scene = o3d.t.geometry.RaycastingScene()

    plane = o3d.t.geometry.TriangleMesh.create_box(30.0, 30.0, 0.001)
    plane = plane.translate((-15, -15, -0.001))

    num_az = int(fov[0] / 0.15)
    num_el = int(fov[1] / 0.36)
    num_rays = num_az * num_el

    DEG2RAD = np.pi / 180.0
    phi, theta = np.meshgrid(
        np.linspace(-fov[0] / 2 * DEG2RAD, fov[0] / 2 * DEG2RAD, num_az),
        np.linspace((90 - fov[1] / 2) * DEG2RAD, (90 + fov[1] / 2) * DEG2RAD, num_el),
    )
    phi = phi.reshape(-1)
    theta = theta.reshape(-1)

    rays = np.zeros((num_rays, 6), dtype=np.float32)
    rays[:, 3:] = np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )

    model_meshes_t = update_urdf_mesh(model, model_meshes, q)

    scene.add_triangles(plane.to(o3d_device))
    for mesh in model_meshes_t.values():
        mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_tensor.to(o3d_device))
    for mesh in scene_meshes.values():
        mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_tensor.to(o3d_device))

    lidar_frame = np.eye(4)
    lidar_frame[:3, :3] = model.GetLink(lidar_name).GetRotation()
    lidar_frame[:3, -1] = model.GetLink(lidar_name).GetPosition()

    rays[:, :3] = lidar_frame[:3, -1]
    rays[:, 3:] = np.einsum("ij,nj -> ni", lidar_frame[:3, :3], rays[:, 3:])
    rays_tensor = o3d.core.Tensor(rays)
    raycast_result = scene.cast_rays(rays_tensor)
    t_hit = raycast_result["t_hit"].numpy()
    normals = raycast_result["primitive_normals"].numpy()
    mask = np.isfinite(t_hit)
    t_hit = t_hit[mask]
    normals = normals[mask]
    rays_valid = rays[mask]

    lidar = rays_valid[:, :3] + t_hit[:, None] * rays_valid[:, 3:]
    lidar_pcd = o3d.geometry.PointCloud()
    lidar_pcd.points = o3d.utility.Vector3dVector(lidar)

    origin = origin.transform(lidar_frame)

    if visualize:
        o3d.visualization.draw_geometries(
            [mesh for mesh in model_meshes_t.values()]
            + [mesh for mesh in scene_meshes.values()]
            + [origin, lidar_pcd]
        )

    return lidar_pcd.transform(np.linalg.inv(lidar_frame)), lidar_frame
