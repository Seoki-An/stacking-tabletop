import os
import pathlib
import numpy as np
from typing import Dict, Tuple

import open3d as o3d
import inrol_urdf_parser as urdf
from diffsimpy import diffsim

from .urdf import apply_nonuniform_scale
from utils.wavefront import WavefrontImporter
from utils.dsf import DiffSupportSimple
from utils.geometry import fix_normals_outward

EXCAVATOR_PATH = os.path.join(os.getcwd(), "assets/excavator_3")
GRIPPER_PATH = os.path.join(os.getcwd(), "assets/stone_grab_3")


def _model_id_from_asset_path(path: pathlib.Path) -> int:
    stem = path.stem
    if stem.endswith("_mesh"):
        stem = stem[: -len("_mesh")]
    return int(stem.split("_")[-1])


def get_stone_model(
    root_dir: str = "assets/stone",
) -> Tuple[
    Dict[int, o3d.geometry.TriangleMesh],
    Dict[int, diffsim.BodyConfig],
    Dict[int, o3d.geometry.PointCloud],
    Dict[int, o3d.geometry.TriangleMesh],
]:
    stone_dir = pathlib.Path(os.path.join(os.getcwd(), root_dir))

    dsf_meshes = {}
    meshes = {}
    configs = {}
    obj_paths = list(stone_dir.rglob("*.obj"))

    for obj_path in obj_paths:
        id = _model_id_from_asset_path(obj_path)
        dsf_importer = WavefrontImporter(str(obj_path))
        vertices_list = []
        triangles_list = []
        offset = 0
        for dsf_obj in dsf_importer.get_objects():
            dsf = DiffSupportSimple(
                vertex_set=dsf_obj.vertices.T,
                sharpness=dsf_obj.sharpness,
            )
            vertices, triangles = dsf.get_mesh(resolution=4)
            vertices_list.append(vertices)
            triangles_list.append(triangles + offset)
            offset += vertices.shape[0]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            np.concatenate(vertices_list, axis=0)
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            np.concatenate(triangles_list, axis=0)
        )

        mesh.orient_triangles()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh = fix_normals_outward(mesh)

        dsf_meshes[id] = mesh
        configs[id] = diffsim.BodyConfig(str(obj_path))

    pcds = {}
    pcd_paths = list(stone_dir.rglob("*.pcd"))
    for pcd_path in pcd_paths:
        id = _model_id_from_asset_path(pcd_path)
        pcds[id] = o3d.io.read_point_cloud(str(pcd_path))

    meshes = {}
    mesh_paths = list(stone_dir.rglob("*.ply"))
    for mesh_path in mesh_paths:
        id = _model_id_from_asset_path(mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.orient_triangles()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh = fix_normals_outward(mesh)
        meshes[id] = mesh

    return dsf_meshes, configs, pcds, meshes


def get_excavator_model() -> Tuple[urdf.URDF, Dict[str, o3d.geometry.TriangleMesh]]:

    urdf_path = os.path.join(EXCAVATOR_PATH, "vdk23_cx.urdf")
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    model = urdf.URDF()
    model.Parse(urdf_path)

    link_names = [
        "base_link",
        "cs_cor",
        "cs_boom",
        "cs_arm",
        "cs_bucket",
        "cs_tilt",
        "cs_rotate",
        "grip_body",
        "grip_left",
        "grip_right",
    ]
    link_meshes = {}
    for name in link_names:
        link = model.GetLink(name)
        geom = link.geoms[0]
        mesh = o3d.io.read_triangle_mesh(geom.mesh_path)
        if mesh.is_empty():
            raise FileNotFoundError(f"Mesh empty or not found: {geom.mesh_path}")

        mesh_scale = geom.mesh_scale if geom.mesh_scale is not None else [1, 1, 1]
        apply_nonuniform_scale(mesh, mesh_scale, np.zeros(3))
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        link_meshes[name] = mesh

    return model, link_meshes


def get_gripper_model() -> Tuple[urdf.URDF, Dict[str, o3d.geometry.TriangleMesh]]:

    urdf_path = os.path.join(GRIPPER_PATH, "stone_grab.urdf")
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    model = urdf.URDF()
    model.Parse(urdf_path)

    link_names = [
        "grip_body",
        "grip_left",
        "grip_right",
    ]
    link_meshes = {}
    for name in link_names:
        link = model.GetLink(name)
        geom = link.geoms[0]
        mesh = o3d.io.read_triangle_mesh(geom.mesh_path)
        if mesh.is_empty():
            raise FileNotFoundError(f"Mesh empty or not found: {geom.mesh_path}")

        mesh_scale = geom.mesh_scale if geom.mesh_scale is not None else [1, 1, 1]
        apply_nonuniform_scale(mesh, mesh_scale, np.zeros(3))
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        link_meshes[name] = mesh

    return model, link_meshes
