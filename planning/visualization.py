import copy
import tqdm
import numpy as np
from typing import Dict, List

import imageio
import open3d as o3d
import inrol_urdf_parser as urdf
from model import update_urdf_mesh
from utils import SuppressOutput


def trajectory_visualization(
    path: List[np.ndarray],
    model: urdf.URDF,
    model_meshes: Dict[str, o3d.geometry.TriangleMesh],
    scene_meshes: Dict[str, o3d.geometry.TriangleMesh],
    save_path: str,
    camera_center: np.ndarray = np.zeros(3),
    camera_position: np.ndarray = np.array([0, 0, 5]),
    pcd: o3d.geometry.PointCloud = None,
    fps: float = 30.0,
):
    width, height = 1280, 1024
    with SuppressOutput():
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    mat_scene = o3d.visualization.rendering.MaterialRecord()
    mat_scene.shader = "defaultLit"
    mat_scene.base_color = [0.8, 0.8, 0.8, 0.2]

    mat_ground = o3d.visualization.rendering.MaterialRecord()
    mat_ground.shader = "defaultLit"
    mat_ground.base_color = [0.5, 0.7, 0.3, 0.5]

    mat_pcd = o3d.visualization.rendering.MaterialRecord()
    mat_pcd.shader = "defaultLit"
    mat_pcd.base_color = [0.2, 0.2, 0.8, 0.9]

    scene.camera.look_at(
        camera_center, camera_position, [0, 0, 1]  # target  # camera position
    )  # up direction

    plane = o3d.t.geometry.TriangleMesh.create_box(30.0, 30.0, 0.001)
    plane = plane.translate((-15, -15, -0.001))
    if pcd:
        pcd = pcd.translate((0, 0, 0.001))

    frames = []

    for q_t in tqdm.tqdm(path, "Generate video: "):
        excavator_meshes_t = update_urdf_mesh(model, model_meshes, q_t)
        for name, mesh in excavator_meshes_t.items():
            scene.add_geometry(str(name), mesh, mat)
        for name, mesh in scene_meshes.items():
            scene.add_geometry(str(name), mesh, mat_scene)
        if pcd:
            scene.add_geometry("pcd", pcd, mat_pcd)
        scene.add_geometry("ground", plane, mat_ground)
        img = renderer.render_to_image()
        scene.clear_geometry()

        frame = np.asarray(img)
        frames.append(frame.copy())

    imageio.mimsave(save_path, frames, fps=fps)

    print(f"Trajectory planning result is saved in {save_path}")


def trajectory_visualization_with_target(
    q_path: List[np.ndarray],
    target_path: List[np.ndarray],
    model: urdf.URDF,
    model_meshes: Dict[str, o3d.geometry.TriangleMesh],
    scene_meshes: Dict[str, o3d.geometry.TriangleMesh],
    target_mesh: o3d.geometry.TriangleMesh,
    save_path: str,
    camera_center: np.ndarray = np.zeros(3),
    camera_position: np.ndarray = np.array([0, 0, 5]),
    pcd: o3d.geometry.PointCloud = None,
    wall_meshes: Dict[str, o3d.geometry.TriangleMesh] = None,
    fps: float = 30.0,
):
    width, height = 1280, 1024
    with SuppressOutput():
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    mat_scene = o3d.visualization.rendering.MaterialRecord()
    mat_scene.shader = "defaultLit"
    mat_scene.base_color = [0.8, 0.8, 0.8, 0.2]

    mat_wall = o3d.visualization.rendering.MaterialRecord()
    mat_wall.shader = "defaultLitTransparency"
    mat_wall.has_alpha = True
    mat_wall.base_color = [0.78, 0.78, 0.88, 0.18]

    mat_ground = o3d.visualization.rendering.MaterialRecord()
    mat_ground.shader = "defaultLit"
    mat_ground.base_color = [0.5, 0.7, 0.3, 0.5]

    mat_pcd = o3d.visualization.rendering.MaterialRecord()
    mat_pcd.shader = "defaultLit"
    mat_pcd.base_color = [0.2, 0.2, 0.8, 0.9]

    scene.camera.look_at(
        camera_center, camera_position, [0, 0, 1]  # target  # camera position
    )  # up direction

    plane = o3d.t.geometry.TriangleMesh.create_box(30.0, 30.0, 0.001)
    plane = plane.translate((-15, -15, -0.001))
    if pcd:
        pcd = pcd.translate((0, 0, 0.001))

    frames = []

    for q_t, pose_t in tqdm.tqdm(
        zip(q_path, target_path), "Generate video: ", total=len(q_path)
    ):
        excavator_meshes_t = update_urdf_mesh(model, model_meshes, q_t)
        target_mesh_t = copy.deepcopy(target_mesh).transform(pose_t)
        for name, mesh in excavator_meshes_t.items():
            scene.add_geometry(str(name), mesh, mat)
        for name, mesh in scene_meshes.items():
            scene.add_geometry(str(name), mesh, mat_scene)
        if wall_meshes:
            for name, mesh in wall_meshes.items():
                scene.add_geometry(str(name), mesh, mat_wall)
        if pcd:
            scene.add_geometry("pcd", pcd, mat_pcd)
        scene.add_geometry("ground", plane, mat_ground)
        scene.add_geometry("target", target_mesh_t, mat)
        img = renderer.render_to_image()
        scene.clear_geometry()

        frame = np.asarray(img)
        frames.append(frame.copy())

    imageio.mimsave(save_path, frames, fps=fps)

    print(f"Trajectory planning result is saved in {save_path}")


def generate_path_with_opening_angle(result, n_opening_angle=10):
    def _opening_angle(grasp_idx):
        if len(result.grasp_sequence) == 0:
            return 0.0
        grasp_idx = min(grasp_idx, len(result.grasp_sequence) - 1)
        return result.grasp_sequence[grasp_idx].opening_angle

    # Entry i is the angle at the START of sub-path i; entry i+1 is the angle
    # reached at its END (a stationary open/close transition is appended
    # there). Sequences of length 2/6/10 START with an in-hand carry (the
    # stone is already held), so the release must occur at the END of that
    # carry -- the following free retreat runs open. Length 4/8 start with a
    # free pick approach instead (close at its end, retreat holding).
    opening_angle_sequence = []
    if len(result.q_path_sequence) == 2:
        opening_angle_sequence = [
            _opening_angle(0),
            0.0,
            0.0,
        ]
    elif len(result.q_path_sequence) == 4:
        opening_angle_sequence = [
            0.0,
            _opening_angle(0),
            _opening_angle(0),
            0.0,
            0.0,
        ]
    elif len(result.q_path_sequence) == 6:
        opening_angle_sequence = [
            _opening_angle(0),
            0.0,
            0.0,
            _opening_angle(1),
            _opening_angle(1),
            0.0,
            0.0,
        ]
    elif len(result.q_path_sequence) == 8:
        opening_angle_sequence = [
            0.0,
            _opening_angle(0),
            _opening_angle(0),
            0.0,
            0.0,
            _opening_angle(2),
            _opening_angle(2),
            0.0,
            0.0,
        ]
    elif len(result.q_path_sequence) == 10:
        opening_angle_sequence = [
            _opening_angle(0),
            0.0,
            0.0,
            _opening_angle(1),
            _opening_angle(1),
            0.0,
            0.0,
            _opening_angle(3),
            _opening_angle(3),
            0.0,
            0.0,
        ]
    else:
        raise ValueError(
            f"Unexpected path sequence length: {len(result.q_path_sequence)}"
        )

    q_path = []
    target_path = []
    for i_path, path_sub in enumerate(result.q_path_sequence):
        start_angle = copy.deepcopy(opening_angle_sequence[i_path])
        end_angle = copy.deepcopy(opening_angle_sequence[i_path + 1])

        path_sub = [
            np.concatenate([np.copy(q_t), start_angle * np.ones(2)]) for q_t in path_sub
        ]
        target_sub = [
            pose_t.as_matrix() for pose_t in result.target_path_sequence[i_path]
        ]
        path_sub = path_sub + [
            np.concatenate(
                [
                    np.copy(path_sub[-1][:6]),
                    (
                        start_angle * (n_opening_angle - j) / n_opening_angle
                        + end_angle * j / n_opening_angle
                    )
                    * np.ones(2),
                ]
            )
            for j in range(n_opening_angle + 1)
        ]
        target_sub = target_sub + [target_sub[-1]] * (n_opening_angle + 1)

        q_path.extend(path_sub)
        target_path.extend(target_sub)
    return q_path, target_path
