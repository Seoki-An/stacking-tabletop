#!/usr/bin/env python3

"""Smoke-test diffsim grasp simulation with the tabletop SR gripper."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
from typing import Iterable

# This Open3D build reads EGL_PLATFORM while Python itself starts. Restart once
# with the surfaceless backend when a headless video render was requested.
video_requested = any(
    arg == "--video" or arg.startswith("--video=") for arg in sys.argv[1:]
)
if video_requested and not os.environ.get("EGL_PLATFORM"):
    environment = os.environ.copy()
    environment["EGL_PLATFORM"] = "surfaceless"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)

import imageio.v2 as imageio
import inrol_urdf_parser as urdf
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsimpy import diffsim
from model import get_stone_model
from planning import get_grasp_init, get_planner, planner
from utils import SuppressOutput
from utils.dsf import DiffSupportSimple
from utils.wavefront import WavefrontImporter

SR_GRIPPER_URDF = REPO_ROOT / "assets/sr_gripper/sr_gripper.urdf"
UR5E_URDF = REPO_ROOT / "assets/ur5e/ur5e.urdf"


def _parse_vec(text: str, size: int, name: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=float)
    if values.size != size:
        raise argparse.ArgumentTypeError(
            f"{name} must contain {size} comma-separated numbers"
        )
    return values


def _copy_mesh(mesh: o3d.geometry.TriangleMesh, color: Iterable[float]):
    result = copy.deepcopy(mesh)
    result.paint_uniform_color(color)
    return result


def _load_sr_gripper_geometries():
    model = urdf.URDF()
    model.Parse(str(SR_GRIPPER_URDF))

    visual_meshes = {}
    collision_meshes = {}
    mesh_cache = {}
    for geometry_name in model.GetGeometryNames():
        geometry = model.GetGeometry(geometry_name)
        mesh_path = Path(geometry.mesh_path)
        if not mesh_path.is_absolute():
            mesh_path = (
                SR_GRIPPER_URDF.parent / mesh_path
                if geometry.visual
                else REPO_ROOT / mesh_path
            )
        mesh_path = mesh_path.resolve()

        if geometry.visual and geometry.type == "mesh":
            if mesh_path not in mesh_cache:
                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                if mesh.is_empty():
                    raise FileNotFoundError(f"mesh empty or not found: {mesh_path}")
                mesh.compute_vertex_normals()
                mesh.compute_triangle_normals()
                mesh_cache[mesh_path] = mesh
            visual_meshes[geometry_name] = mesh_cache[mesh_path]
        elif not geometry.visual and geometry.type == "dsf_vert":
            if mesh_path not in mesh_cache:
                mesh = o3d.geometry.TriangleMesh()
                for dsf_object in WavefrontImporter(str(mesh_path)).get_objects():
                    sharpness = float(
                        np.asarray(dsf_object.sharpness).reshape(-1)[0]
                    )
                    dsf = DiffSupportSimple(
                        vertex_set=dsf_object.vertices.T,
                        sharpness=sharpness,
                    )
                    vertices, triangles = dsf.get_mesh(resolution=2)
                    mesh += o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(vertices),
                        o3d.utility.Vector3iVector(triangles),
                    )
                # Smooth support sampling can produce nearly coincident points on
                # flat/cylindrical regions. Merge them before orienting the hull;
                # otherwise Open3D renders overlapping, inward-facing triangles.
                mesh.merge_close_vertices(1e-6)
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_degenerate_triangles()
                mesh.remove_unreferenced_vertices()
                if not mesh.orient_triangles():
                    raise ValueError(f"DSF mesh is not orientable: {mesh_path}")
                mesh.compute_vertex_normals()
                mesh.compute_triangle_normals()
                mesh_cache[mesh_path] = mesh
            collision_meshes[geometry_name] = mesh_cache[mesh_path]

    return model, visual_meshes, collision_meshes


def _make_gripper_geometries(model, source_meshes, grasp, color):
    theta = grasp.opening_angle
    model.SetState(np.array([theta, -theta, theta, -theta]))

    grasp_pose = grasp.pose.as_matrix()
    geometries = []
    for geometry_name, source_mesh in source_meshes.items():
        geometry = model.GetGeometry(geometry_name)
        local_pose = np.eye(4)
        local_pose[:3, :3] = geometry.GetRotation()
        local_pose[:3, 3] = geometry.GetPosition()

        mesh = _copy_mesh(source_mesh, color)
        mesh.transform(grasp_pose @ local_pose)
        geometries.append(mesh)
    return geometries


def _make_trajectory_geometries(model, source_meshes, grasp_sequence, max_frames: int):
    if not grasp_sequence or max_frames <= 0:
        return []

    frame_ids = np.linspace(
        0,
        len(grasp_sequence) - 1,
        min(max_frames, len(grasp_sequence)),
        dtype=int,
    )
    geometries = []
    for frame_idx, grasp_idx in enumerate(frame_ids):
        alpha = frame_idx / max(len(frame_ids) - 1, 1)
        color = [0.15 * (1.0 - alpha), 0.35, 0.9 * (1.0 - alpha)]
        geometries.extend(
            _make_gripper_geometries(
                model,
                source_meshes,
                grasp_sequence[grasp_idx],
                color,
            )
        )
    return geometries


def _make_material(color):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = color
    return material


def _render_grasp_video(
    video_path: Path,
    stone_mesh,
    target,
    gripper_model,
    gripper_meshes,
    initial_grasp,
    grasp_sequence,
    fps: int,
    frame_stride: int,
    show_initial: bool,
):
    if fps <= 0:
        raise ValueError("--fps must be positive")
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = list(range(0, len(grasp_sequence), frame_stride))
    if not frame_ids:
        grasp_frames = [initial_grasp]
    else:
        if frame_ids[-1] != len(grasp_sequence) - 1:
            frame_ids.append(len(grasp_sequence) - 1)
        grasp_frames = [grasp_sequence[index] for index in frame_ids]

    width, height = 960, 720
    with SuppressOutput():
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    center = np.asarray(target.pose.position()) + np.array([0.0, 0.0, 0.08])
    camera = center + np.array([0.35, -0.45, 0.25])
    renderer.scene.camera.look_at(center, camera, [0.0, 0.0, 1.0])
    renderer.scene.scene.set_indirect_light_intensity(25000)

    materials = {
        "ground": _make_material([0.45, 0.55, 0.42, 1.0]),
        "stone": _make_material([0.62, 0.62, 0.58, 1.0]),
        "initial": _make_material([0.15, 0.35, 0.90, 1.0]),
        "current": _make_material([0.10, 0.75, 0.25, 1.0]),
    }

    stone = copy.deepcopy(stone_mesh)
    stone.transform(target.pose.as_matrix())
    ground = o3d.geometry.TriangleMesh.create_box(0.6, 0.6, 0.001)
    ground.translate([-0.3, -0.3, -0.001])
    ground.compute_vertex_normals()
    initial_gripper = (
        _make_gripper_geometries(
            gripper_model, gripper_meshes, initial_grasp, [0.15, 0.35, 0.90]
        )
        if show_initial
        else []
    )

    print(f"Rendering {len(grasp_frames)} video frames...")
    with imageio.get_writer(str(video_path), fps=fps) as writer:
        for grasp in grasp_frames:
            scene = renderer.scene
            scene.add_geometry("ground", ground, materials["ground"])
            scene.add_geometry("stone", stone, materials["stone"])
            for index, mesh in enumerate(initial_gripper):
                scene.add_geometry(
                    f"initial_gripper_{index}", mesh, materials["initial"]
                )

            current_gripper = _make_gripper_geometries(
                gripper_model, gripper_meshes, grasp, [0.10, 0.75, 0.25]
            )
            for index, mesh in enumerate(current_gripper):
                scene.add_geometry(f"gripper_{index}", mesh, materials["current"])

            writer.append_data(np.asarray(renderer.render_to_image()))
            scene.clear_geometry()

    print(f"Video saved to {video_path}")


def _make_sr_planner(n_threads: int):
    _, config = get_planner(n_threads=n_threads)
    config.gripper.file_name = SR_GRIPPER_URDF
    config.manipulator.file_name = UR5E_URDF
    config.manipulator.end_effector_name = "tool0"

    config.gripper.height = 0.22
    config.gripper.virtual_closing_torque = 0.5
    config.gripper.virtual_approach_force = 5.0
    config.gripper.sim_damping = 5.0
    config.gripper.sim_step = 500
    config.gripper.sim_dt = 0.01
    config.gripper.set_sim_grasp_offset_tol(np.array([0.05, 0.10, 0.30]))
    config.gripper.set_opening_angle_range(np.array([0.0, 0.9]))

    config.graspgen.separate_margin = 0.003
    config.graspgen.plane_separate_margin = 0.001
    config.graspgen.contact_margin = 0.006
    config.graspgen.gripper_max_width = 0.18

    return planner.Context(config), config


def _simulate_grasp(context, target, grasp, plane_id, num_step, time_step):
    if hasattr(context, "simulate_grasp"):
        return context.simulate_grasp(
            target,
            grasp,
            object_ids=[],
            plane_ids=[plane_id],
            num_step=num_step,
            time_step=time_step,
        )
    if hasattr(context, "simulate_single_grasp"):
        return context.simulate_single_grasp(target, grasp, num_step, time_step)
    raise RuntimeError(
        "The installed diffsimpy has no grasp-simulation binding. "
        "Rebuild ../diffsim/interop/python."
    )


def _print_trajectory(trajectory):
    print("simulate_grasp result")
    print(f"  feasible      : {trajectory.is_feasible}")
    print(f"  frames        : {len(trajectory.grasp_sequence)}")
    if not trajectory.grasp_sequence:
        return

    final_grasp = trajectory.grasp_sequence[-1]
    print(f"  opening_angle : {final_grasp.opening_angle:.6f} rad")
    print(f"  position      : {np.asarray(final_grasp.pose.position())}")
    print(f"  orientation   : {np.asarray(final_grasp.pose.orientation())}")
    print(f"  contacts      : {len(trajectory.contact_points)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a tabletop stone grasp with the SR gripper."
    )
    parser.add_argument("--stone-id", type=int, default=1)
    parser.add_argument(
        "--stone-position",
        type=lambda text: _parse_vec(text, 3, "stone-position"),
        default=np.array([0.0, 0.0, 0.025]),
    )
    parser.add_argument(
        "--stone-rpy",
        type=lambda text: _parse_vec(text, 3, "stone-rpy"),
        default=np.zeros(3),
        help="stone roll,pitch,yaw in degrees",
    )
    parser.add_argument(
        "--grasp-offset",
        type=lambda text: _parse_vec(text, 3, "grasp-offset"),
        default=np.array([0.0, 0.0, 0.20]),
        help="initial gripper-base offset from the stone position",
    )
    parser.add_argument(
        "--grasp-rpy",
        type=lambda text: _parse_vec(text, 3, "grasp-rpy"),
        default=np.array([180.0, 0.0, 0.0]),
        help="initial gripper roll,pitch,yaw in degrees",
    )
    parser.add_argument("--opening-angle", type=float, default=0.8)
    parser.add_argument("--num-step", type=int, default=500)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--trajectory-frames", type=int, default=8)
    parser.add_argument(
        "--video",
        type=Path,
        help="write the simulated trajectory to a video file",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="render smooth DSF surfaces instead of the visual meshes",
    )
    parser.add_argument(
        "--skip-solve",
        action="store_true",
        help="simulate the initial grasp without grasp optimization",
    )
    parser.add_argument("--no-visualize", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    context, _ = _make_sr_planner(args.n_threads)
    stone_meshes, stone_configs, _, _ = get_stone_model()
    gripper_model, visual_meshes, collision_meshes = (
        _load_sr_gripper_geometries()
    )
    gripper_meshes = collision_meshes if args.show_collision else visual_meshes

    if args.stone_id not in stone_configs:
        available = ", ".join(str(i) for i in sorted(stone_configs))
        raise KeyError(f"unknown stone id {args.stone_id}; available: {available}")

    target = copy.deepcopy(stone_configs[args.stone_id])
    target.pose.setPosition(args.stone_position)
    target.pose.setOrientation(
        Rotation.from_euler("xyz", args.stone_rpy, degrees=True).as_quat()
    )

    grasp = get_grasp_init(
        args.stone_position + args.grasp_offset,
        np.deg2rad(args.grasp_rpy),
        euler_order="xyz",
    )
    grasp.opening_angle = args.opening_angle

    plane_id = context.add_plane(diffsim.PlaneGeometryConfig())
    if not args.skip_solve:
        solution = context.solve_single_grasp(target, grasp, True)
        print("solve_single_grasp seed")
        print(f"  feasible      : {solution.is_feasible}")
        print(f"  score         : {solution.score:.6g}")
        grasp = solution.grasp

    trajectory = _simulate_grasp(
        context,
        target,
        grasp,
        plane_id,
        args.num_step,
        args.time_step,
    )
    _print_trajectory(trajectory)

    if args.video:
        _render_grasp_video(
            args.video,
            stone_meshes[args.stone_id],
            target,
            gripper_model,
            gripper_meshes,
            grasp,
            trajectory.grasp_sequence,
            args.fps,
            args.frame_stride,
            not args.show_collision,
        )

    if args.no_visualize:
        return

    stone_mesh = _copy_mesh(stone_meshes[args.stone_id], [0.62, 0.62, 0.58])
    stone_mesh.transform(target.pose.as_matrix())

    initial_gripper = (
        []
        if args.show_collision
        else _make_gripper_geometries(
            gripper_model, gripper_meshes, grasp, [0.15, 0.35, 0.90]
        )
    )
    grasp_sequence = trajectory.grasp_sequence
    final_grasp = grasp_sequence[-1] if grasp_sequence else grasp
    trajectory_grippers = (
        []
        if args.show_collision
        else _make_trajectory_geometries(
            gripper_model,
            gripper_meshes,
            grasp_sequence,
            args.trajectory_frames,
        )
    )
    final_gripper = _make_gripper_geometries(
        gripper_model, gripper_meshes, final_grasp, [0.10, 0.75, 0.25]
    )

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    ground = o3d.geometry.TriangleMesh.create_box(0.6, 0.6, 0.001)
    ground.translate([-0.3, -0.3, -0.001])
    ground.paint_uniform_color([0.45, 0.55, 0.42])
    o3d.visualization.draw_geometries(
        [origin, ground, stone_mesh]
        + initial_gripper
        + trajectory_grippers
        + final_gripper,
        window_name=(
            "SR grasp DSF collision geometry"
            if args.show_collision
            else "SR grasp visual geometry"
        ),
    )


if __name__ == "__main__":
    main()
