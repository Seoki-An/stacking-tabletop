#!/usr/bin/env python3
import argparse
import copy
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT_DIR = Path(__file__).resolve().parents[2]
DIFFSIM_PY_BUILD = ROOT_DIR.parent / "diffsim" / "interop" / "python" / "build"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if DIFFSIM_PY_BUILD.exists() and str(DIFFSIM_PY_BUILD) not in sys.path:
    sys.path.insert(0, str(DIFFSIM_PY_BUILD))

from diffsimpy import sceneid
from agent.env.components.contexts import get_sceneid
from model import get_stone_model
from utils import pose_to_transformation_matrix

DEFAULT_SCENE_PCD = ROOT_DIR / ".data" / "scene_pcd" / "scene_0522_1.pcd"
DEFAULT_ACTION_SEQUENCE = (
    ROOT_DIR
    / "sessions"
    / "old"
    / "plans"
    / "260522_1"
    / "plan_2"
    / "action_sequence.pkl"
)
DEFAULT_RECONSTRUCTED_STATE_NAME = "state_sceneid.pkl"
DEFAULT_PLAN_STATE_NAME = "state.pkl"


def _logs_sceneid():
    from scripts.desktop import sceneid_from_logs

    return sceneid_from_logs


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_action_sequence(path: Path) -> list[dict]:
    data = load_pickle(path)
    if not isinstance(data, list):
        raise TypeError(f"expected a list of actions in {path}, got {type(data)!r}")
    return data


def target_offset_for_action_sequence(action_path: Path) -> np.ndarray:
    params_path = action_path.with_name("planning_params.pkl")
    if not params_path.exists():
        return np.zeros(2, dtype=float)

    params = load_pickle(params_path)
    return np.asarray(params.get("target_structure_offset", np.zeros(2)), dtype=float)


def asset_dir_for_action_sequence(action_path: Path, fallback: str) -> str:
    config_path = action_path.with_name("config.yml")
    if not config_path.exists():
        return fallback

    text = config_path.read_text()
    match = re.search(r"load_dir:.*?\n(?:.*\n){0,12}?\s+_val:\s*([^\n]+)", text)
    if match:
        return match.group(1).strip().strip("\"'")
    return fallback


def read_scene_pcd(pcd_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise ValueError(f"empty point cloud: {pcd_path}")
    return pcd


def state_from_state_pkl(path: Path):
    data = load_pickle(path)
    if not isinstance(data, dict):
        return data, {}
    state = data.get("resume_state", None)
    if state is None:
        for step in reversed(data.get("steps", [])):
            if not isinstance(step, dict):
                continue
            state = step.get("resume_state", None) or step.get("raw_state", None)
            if state is not None:
                break
    if state is None:
        raise ValueError(f"no resume/raw state found in {path}")
    return state, data


def state_world_pose_entries(
    state,
    target_offset: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    stone_set = getattr(state, "stone_set", None)
    stone_seq = getattr(state, "stone_seq", None)
    stone_poses = getattr(state, "stone_poses", None)
    if stone_set is None or stone_seq is None or stone_poses is None:
        raise ValueError("state is missing stone_set, stone_seq, or stone_poses")

    entries = []
    for stone_idx in stone_seq:
        stone_id = int(stone_set[int(stone_idx)])
        if stone_id not in stone_poses:
            continue
        pose = np.asarray(stone_poses[stone_id], dtype=np.float64).copy()
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            continue
        pose[:2] += np.asarray(target_offset, dtype=np.float64)[:2]
        entries.append((stone_id, pose))
    return entries


def ground_plane_from_state_data(data: dict) -> np.ndarray | None:
    for key in ("reconstructed_from_logs", "reconstructed_from_sceneid"):
        item = data.get(key, None)
        if isinstance(item, dict) and item.get("ground_plane_model", None) is not None:
            plane = np.asarray(item["ground_plane_model"], dtype=np.float64)
            if plane.shape == (4,) and np.all(np.isfinite(plane)):
                return plane
    return None


def downsample_and_select_points(
    pcd: o3d.geometry.PointCloud,
    pcd_path: Path,
    voxel_size: float,
    max_points: int | None,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size)

    points = np.asarray(pcd.points, dtype=np.float64)
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(points), size=max_points, replace=False)
        pcd = pcd.select_by_index(np.sort(indices))
        points = np.asarray(pcd.points, dtype=np.float64)

    if len(points) == 0:
        raise ValueError(f"no scene points left after filtering: {pcd_path}")

    # .T of a C-contiguous (N,3) array is already Fortran-contiguous (3,N),
    # which matches the f_style Eigen::Map in the C++ binding.
    return pcd, np.asfortranarray(points.T)


def planned_pose_entries(
    action_sequence: list[dict],
    target_offset: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    entries = []
    for action in action_sequence:
        stone_id = int(action["stone_id"])
        pose = np.asarray(action["pose"], dtype=np.float64).copy()
        pose[:2] += target_offset[:2]
        entries.append((stone_id, pose))
    return entries


def motion_log_feasible_entries(path: Path) -> dict[int, tuple[int, float]]:
    entries: dict[int, tuple[int, float]] = {}
    if path is None or not path.exists():
        return entries

    pattern = re.compile(
        r"step=(?P<step>\d+).*?stone_id=(?P<stone>\d+)"
        r".*?place_world_y=(?P<y>[-+0-9.eE]+).*?feasible=True"
    )
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        entries[int(match.group("step"))] = (
            int(match.group("stone")),
            float(match.group("y")),
        )
    return entries


def extend_step_sequence_from_motion_log(
    step_sequence: list[int],
    step_index: int,
    motion_log_path: Path | None,
) -> tuple[list[int], dict[int, float]]:
    if motion_log_path is None or len(step_sequence) >= step_index:
        return step_sequence, {}

    feasible_entries = motion_log_feasible_entries(motion_log_path)
    extended = list(step_sequence)
    motion_log_y_by_stone = {}
    for step in range(len(extended) + 1, step_index + 1):
        if step not in feasible_entries:
            raise ValueError(
                f"motion log {motion_log_path} has no feasible entry for step {step}"
            )
        stone_id, world_y = feasible_entries[step]
        extended.append(stone_id)
        motion_log_y_by_stone[stone_id] = world_y

    print(
        "motion log extra steps: "
        + ", ".join(
            f"step {step} -> stone {extended[step - 1]}"
            for step in range(len(step_sequence) + 1, step_index + 1)
        )
    )
    return extended, motion_log_y_by_stone


def append_motion_log_y_seed_poses(
    planned_entries: list[tuple[int, np.ndarray]],
    initial_pose_sources: dict[int, str],
    step_sequence: list[int],
    motion_log_y_by_stone: dict[int, float],
) -> list[tuple[int, np.ndarray]]:
    if not motion_log_y_by_stone:
        return planned_entries

    existing = {stone_id: pose for stone_id, pose in planned_entries}
    existing_poses = np.vstack([pose for _, pose in planned_entries])
    seed_x = float(np.median(existing_poses[:, 0]))
    seed_z = float(np.median(existing_poses[:, 2]))
    updated = list(planned_entries)
    for stone_id in step_sequence:
        if stone_id in existing or stone_id not in motion_log_y_by_stone:
            continue
        pose = np.array(
            [
                seed_x,
                motion_log_y_by_stone[stone_id],
                seed_z,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )
        updated.append((stone_id, pose))
        initial_pose_sources[stone_id] = "motion_log_y_seed"

    n_seeded = sum(
        1 for stone_id in motion_log_y_by_stone if stone_id in initial_pose_sources
    )
    if n_seeded:
        print(
            "motion log pose seeds are y-only rough initializers; "
            "use --manual-init-gui to set x/z/orientation before ICP/SceneID"
        )
    return updated


def crop_scene_pcd_near_expected_poses(
    pcd: o3d.geometry.PointCloud,
    pcd_path: Path,
    planned_entries: list[tuple[int, np.ndarray]],
    visual_meshes: dict[int, o3d.geometry.TriangleMesh],
    margin: float,
    max_margin: float,
) -> o3d.geometry.PointCloud:
    mins = []
    maxs = []

    for stone_id, pose in planned_entries:
        if stone_id in visual_meshes:
            mesh = copy.deepcopy(visual_meshes[stone_id])
            mesh.transform(pose_to_transformation_matrix(pose))
            bbox = mesh.get_axis_aligned_bounding_box()
            mins.append(np.asarray(bbox.min_bound, dtype=float))
            maxs.append(np.asarray(bbox.max_bound, dtype=float))
        else:
            mins.append(pose[:3])
            maxs.append(pose[:3])

    if not mins:
        raise ValueError("cannot crop scene PCD without planned poses")

    base_min_bound = np.min(np.vstack(mins), axis=0)
    base_max_bound = np.max(np.vstack(maxs), axis=0)

    margins = [margin]
    while margins[-1] < max_margin:
        margins.append(min(max(margins[-1] * 2.0, margins[-1] + 0.1), max_margin))

    cropped = o3d.geometry.PointCloud()
    min_bound = base_min_bound - margin
    max_bound = base_max_bound + margin
    used_margin = margin
    for used_margin in margins:
        min_bound = base_min_bound - used_margin
        max_bound = base_max_bound + used_margin
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        cropped = pcd.crop(bbox)
        if not cropped.is_empty():
            break

    print(
        "cropped scene PCD near expected poses: "
        f"{len(cropped.points)} / {len(pcd.points)} points, "
        f"margin={used_margin}, min={min_bound}, max={max_bound}"
    )

    if cropped.is_empty():
        raise ValueError(
            f"crop removed all scene points from {pcd_path}; "
            "try --no-crop, --no-offset, or a larger --crop-max-margin"
        )
    return cropped


def remove_ground_plane_ransac(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    normal_z_min: float,
    min_inlier_ratio: float,
) -> o3d.geometry.PointCloud:
    n_points = len(pcd.points)
    if n_points < ransac_n:
        print(
            "ground removal skipped: "
            f"only {n_points} points available for ransac_n={ransac_n}"
        )
        return pcd

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    normal = np.asarray(plane_model[:3], dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm > 0.0:
        normal /= normal_norm

    inlier_ratio = len(inliers) / n_points
    if abs(normal[2]) < normal_z_min:
        print(
            "ground removal skipped: "
            f"RANSAC plane normal={normal}, inliers={len(inliers)} / {n_points}"
        )
        return pcd
    if inlier_ratio < min_inlier_ratio:
        print(
            "ground removal skipped: "
            f"RANSAC plane inlier ratio {inlier_ratio:.3f} "
            f"< {min_inlier_ratio:.3f}"
        )
        return pcd

    filtered = pcd.select_by_index(inliers, invert=True)
    print(
        "removed ground plane by RANSAC: "
        f"{len(inliers)} / {n_points} points, "
        f"normal={normal}, d={plane_model[3]:.4f}, "
        f"remaining={len(filtered.points)}"
    )
    return filtered


def make_sceneid_config(args: argparse.Namespace) -> sceneid.Config:
    cfg = sceneid.Config()
    cfg.n_threads = args.n_threads
    cfg.log_interval = args.log_interval
    cfg.tr.max_iter = args.max_iter
    cfg.tr.eps = args.trust_region_eps
    cfg.tr.delta_init = args.delta_init
    cfg.obj.k_pcd = args.k_pcd
    cfg.obj.pcd_huber_delta = args.pcd_huber_delta
    cfg.obj.pcd_max_gap = args.pcd_max_gap
    cfg.obj.k_gap_c = args.k_gap_c
    cfg.obj.k_comp = args.k_comp
    cfg.graph.max_iter = args.graph_max_iter
    return cfg


def add_planned_bodies(
    context: sceneid.Context,
    planned_entries: list[tuple[int, np.ndarray]],
    stone_configs: dict[int, object],
    stone_pcds: dict[int, o3d.geometry.PointCloud],
) -> tuple[dict[int, int], dict[int, np.ndarray]]:
    sceneid_to_stone: dict[int, int] = {}
    initial_poses: dict[int, np.ndarray] = {}

    for step, (stone_id, pose) in enumerate(planned_entries, start=1):
        if stone_id not in stone_configs:
            available = ", ".join(str(i) for i in sorted(stone_configs))
            raise KeyError(f"stone id {stone_id} missing from assets: {available}")

        body_config = copy.deepcopy(stone_configs[stone_id])
        body_config.pose.setVector(pose)
        body_id = context.add_body(body_config)
        sceneid_to_stone[int(body_id)] = stone_id
        initial_poses[int(body_id)] = pose

        pcd_registered = False
        if stone_id in stone_pcds and not stone_pcds[stone_id].is_empty():
            pcd_pts = np.asfortranarray(
                np.asarray(stone_pcds[stone_id].points, dtype=np.float64).T
            )
            context.set_body_pcd(int(body_id), pcd_pts)
            pcd_registered = True

        print(
            f"add step {step:02d}: sceneid body {int(body_id):>3} "
            f"stone {stone_id:>3} init xyz={pose[:3]}"
            + (" [pcd]" if pcd_registered else " [dsf fallback]")
        )

    return sceneid_to_stone, initial_poses


def select_visual_meshes(dsf_meshes: dict, ply_meshes: dict) -> dict:
    meshes = dict(dsf_meshes)
    meshes.update(ply_meshes)
    return meshes


def posed_mesh(
    mesh: o3d.geometry.TriangleMesh,
    pose: np.ndarray,
    color: tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    geom = copy.deepcopy(mesh)
    geom.paint_uniform_color(color)
    geom.transform(pose_to_transformation_matrix(pose))
    geom.compute_vertex_normals()
    return geom


def posed_wireframe(
    mesh: o3d.geometry.TriangleMesh,
    pose: np.ndarray,
    color: tuple[float, float, float],
) -> o3d.geometry.LineSet:
    geom = posed_mesh(mesh, pose, color)
    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(geom)
    line_set.paint_uniform_color(color)
    return line_set


def plane_wire_grid(
    plane_model: np.ndarray,
    bounds: o3d.geometry.AxisAlignedBoundingBox,
    color: tuple[float, float, float] = (0.05, 0.75, 0.25),
    n_lines: int = 12,
) -> o3d.geometry.LineSet | None:
    a, b, c, d = np.asarray(plane_model, dtype=float)
    normal = np.array([a, b, c], dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-9:
        return None
    normal /= normal_norm
    d /= normal_norm

    center = np.asarray(bounds.get_center(), dtype=float)
    center -= (normal.dot(center) + d) * normal

    tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(tangent) < 1e-9:
        tangent = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)

    extent = np.asarray(bounds.get_extent(), dtype=float)
    half_size = max(float(np.linalg.norm(extent[:2])) * 0.6, 1.0)
    coords = np.linspace(-half_size, half_size, n_lines)

    points = []
    lines = []
    for u in coords:
        i = len(points)
        points.extend(
            [
                center + u * tangent - half_size * bitangent,
                center + u * tangent + half_size * bitangent,
            ]
        )
        lines.append([i, i + 1])
    for v in coords:
        i = len(points)
        points.extend(
            [
                center - half_size * tangent + v * bitangent,
                center + half_size * tangent + v * bitangent,
            ]
        )
        lines.append([i, i + 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.paint_uniform_color(color)
    return line_set


def visualize_result(
    scene_pcd: o3d.geometry.PointCloud,
    visual_meshes: dict[int, o3d.geometry.TriangleMesh],
    sceneid_to_stone: dict[int, int],
    initial_poses: dict[int, np.ndarray],
    optimal_poses: dict,
    ground_plane_model: np.ndarray | None = None,
) -> None:
    pcd = copy.deepcopy(scene_pcd)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
    )
    pcd.normalize_normals()
    pcd.paint_uniform_color([0.45, 0.45, 0.45])

    geoms: list[o3d.geometry.Geometry] = [
        pcd,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
    ]
    if ground_plane_model is not None:
        grid = plane_wire_grid(ground_plane_model, pcd.get_axis_aligned_bounding_box())
        if grid is not None:
            geoms.append(grid)

    for body_id, stone_id in sorted(sceneid_to_stone.items()):
        if stone_id not in visual_meshes:
            continue

        geoms.append(
            posed_wireframe(
                visual_meshes[stone_id],
                initial_poses[body_id],
                (0.1, 0.35, 1.0),
            )
        )

        pose = optimal_poses.get(body_id)
        if pose is None:
            continue
        geoms.append(
            posed_mesh(
                visual_meshes[stone_id],
                np.asarray(pose.vectorized()),
                (1.0, 0.45, 0.05),
            )
        )

    print(
        "visualization: scene PCD gray, planned wireframe blue, "
        "identified mesh orange, estimated ground plane green"
    )
    o3d.visualization.draw_geometries(
        geoms,
        window_name="sceneid result vs scene PCD",
        mesh_show_back_face=True,
    )


def visualize_state_vs_pcd(
    scene_pcd: o3d.geometry.PointCloud,
    visual_meshes: dict[int, o3d.geometry.TriangleMesh],
    state_entries: list[tuple[int, np.ndarray]],
    ground_plane_model: np.ndarray | None,
    state_path: Path,
) -> None:
    pcd = copy.deepcopy(scene_pcd)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
    )
    pcd.normalize_normals()
    pcd.paint_uniform_color([0.45, 0.45, 0.45])

    geoms: list[o3d.geometry.Geometry] = [
        pcd,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
    ]
    if ground_plane_model is not None:
        grid = plane_wire_grid(ground_plane_model, pcd.get_axis_aligned_bounding_box())
        if grid is not None:
            geoms.append(grid)

    shown = 0
    for stone_id, pose in state_entries:
        if stone_id not in visual_meshes:
            continue
        geoms.append(posed_mesh(visual_meshes[stone_id], pose, (1.0, 0.45, 0.05)))
        shown += 1

    print(
        f"state/PCD comparison: {state_path}, "
        f"{len(state_entries)} state poses ({shown} visualized), "
        "scene PCD gray, state meshes orange, estimated ground plane green"
    )
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"state vs scene PCD: {state_path.name}",
        mesh_show_back_face=True,
    )


def save_optimal_pose_output(
    path: Path,
    plan_dir: Path,
    action_path: Path,
    scene_pcd_path: Path,
    step_index: int,
    sceneid_to_stone: dict[int, int],
    initial_poses: dict[int, np.ndarray],
    initial_pose_sources: dict[int, str],
    optimal_poses_by_stone: dict[int, np.ndarray],
    ground_height: float | None,
    ground_plane_model: np.ndarray | None,
    skipped_sceneid_solve: bool,
    elapsed: float,
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    initial_poses_by_stone = {
        sceneid_to_stone[body_id]: pose.copy()
        for body_id, pose in initial_poses.items()
    }
    result = {
        "plan_dir": str(plan_dir),
        "action_sequence": str(action_path),
        "step_index": step_index,
        "scene_scan_dir": None,
        "scene_pcd_path": str(scene_pcd_path),
        "stone_ids": sorted(optimal_poses_by_stone),
        "initial_poses": initial_poses_by_stone,
        "initial_pose_sources": dict(initial_pose_sources),
        "optimal_poses": {
            stone_id: pose.copy() for stone_id, pose in optimal_poses_by_stone.items()
        },
        "ground_height": ground_height,
        "ground_plane_model": (
            ground_plane_model.copy() if ground_plane_model is not None else None
        ),
        "skipped_sceneid_solve": skipped_sceneid_solve,
        "elapsed": elapsed,
    }

    def write_pickle(output_path: Path) -> None:
        with output_path.open("wb") as f:
            pickle.dump(result, f)

    def write_json(output_path: Path) -> None:
        json_result = copy.deepcopy(result)
        json_result["initial_poses"] = {
            str(stone_id): pose.tolist()
            for stone_id, pose in json_result["initial_poses"].items()
        }
        json_result["optimal_poses"] = {
            str(stone_id): pose.tolist()
            for stone_id, pose in json_result["optimal_poses"].items()
        }
        json_result["ground_plane_model"] = (
            json_result["ground_plane_model"].tolist()
            if json_result["ground_plane_model"] is not None
            else None
        )
        with output_path.open("w") as f:
            json.dump(json_result, f, indent=2)
            f.write("\n")

    def write_txt(output_path: Path) -> None:
        with output_path.open("w") as f:
            f.write(f"plan_dir: {plan_dir}\n")
            f.write(f"action_sequence: {action_path}\n")
            f.write(f"step_index: {step_index}\n")
            f.write(f"scene_pcd_path: {scene_pcd_path}\n")
            f.write(f"ground_height: {ground_height}\n")
            f.write(
                "ground_plane_model: "
                f"{ground_plane_model.tolist() if ground_plane_model is not None else None}\n"
            )
            f.write(f"skipped_sceneid_solve: {skipped_sceneid_solve}\n")
            f.write(f"elapsed: {elapsed:.6f}\n")
            for stone_id in sorted(optimal_poses_by_stone):
                f.write(
                    f"stone {stone_id}: "
                    f"init_source={initial_pose_sources[stone_id]} "
                    f"init={initial_poses_by_stone[stone_id].tolist()} "
                    f"optimal={optimal_poses_by_stone[stone_id].tolist()}\n"
                )

    suffix = path.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        write_pickle(path)
        print(f"wrote optimal poses: {path}")
    elif suffix == ".json":
        write_json(path)
        print(f"wrote optimal poses: {path}")
    else:
        pkl_path = path.with_suffix(".pkl")
        txt_path = path.with_suffix(".txt")
        write_pickle(pkl_path)
        write_txt(txt_path)
        print(f"wrote optimal poses: {pkl_path}")
        print(f"wrote optimal poses: {txt_path}")


def write_reconstructed_state_for_generate_sequence(
    output_path: Path,
    plan_dir: Path,
    state_path: Path | None,
    scene_pcd_path: Path,
    step_index: int,
    placed_stone_ids: list[int],
    optimal_poses_by_stone: dict[int, np.ndarray],
    ground_height: float | None,
    ground_plane_model: np.ndarray | None,
    args: argparse.Namespace,
) -> None:
    logs_sceneid = _logs_sceneid()
    record = {
        "plan_dir": str(plan_dir),
        "state_path": str(state_path) if state_path is not None else None,
        "step_index": step_index,
        "placed_stone_ids": [int(stone_id) for stone_id in placed_stone_ids],
        "scene_dir": str(scene_pcd_path.parent),
        "scene_scan_dir": str(scene_pcd_path.parent),
        "scene_pcd_path": str(scene_pcd_path),
        "optimal_poses": {
            int(stone_id): pose.copy()
            for stone_id, pose in optimal_poses_by_stone.items()
        },
        "ground_height": ground_height,
        "ground_plane_model": (
            ground_plane_model.copy() if ground_plane_model is not None else None
        ),
    }
    reconstruct_args = argparse.Namespace(
        no_offset=args.no_offset,
        no_mark_pose_identified=args.no_mark_pose_identified,
    )
    logs_sceneid.write_reconstructed_state_pkl(
        record,
        reconstruct_args,
        output_path,
    )


def run_sceneid_from_plan(args: argparse.Namespace) -> None:
    logs_sceneid = _logs_sceneid()
    scene_pcd_path = args.scene_pcd.resolve()
    compare_state_path = (
        args.compare_state_pcd.resolve() if args.compare_state_pcd is not None else None
    )
    action_path = args.action_sequence.resolve()
    action_sequence_is_default = (
        args.action_sequence.resolve() == DEFAULT_ACTION_SEQUENCE.resolve()
    )
    plan_dir_arg = args.plan_dir.resolve() if args.plan_dir is not None else None
    if plan_dir_arg is not None:
        if not plan_dir_arg.is_dir():
            raise NotADirectoryError(f"plan directory not found: {plan_dir_arg}")
        plan_action_path = plan_dir_arg / "action_sequence.pkl"
        if action_sequence_is_default:
            if not plan_action_path.exists():
                raise FileNotFoundError(
                    f"action sequence not found in --plan-dir: {plan_action_path}"
                )
            action_path = plan_action_path.resolve()

    state_path = args.state_path.resolve() if args.state_path is not None else None
    if state_path is None and plan_dir_arg is not None:
        for candidate_name in (
            DEFAULT_RECONSTRUCTED_STATE_NAME,
            DEFAULT_PLAN_STATE_NAME,
        ):
            candidate = plan_dir_arg / candidate_name
            if candidate.exists():
                state_path = candidate.resolve()
                break

    plan_source_path = state_path or compare_state_path
    if plan_source_path is not None:
        if not plan_source_path.exists():
            raise FileNotFoundError(f"state file not found: {plan_source_path}")
        state_plan_dir = plan_source_path.parent
        state_action_path = state_plan_dir / "action_sequence.pkl"
        if action_sequence_is_default and state_action_path.exists():
            action_path = state_action_path.resolve()
        elif action_path.parent.resolve() != state_plan_dir.resolve():
            raise ValueError(
                "--state-path and --action-sequence must belong to the same "
                "plan directory. Either pass the matching action_sequence.pkl "
                "or omit --action-sequence so it is inferred from --state-path."
            )
    plan_dir = action_path.parent

    action_sequence = load_action_sequence(action_path)
    if args.step_index is None and state_path is not None:
        step_index = logs_sceneid.max_state_step_index(
            plan_dir,
            len(action_sequence),
            state_path,
        )
    else:
        step_index = (
            len(action_sequence) if args.step_index is None else args.step_index
        )
    max_step_index = (
        step_index
        if state_path is not None or args.motion_log_extra_steps is not None
        else len(action_sequence)
    )
    if step_index <= 0 or step_index > max_step_index:
        raise ValueError(
            f"--step-index must be in [1, {max_step_index}], got {step_index}"
        )
    step_sequence = (
        logs_sceneid.state_backed_step_sequence(
            plan_dir,
            action_sequence,
            step_index,
            state_path,
        )
        if state_path is not None
        else [
            logs_sceneid.action_stone_id(action)
            for action in action_sequence[: min(step_index, len(action_sequence))]
        ]
    )
    step_sequence, motion_log_y_by_stone = extend_step_sequence_from_motion_log(
        step_sequence,
        step_index,
        (
            args.motion_log_extra_steps.resolve()
            if args.motion_log_extra_steps is not None
            else None
        ),
    )
    if len(step_sequence) < step_index:
        raise ValueError(
            f"--step-index {step_index} exceeds available placed stones "
            f"from {state_path or action_path}; pass --motion-log-extra-steps "
            "to seed missing logged steps"
        )
    target_offset = (
        np.zeros(2)
        if args.no_offset
        else target_offset_for_action_sequence(action_path)
    )
    asset_dir = args.asset_dir or asset_dir_for_action_sequence(
        action_path, "assets/stone"
    )

    if compare_state_path is not None:
        print(f"scene PCD: {scene_pcd_path}")
        print(f"compare state: {compare_state_path}")
        print(f"action sequence: {action_path}")
        print(f"asset dir: {asset_dir}")
        print(f"target offset xy: {target_offset}")
        scene_pcd = read_scene_pcd(scene_pcd_path)
        dsf_meshes, _, _, ply_meshes = get_stone_model(asset_dir)
        visual_meshes = select_visual_meshes(dsf_meshes, ply_meshes)
        state, state_data = state_from_state_pkl(compare_state_path)
        state_entries = state_world_pose_entries(state, target_offset)
        visualize_state_vs_pcd(
            scene_pcd,
            visual_meshes,
            state_entries,
            ground_plane_from_state_data(state_data),
            compare_state_path,
        )
        return

    print(f"scene PCD: {scene_pcd_path}")
    print(f"action sequence: {action_path}")
    if state_path is not None:
        print(f"state path: {state_path}")
    print(f"step index: {step_index} / {len(step_sequence)} placed seeds")
    print(f"asset dir: {asset_dir}")
    print(f"target offset xy: {target_offset}")

    dsf_meshes, stone_configs, stone_pcds, ply_meshes = get_stone_model(asset_dir)
    visual_meshes = select_visual_meshes(dsf_meshes, ply_meshes)
    action_entries_by_stone = {
        stone_id: pose
        for stone_id, pose in planned_pose_entries(
            action_sequence[:step_index],
            target_offset,
        )
    }
    action_entries_plan = [
        (stone_id, action_entries_by_stone[stone_id])
        for stone_id in step_sequence
        if stone_id in action_entries_by_stone
    ]
    if state_path is not None:
        planned_entries_plan, initial_pose_sources = (
            logs_sceneid.state_pose_entries_from_plan_state(
                plan_dir,
                step_sequence,
                step_index,
                target_offset,
                action_entries_plan,
                state_path,
            )
        )
    else:
        planned_entries_plan = action_entries_plan
        initial_pose_sources = {
            stone_id: "plan" for stone_id, _ in planned_entries_plan
        }
    planned_entries_plan = append_motion_log_y_seed_poses(
        planned_entries_plan,
        initial_pose_sources,
        step_sequence,
        motion_log_y_by_stone,
    )

    scene_pcd = read_scene_pcd(scene_pcd_path)

    calibrated_ground_height = None
    calibrated_ground_plane_model = None
    if not args.no_ground_height_init:
        reference_xy = np.mean(
            np.vstack([pose[:2] for _, pose in planned_entries_plan]),
            axis=0,
        )
        ground_height_pcd = scene_pcd
        if not args.no_ground_height_crop:
            try:
                ground_height_pcd = crop_scene_pcd_near_expected_poses(
                    scene_pcd,
                    scene_pcd_path,
                    planned_entries_plan,
                    visual_meshes,
                    args.ground_height_crop_margin,
                    args.ground_height_crop_max_margin,
                )
            except Exception as exc:
                print(
                    "ground height crop skipped: "
                    f"{exc}; using full scene PCD for calibration"
                )
        ground_estimate = logs_sceneid.estimate_ground_height_ransac(
            ground_height_pcd,
            reference_xy,
            args.ground_height_distance_threshold,
            args.ground_ransac_n,
            args.ground_num_iterations,
            args.ground_normal_z_min,
            args.ground_min_inlier_ratio,
            args.ground_height_max_abs,
        )
        if ground_estimate is not None:
            calibrated_ground_height, calibrated_ground_plane_model = ground_estimate

    pose_ground_delta = logs_sceneid.ground_height_pose_delta(
        plan_dir,
        None if args.no_ground_height_init else calibrated_ground_height,
    )
    planned_entries = logs_sceneid.apply_ground_height_to_plan_poses(
        planned_entries_plan,
        pose_ground_delta,
    )
    logs_sceneid.mark_ground_height_sources(
        initial_pose_sources,
        planned_entries,
        pose_ground_delta,
    )
    planned_entries, manual_applied = logs_sceneid.apply_manual_initial_poses(
        planned_entries,
        initial_pose_sources,
        step_sequence,
        step_index,
        args.manual_init_by_step,
        args.manual_init_fixed,
    )
    if manual_applied:
        print(
            "manual init poses: "
            + ", ".join(
                f"step {step} -> stone {stone_id}" for step, stone_id in manual_applied
            )
            + (" (fixed)" if args.manual_init_fixed else " (ICP/sceneid refinable)")
        )
    if args.visualize:
        o3d.visualization.draw_geometries(
            [scene_pcd],
            window_name="input scene PCD before crop and ground removal",
        )

    if not args.no_crop:
        scene_pcd = crop_scene_pcd_near_expected_poses(
            scene_pcd,
            scene_pcd_path,
            planned_entries,
            visual_meshes,
            args.crop_margin,
            args.crop_max_margin,
        )

    visualization_pcd = o3d.geometry.PointCloud(scene_pcd)
    if args.visualize:
        o3d.visualization.draw_geometries(
            [visualization_pcd],
            window_name="cropped scene PCD before ground removal",
        )
    if not args.no_ground_removal:
        filtered_by_model = (
            None
            if args.no_ground_height_init
            else logs_sceneid.remove_ground_points_by_plane_model(
                scene_pcd,
                calibrated_ground_plane_model,
                args.ground_distance_threshold,
                args.ground_normal_z_min,
            )
        )
        if filtered_by_model is not None:
            scene_pcd = filtered_by_model
        else:
            scene_pcd = remove_ground_plane_ransac(
                scene_pcd,
                args.ground_distance_threshold,
                args.ground_ransac_n,
                args.ground_num_iterations,
                args.ground_normal_z_min,
                args.ground_min_inlier_ratio,
            )
        visualization_pcd = o3d.geometry.PointCloud(scene_pcd)

    if args.manual_init_gui:
        print("manual init GUI uses cropped scene PCD after ground removal")
        planned_entries, manual_gui_poses, _ = (
            logs_sceneid.run_manual_initialization_gui(
                scene_pcd,
                planned_entries,
                initial_pose_sources,
                step_sequence,
                step_index,
                visual_meshes,
                args,
                scene_pcd_path,
            )
        )
        manual_init_gui_output = (
            args.manual_init_gui_output
            if args.manual_init_gui_output is not None
            else plan_dir / f"manual_init_step{step_index}.json"
        )
        if manual_gui_poses:
            saved_manual_poses = dict(args.manual_init_by_step)
            saved_manual_poses.update(manual_gui_poses)
            logs_sceneid.save_manual_init_pose_file(
                manual_init_gui_output,
                saved_manual_poses,
            )
        args.manual_init_by_step.update(manual_gui_poses)

    if args.manual_fix_gui:
        args.fix_initial_steps = logs_sceneid.run_manual_fix_gui(
            scene_pcd,
            planned_entries,
            initial_pose_sources,
            step_sequence,
            step_index,
            visual_meshes,
            args,
            scene_pcd_path,
            args.fix_initial_steps,
        )
    user_fixed_steps = logs_sceneid.mark_user_fixed_step_sources(
        initial_pose_sources,
        step_sequence,
        step_index,
        args.fix_initial_steps,
    )
    if user_fixed_steps:
        print(
            "fixing user-specified initialized poses: "
            + ", ".join(
                f"step {step} -> stone {stone_id}"
                for step, stone_id in user_fixed_steps
            )
        )

    pre_icp_entries = [(stone_id, pose.copy()) for stone_id, pose in planned_entries]
    planned_entries = logs_sceneid.refine_plan_initial_poses_with_icp(
        scene_pcd,
        planned_entries,
        initial_pose_sources,
        stone_pcds,
        visual_meshes,
        args,
        {},
    )
    if args.visualize and not args.no_icp_init:
        sceneid_to_stone_icp = {
            body_id: stone_id
            for body_id, (stone_id, _) in enumerate(planned_entries, start=1)
        }
        pre_icp_by_stone = {stone_id: pose for stone_id, pose in pre_icp_entries}
        initial_poses_icp = {
            body_id: pre_icp_by_stone.get(stone_id, pose).copy()
            for body_id, (stone_id, pose) in enumerate(planned_entries, start=1)
        }
        refined_poses_icp = {
            body_id: logs_sceneid.PoseVector(pose.copy())
            for body_id, (_, pose) in enumerate(planned_entries, start=1)
        }
        print(
            "visualization after ICP: blue wireframe = before ICP, "
            "orange mesh = after ICP"
        )
        visualize_result(
            visualization_pcd,
            visual_meshes,
            sceneid_to_stone_icp,
            initial_poses_icp,
            refined_poses_icp,
            None if args.no_ground_height_init else calibrated_ground_plane_model,
        )

    scene_pcd, points = downsample_and_select_points(
        scene_pcd, scene_pcd_path, args.voxel_size, args.max_points
    )
    if args.visualize_solver_pcd:
        visualization_pcd = scene_pcd
    print(f"scene points passed to sceneid: {points.shape[1]}")

    if args.skip_sceneid_solve:
        sceneid_to_stone = {}
        initial_poses = {}
        solution_poses = {}
        for body_id, (stone_id, pose) in enumerate(planned_entries, start=1):
            sceneid_to_stone[body_id] = stone_id
            initial_poses[body_id] = pose.copy()
            solution_poses[body_id] = logs_sceneid.PoseVector(pose.copy())
        optimal_poses_by_stone = {
            stone_id: pose.copy() for stone_id, pose in planned_entries
        }
        elapsed = 0.0
        print("skipped diffsimpy sceneid solve; using initialized poses as output")
    else:
        context, config = get_sceneid(
            ground_height=(
                None if args.no_ground_height_init else calibrated_ground_height
            )
        )
        logs_sceneid.apply_sceneid_args_to_config(config, args)
        logs_sceneid.set_sceneid_ground_height(
            config,
            None if args.no_ground_height_init else calibrated_ground_height,
        )
        sceneid_to_stone, initial_poses = add_planned_bodies(
            context, planned_entries, stone_configs, stone_pcds
        )

        started = time.time()
        solution = context.solve(points)
        elapsed = time.time() - started
        solution_poses = solution.optimal_poses
        optimal_poses_by_stone = logs_sceneid.pose_map_from_solution(
            solution,
            sceneid_to_stone,
        )

    solution_poses, optimal_poses_by_stone = logs_sceneid.fix_locked_solution_poses(
        sceneid_to_stone,
        initial_poses,
        initial_pose_sources,
        solution_poses,
        optimal_poses_by_stone,
    )

    print(f"solve elapsed: {elapsed:.3f}s")
    print("optimal poses:")
    for body_id, pose in sorted(solution_poses.items()):
        stone_id = sceneid_to_stone.get(int(body_id), "?")
        print(
            f"  body {int(body_id):>3} stone {stone_id:>3}: "
            f"{np.asarray(pose.vectorized())}"
        )

    if args.optimal_pose_output is not None:
        save_optimal_pose_output(
            args.optimal_pose_output,
            plan_dir,
            action_path,
            scene_pcd_path,
            step_index,
            sceneid_to_stone,
            initial_poses,
            initial_pose_sources,
            optimal_poses_by_stone,
            None if args.no_ground_height_init else calibrated_ground_height,
            None if args.no_ground_height_init else calibrated_ground_plane_model,
            args.skip_sceneid_solve,
            elapsed,
        )

    if args.reconstruct_state:
        state_output = (
            args.state_output.resolve()
            if args.state_output is not None
            else action_path.parent / DEFAULT_RECONSTRUCTED_STATE_NAME
        )
        write_reconstructed_state_for_generate_sequence(
            state_output,
            plan_dir,
            state_path,
            scene_pcd_path,
            step_index,
            step_sequence,
            optimal_poses_by_stone,
            None if args.no_ground_height_init else calibrated_ground_height,
            None if args.no_ground_height_init else calibrated_ground_plane_model,
            args,
        )

    if args.visualize:
        visualize_result(
            visualization_pcd,
            visual_meshes,
            sceneid_to_stone,
            initial_poses,
            solution_poses,
            None if args.no_ground_height_init else calibrated_ground_plane_model,
        )


def parse_args() -> argparse.Namespace:
    logs_sceneid = _logs_sceneid()
    parser = argparse.ArgumentParser(
        description="Skeleton harness for testing diffsim sceneid on a planned stack."
    )
    parser.add_argument("--scene-pcd", type=Path, default=DEFAULT_SCENE_PCD)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=None,
        help=(
            "Plan directory shortcut. When --action-sequence is omitted, this "
            "uses action_sequence.pkl in the directory. When --state-path is "
            f"omitted, this prefers {DEFAULT_RECONSTRUCTED_STATE_NAME}, then "
            f"{DEFAULT_PLAN_STATE_NAME}."
        ),
    )
    parser.add_argument(
        "--action-sequence",
        type=Path,
        default=DEFAULT_ACTION_SEQUENCE,
        help="Path to action_sequence.pkl.",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=None,
        help=(
            "Use only the first N planned placements from action_sequence.pkl. "
            "Defaults to the full sequence."
        ),
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help=(
            "Optional state.pkl to use for placed-stone order and initial poses. "
            "When omitted, initialization uses action_sequence.pkl."
        ),
    )
    parser.add_argument(
        "--compare-state-pcd",
        type=Path,
        default=None,
        help=(
            "Compare a reconstructed state pickle against --scene-pcd and exit. "
            "State poses are shifted by the plan target_structure_offset."
        ),
    )
    parser.add_argument(
        "--motion-log-extra-steps",
        type=Path,
        default=None,
        help=(
            "Use feasible entries from a motion_log.txt to append missing placed "
            "stone ids beyond the saved state/action sequence. The log provides "
            "only place_world_y, so added poses are rough seeds for manual GUI."
        ),
    )
    parser.add_argument(
        "--asset-dir",
        default=None,
        help="Stone asset directory. Defaults to the plan config.yml, then assets/stone.",
    )
    parser.add_argument(
        "--no-offset",
        action="store_true",
        help="Do not apply planning_params['target_structure_offset'] to planned xy poses.",
    )
    parser.add_argument(
        "--apply-target-offset",
        action="store_true",
        help="Compatibility flag; target offset is applied by default.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling the scene PCD before solve.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Randomly cap scene points after voxel downsampling. Use 0 to disable.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.5,
        help="Meters to expand the expected posed-stone bounds before cropping.",
    )
    parser.add_argument(
        "--crop-max-margin",
        type=float,
        default=3.0,
        help="Maximum crop margin to try if the initial crop contains no points.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Use the full scene PCD instead of cropping near expected stone poses.",
    )
    parser.add_argument(
        "--no-ground-removal",
        action="store_true",
        help="Do not remove a horizontal ground plane from the scene PCD.",
    )
    parser.add_argument(
        "--ground-distance-threshold",
        type=float,
        default=0.08,
        help="RANSAC plane inlier distance threshold for ground removal.",
    )
    parser.add_argument(
        "--ground-ransac-n",
        type=int,
        default=3,
        help="Number of points sampled per RANSAC plane proposal.",
    )
    parser.add_argument(
        "--ground-num-iterations",
        type=int,
        default=1000,
        help="Number of RANSAC iterations for ground plane segmentation.",
    )
    parser.add_argument(
        "--ground-normal-z-min",
        type=float,
        default=0.7,
        help="Minimum absolute z component for accepting the plane as ground.",
    )
    parser.add_argument(
        "--ground-min-inlier-ratio",
        type=float,
        default=0.02,
        help="Minimum RANSAC inlier ratio for removing the detected ground plane.",
    )
    parser.add_argument(
        "--no-ground-height-init",
        action="store_true",
        help=(
            "Do not estimate a ground-plane z offset. By default plan-seeded "
            "poses get this z offset before ICP/sceneid."
        ),
    )
    parser.add_argument(
        "--ground-height-distance-threshold",
        type=float,
        default=0.04,
        help="RANSAC inlier distance threshold for estimating ground height.",
    )
    parser.add_argument(
        "--ground-height-max-abs",
        type=float,
        default=1.0,
        help="Reject calibrated ground heights whose absolute value exceeds this.",
    )
    parser.add_argument(
        "--no-ground-height-crop",
        action="store_true",
        help="Estimate ground height from the full scene PCD instead of the stone neighborhood.",
    )
    parser.add_argument("--ground-height-crop-margin", type=float, default=0.75)
    parser.add_argument("--ground-height-crop-max-margin", type=float, default=3.0)
    parser.add_argument(
        "--manual-init-poses",
        type=Path,
        default=None,
        help=(
            "JSON or pickle file with step-indexed manual initial poses in scene "
            "coordinates. The format matches sceneid_from_logs.py."
        ),
    )
    parser.add_argument(
        "--manual-init-step-pose",
        action="append",
        default=[],
        help=(
            "Manual initial pose for one planned step, e.g. "
            "3:x,y,z,qx,qy,qz,qw. Can be repeated."
        ),
    )
    parser.add_argument(
        "--manual-init-fixed",
        action="store_true",
        help=(
            "Treat manual initial poses as locked outputs: skip ICP for those "
            "stones and clamp sceneid results back to the manual pose."
        ),
    )
    parser.add_argument(
        "--fix-initial-steps",
        default="",
        help=(
            "Comma-separated placed step indices to lock at their initialized "
            "poses before ICP/sceneid."
        ),
    )
    parser.add_argument(
        "--manual-fix-gui",
        action="store_true",
        help=(
            "Open an Open3D selector to choose which placed steps should be "
            "locked at their initialized poses before ICP/sceneid."
        ),
    )
    parser.add_argument(
        "--manual-init-gui",
        action="store_true",
        help=(
            "Open an Open3D GUI to manually nudge initial stone poses before "
            "crop, ICP, and sceneid."
        ),
    )
    parser.add_argument(
        "--manual-init-gui-output",
        type=Path,
        default=None,
        help=(
            "JSON file to write edited manual GUI poses so they can be reused "
            "later with --manual-init-poses. Defaults to "
            "manual_init_step<N>.json in the plan directory."
        ),
    )
    parser.add_argument(
        "--optimal-pose-output",
        type=Path,
        default=None,
        help=(
            "Optional path to save final optimal poses. Use .json or .pkl for "
            "one file, or a suffix-free prefix to write .pkl and .txt like "
            "sceneid_from_logs.py."
        ),
    )
    parser.add_argument(
        "--reconstruct-state",
        action="store_true",
        help=(
            "Write a generate_sequence.py resume-ready state pickle from the "
            "identified optimal poses."
        ),
    )
    parser.add_argument(
        "--state-output",
        type=Path,
        default=None,
        help=(
            "Output path for --reconstruct-state. Defaults to "
            f"{DEFAULT_RECONSTRUCTED_STATE_NAME} in the action sequence directory."
        ),
    )
    parser.add_argument(
        "--no-mark-pose-identified",
        action="store_true",
        help=(
            "Do not label reconstructed placed stones as scene-pose-identified "
            "in the generated resume State."
        ),
    )
    parser.add_argument(
        "--manual-init-gui-translation-step",
        type=float,
        default=0.02,
        help="Translation nudge size in meters for --manual-init-gui.",
    )
    parser.add_argument(
        "--manual-init-gui-rotation-step-deg",
        type=float,
        default=5.0,
        help="Rotation nudge size in degrees for --manual-init-gui.",
    )
    parser.add_argument(
        "--manual-init-gui-point-size",
        type=float,
        default=2.0,
        help="Scene PCD point size for --manual-init-gui.",
    )
    parser.add_argument(
        "--no-icp-init",
        action="store_true",
        help="Skip Open3D ICP initialization before sceneid.",
    )
    parser.add_argument(
        "--skip-sceneid-solve",
        action="store_true",
        help="Stop after ground/ICP initialization and show those poses as output.",
    )
    parser.add_argument(
        "--icp-plan-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ICP only for poses initialized from the plan.",
    )
    parser.add_argument(
        "--icp-axes",
        default="",
        help=(
            "Axes to vary for orientation hypotheses before ICP, e.g. z or xyz. "
            "Defaults to empty, which uses only the given initial orientation."
        ),
    )
    parser.add_argument(
        "--icp-angle-step-deg",
        type=float,
        default=90.0,
        help="Angle step for ICP orientation hypotheses when --icp-axes is set.",
    )
    parser.add_argument("--icp-crop-margin", type=float, default=0.35)
    parser.add_argument("--icp-source-voxel-size", type=float, default=0.01)
    parser.add_argument("--icp-mesh-points", type=int, default=20000)
    parser.add_argument(
        "--icp-voxel-sizes",
        default="0.05,0.02,0.01",
        help="Comma-separated voxel sizes for multiscale ICP.",
    )
    parser.add_argument(
        "--icp-max-iters",
        default="30,14,7",
        help="Comma-separated max iterations matching --icp-voxel-sizes.",
    )
    parser.add_argument(
        "--icp-correspondence-distance-scale",
        type=float,
        default=1.5,
        help=(
            "Max correspondence distance multiplier per ICP scale. The actual "
            "distance is this value times each --icp-voxel-sizes entry."
        ),
    )
    parser.add_argument("--icp-min-target-points", type=int, default=5000)
    parser.add_argument("--icp-min-correspondences", type=int, default=5000)
    parser.add_argument("--icp-min-fitness", type=float, default=0.2)
    parser.add_argument(
        "--icp-min-score-gain",
        type=float,
        default=0.0,
        help=(
            "Require the best ICP hypothesis score to improve over the initial "
            "pose score by at least this amount."
        ),
    )
    parser.add_argument(
        "--icp-translation-weight",
        type=float,
        default=1.0,
        help="Penalty per meter of ICP translation away from the initial pose.",
    )
    parser.add_argument(
        "--icp-rotation-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty for ICP rotation away from the initial pose, scaled by "
            "degrees / 180."
        ),
    )
    parser.add_argument("--icp-max-translation", type=float, default=0.6)
    parser.add_argument("--icp-rmse-weight", type=float, default=0.25)
    parser.add_argument(
        "--no-icp-empty-target-prior-fallback",
        action="store_true",
        help="Compatibility flag; no prior fallback is available in this plan harness.",
    )
    parser.add_argument(
        "--no-icp-low-correspondence-prior-fallback",
        action="store_true",
        help="Compatibility flag; no prior fallback is available in this plan harness.",
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Print solver progress every N iterations (0 = silent).",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=0,
        help="OpenMP thread count (0 = use OMP_NUM_THREADS / system default).",
    )
    parser.add_argument("--graph-max-iter", type=int, default=100)
    parser.add_argument("--trust-region-eps", type=float, default=0.1)
    parser.add_argument("--delta-init", type=float, default=0.125)
    parser.add_argument("--k-pcd", type=float, default=1.0)
    parser.add_argument("--pcd-huber-delta", type=float, default=0.02)
    parser.add_argument("--pcd-max-gap", type=float, default=0.15)
    parser.add_argument("--k-gap-c", type=float, default=80.0)
    parser.add_argument("--k-comp", type=float, default=0.0)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Open Open3D viewers for PCD filtering, post-ICP poses, and final "
            "identified poses. Use --no-visualize for batch runs."
        ),
    )
    parser.add_argument(
        "--visualize-solver-pcd",
        action="store_true",
        help="Visualize the downsampled solver PCD instead of the cropped filtered PCD.",
    )
    args = parser.parse_args()

    if args.max_points == 0:
        args.max_points = None
    args.icp_voxel_sizes = [
        float(value) for value in str(args.icp_voxel_sizes).split(",") if value.strip()
    ]
    args.icp_max_iters = [
        int(value) for value in str(args.icp_max_iters).split(",") if value.strip()
    ]
    if len(args.icp_voxel_sizes) != len(args.icp_max_iters):
        raise ValueError("--icp-voxel-sizes and --icp-max-iters must have same length")
    if args.icp_correspondence_distance_scale <= 0.0:
        raise ValueError("--icp-correspondence-distance-scale must be > 0")
    if args.icp_translation_weight < 0.0:
        raise ValueError("--icp-translation-weight must be >= 0")
    if args.icp_rotation_weight < 0.0:
        raise ValueError("--icp-rotation-weight must be >= 0")
    args.icp_prior_poses = False
    args.icp_prior_min_correspondences = None
    args.icp_prior_min_fitness = None
    args.fix_initial_steps = logs_sceneid.parse_int_list(args.fix_initial_steps)
    args.manual_init_by_step = logs_sceneid.collect_manual_init_poses(args)
    if args.manual_init_gui_translation_step <= 0.0:
        raise ValueError("--manual-init-gui-translation-step must be > 0")
    if args.manual_init_gui_rotation_step_deg <= 0.0:
        raise ValueError("--manual-init-gui-rotation-step-deg must be > 0")
    if args.manual_init_gui_point_size <= 0.0:
        raise ValueError("--manual-init-gui-point-size must be > 0")
    return args


if __name__ == "__main__":
    run_sceneid_from_plan(parse_args())
