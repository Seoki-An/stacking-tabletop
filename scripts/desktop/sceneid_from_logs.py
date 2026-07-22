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
from scipy.spatial.transform import Rotation

ROOT_DIR = Path(__file__).resolve().parents[2]
DIFFSIM_PY_BUILD = ROOT_DIR.parent / "diffsim" / "interop" / "python" / "build"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if DIFFSIM_PY_BUILD.exists() and str(DIFFSIM_PY_BUILD) not in sys.path:
    sys.path.insert(0, str(DIFFSIM_PY_BUILD))

from scripts.test.merge_scene_scan_pcds import (
    load_base_frame_pcds,
    load_raw_transformed_pcds,
    merge_pcds,
    natural_key,
)
from scripts.desktop.sceneid_from_plan import (
    add_planned_bodies,
    asset_dir_for_action_sequence,
    crop_scene_pcd_near_expected_poses,
    downsample_and_select_points,
    load_action_sequence,
    make_sceneid_config,
    planned_pose_entries,
    remove_ground_plane_ransac,
    select_visual_meshes,
    target_offset_for_action_sequence,
    visualize_result,
)
from diffsimpy import sceneid
from agent.env.components.contexts import get_sceneid
from agent.env.components.action import Action
from agent.env.components.state import State, StoneTrajectory
from model import get_stone_model
from perception import multiscale_icp
from utils import pose_to_transformation_matrix

DEFAULT_ROOT = ROOT_DIR / "logs" / "260609"
DEFAULT_RECONSTRUCTED_STATE_NAME = "state_sceneid.pkl"
DEFAULT_PLAN_STATE_NAME = "state.pkl"


class PoseVector:
    def __init__(self, pose: np.ndarray):
        self._pose = np.asarray(pose, dtype=np.float64).copy()

    def vectorized(self) -> np.ndarray:
        return self._pose


def scene_scan_dirs(root: Path) -> list[Path]:
    if root.name == "scene_scan":
        return [root]
    if (root / "scene_scan").is_dir():
        return [root / "scene_scan"]
    return sorted([p for p in root.rglob("scene_scan") if p.is_dir()], key=natural_key)


def step_index_from_scene_dir(scene_dir: Path) -> int:
    step_dir = scene_dir.parent if scene_dir.name == "scene_scan" else scene_dir
    match = re.fullmatch(r"step_(\d+)", step_dir.name)
    if not match:
        raise ValueError(f"cannot infer step index from {scene_dir}")
    return int(match.group(1))


def plan_id_from_scene_dir(scene_dir: Path) -> str:
    for parent in scene_dir.parents:
        if parent.name.startswith("plan_"):
            return parent.name.split(".", 1)[0]
    raise ValueError(f"cannot infer plan id from {scene_dir}")


def resolve_plan_dir(scene_dir: Path, plan_dir_arg: Path | None) -> Path:
    if plan_dir_arg is not None:
        plan_dir = plan_dir_arg.resolve()
        parts = plan_dir.name.split("_")
        plan_date = parts[1] if len(parts) >= 3 and parts[1].isdigit() else None
        if plan_date and not (plan_dir / "action_sequence.pkl").exists():
            dated_plan_dir = plan_dir.parent / plan_date / plan_dir.name
            if (dated_plan_dir / "action_sequence.pkl").exists():
                return dated_plan_dir
        if not (plan_dir / "action_sequence.pkl").exists():
            raise FileNotFoundError(f"action_sequence.pkl not found in {plan_dir}")
        return plan_dir

    plan_id = plan_id_from_scene_dir(scene_dir)
    parts = plan_id.split("_")
    plan_date = parts[1] if len(parts) >= 3 and parts[1].isdigit() else None
    candidates = [
        ROOT_DIR / "sessions" / plan_date / plan_id if plan_date else None,
        ROOT_DIR / "sessions" / plan_id,
        (
            ROOT_DIR / "scripts" / "desktop" / "sessions" / plan_date / plan_id
            if plan_date
            else None
        ),
        ROOT_DIR / "scripts" / "desktop" / "sessions" / plan_id,
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    for candidate in candidates:
        if (candidate / "action_sequence.pkl").exists():
            return candidate
    raise FileNotFoundError(
        "Plan directory not found for "
        f"{scene_dir}. Tried:\n  " + "\n  ".join(str(p) for p in candidates)
    )


def read_or_merge_scene_pcd(
    scene_dir: Path,
    remerge: bool,
    raw: bool,
    voxel_size: float,
    merged_name: str,
) -> tuple[o3d.geometry.PointCloud, Path | None]:
    existing = scene_dir / "scene_scan_merged.ply"
    if existing.exists() and not remerge:
        pcd = o3d.io.read_point_cloud(str(existing))
        if pcd.is_empty():
            raise ValueError(f"empty merged scene PCD: {existing}")
        return pcd, existing

    pcds = [] if raw else load_base_frame_pcds(scene_dir)
    if not pcds:
        pcds = load_raw_transformed_pcds(scene_dir)
    if not pcds:
        raise FileNotFoundError(f"no scene scan PCDs found in {scene_dir}")

    merged = merge_pcds(pcds, voxel_size)
    if merged.is_empty():
        raise ValueError(f"merged scene PCD is empty: {scene_dir}")

    output_path = scene_dir / merged_name
    if not o3d.io.write_point_cloud(str(output_path), merged):
        raise RuntimeError(f"failed to write merged scene PCD: {output_path}")
    return merged, output_path


def color_for_index(index: int) -> np.ndarray:
    palette = np.asarray(
        [
            [0.90, 0.12, 0.12],
            [0.12, 0.35, 1.00],
            [0.10, 0.70, 0.25],
            [1.00, 0.55, 0.05],
            [0.60, 0.20, 0.90],
            [0.00, 0.70, 0.80],
            [0.95, 0.75, 0.05],
            [0.85, 0.10, 0.50],
            [0.45, 0.45, 0.45],
            [0.25, 0.75, 0.65],
        ],
        dtype=np.float64,
    )
    return palette[index % len(palette)].copy()


def comparison_label(scene_dir: Path) -> str:
    step_dir = scene_dir.parent if scene_dir.name == "scene_scan" else scene_dir
    return f"{exec_dir_from_scene_dir(scene_dir).name}/{step_dir.name}"


def pcd_bbox_lineset(
    pcd: o3d.geometry.PointCloud,
    color: np.ndarray,
) -> o3d.geometry.LineSet | None:
    if pcd.is_empty():
        return None
    bbox = pcd.get_axis_aligned_bounding_box()
    lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    lineset.paint_uniform_color(color)
    return lineset


def sample_pcd_for_comparison(
    pcd: o3d.geometry.PointCloud,
    max_points: int,
    seed: int,
) -> o3d.geometry.PointCloud:
    n_points = len(pcd.points)
    if max_points <= 0 or n_points <= max_points:
        return o3d.geometry.PointCloud(pcd)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_points, size=max_points, replace=False))
    return pcd.select_by_index(indices.tolist())


def visualize_pcd_comparison(geoms: list, point_size: float) -> None:
    vis = o3d.visualization.Visualizer()
    created = vis.create_window(window_name="scene scan PCD comparison")
    if not created:
        print("  Open3D viewer unavailable; skipping comparison window.")
        vis.destroy_window()
        return

    for geom in geoms:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = point_size
        render_option.mesh_show_back_face = True
        render_option.light_on = True

    vis.run()
    vis.destroy_window()


def compare_scene_pcds(
    scene_dirs: list[Path],
    args: argparse.Namespace,
) -> int:
    requested_steps = getattr(args, "compare_pcd_step_indices", None)
    selected_dirs = []
    for scene_dir in scene_dirs:
        step_index = step_index_from_scene_dir(scene_dir)
        if requested_steps is not None and step_index not in requested_steps:
            continue
        selected_dirs.append(scene_dir)

    if not selected_dirs:
        requested = "all" if requested_steps is None else sorted(requested_steps)
        raise ValueError(
            f"no scene_scan directories matched compare steps: {requested}"
        )

    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.75)]
    combined = o3d.geometry.PointCloud()
    print("\nScene PCD comparison:")
    for idx, scene_dir in enumerate(selected_dirs):
        step_index = step_index_from_scene_dir(scene_dir)
        pcd, scene_pcd_path = read_or_merge_scene_pcd(
            scene_dir,
            args.remerge,
            args.raw,
            args.merge_voxel_size,
            args.merged_name,
        )
        if args.compare_pcd_ground_removal:
            pcd = remove_ground_plane_ransac(
                pcd,
                args.ground_distance_threshold,
                args.ground_ransac_n,
                args.ground_num_iterations,
                args.ground_normal_z_min,
                args.ground_min_inlier_ratio,
            )

        color = color_for_index(idx)
        display_pcd = sample_pcd_for_comparison(
            pcd,
            args.compare_pcd_max_points,
            seed=1009 + 7919 * idx + step_index,
        )
        display_pcd.paint_uniform_color(color)
        geoms.append(display_pcd)
        combined += display_pcd

        if args.compare_pcd_bboxes:
            bbox = pcd_bbox_lineset(display_pcd, color)
            if bbox is not None:
                geoms.append(bbox)

        print(
            f"  {idx + 1}: {comparison_label(scene_dir)} "
            f"color={np.round(color, 3).tolist()} "
            f"points={len(display_pcd.points)}"
        )
        print(f"     source={scene_pcd_path}")
        if display_pcd.is_empty():
            print("     bounds=empty")
        else:
            bbox = display_pcd.get_axis_aligned_bounding_box()
            print(
                f"     min={np.round(np.asarray(bbox.min_bound), 4)}, "
                f"max={np.round(np.asarray(bbox.max_bound), 4)}"
            )

    if args.compare_pcd_output is not None:
        output_path = args.compare_pcd_output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(output_path), combined):
            raise RuntimeError(f"failed to write comparison PCD: {output_path}")
        print(f"  wrote colored comparison PCD: {output_path}")

    if args.compare_pcd_view:
        visualize_pcd_comparison(geoms, args.compare_pcd_point_size)
    return len(selected_dirs)


def pose_map_from_solution(
    solution, sceneid_to_stone: dict[int, int]
) -> dict[int, np.ndarray]:
    poses = {}
    for body_id, pose in solution.optimal_poses.items():
        stone_id = sceneid_to_stone[int(body_id)]
        poses[stone_id] = np.asarray(pose.vectorized(), dtype=np.float64).copy()
    return poses


def exec_dir_from_scene_dir(scene_dir: Path) -> Path:
    step_dir = scene_dir.parent if scene_dir.name == "scene_scan" else scene_dir
    return step_dir.parent


def recursive_cache_key(scene_dir: Path, args: argparse.Namespace) -> Path:
    if getattr(args, "recursive_cache_scope", "plan") == "exec":
        return exec_dir_from_scene_dir(scene_dir)
    return resolve_plan_dir(scene_dir, args.plan_dir)


def apply_recursive_initial_poses(
    planned_entries: list[tuple[int, np.ndarray]],
    prior_poses_by_stone: dict[int, np.ndarray],
    enabled: bool,
    base_sources: dict[int, str] | None = None,
) -> tuple[list[tuple[int, np.ndarray]], dict[int, str]]:
    entries = []
    sources = {}
    base_sources = base_sources or {}
    for stone_id, planned_pose in planned_entries:
        if enabled and stone_id in prior_poses_by_stone:
            entries.append((stone_id, prior_poses_by_stone[stone_id].copy()))
            sources[stone_id] = "prior"
        else:
            entries.append((stone_id, planned_pose.copy()))
            sources[stone_id] = base_sources.get(stone_id, "plan")
    return entries, sources


def plan_state_candidates(
    plan_dir: Path,
    step_index: int,
    state_path: Path | None = None,
) -> list:
    state_path = state_path or plan_dir / DEFAULT_PLAN_STATE_NAME
    if not state_path.exists():
        return []

    try:
        with state_path.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        print(f"  warning: could not read plan state {state_path}: {exc}")
        return []

    candidates = []
    if isinstance(data, dict):
        for step in data.get("steps", []):
            if not isinstance(step, dict) or int(step.get("step", -1)) != step_index:
                continue
            candidates.extend(
                [step.get("resume_state", None), step.get("raw_state", None)]
            )
        candidates.append(data.get("resume_state", None))
        for step in reversed(data.get("steps", [])):
            if isinstance(step, dict):
                candidates.extend(
                    [step.get("resume_state", None), step.get("raw_state", None)]
                )
    return [candidate for candidate in candidates if candidate is not None]


def state_label(state_path: Path | None = None) -> str:
    return state_path.name if state_path is not None else DEFAULT_PLAN_STATE_NAME


def state_stone_ids(state, step_index: int | None = None) -> list[int]:
    stone_set = getattr(state, "stone_set", None)
    stone_seq = getattr(state, "stone_seq", None)
    if stone_set is None or stone_seq is None:
        return []

    stone_ids = []
    seq = stone_seq if step_index is None else stone_seq[:step_index]
    for stone_idx in seq:
        try:
            stone_ids.append(int(stone_set[int(stone_idx)]))
        except (TypeError, ValueError, IndexError):
            continue
    return stone_ids


def max_state_step_index(
    plan_dir: Path,
    fallback_step_index: int,
    state_path: Path | None = None,
) -> int:
    max_step = 0
    for state in plan_state_candidates(plan_dir, fallback_step_index, state_path):
        max_step = max(max_step, len(state_stone_ids(state, None)))
    return max_step or fallback_step_index


def state_backed_step_sequence(
    plan_dir: Path,
    action_sequence: list,
    step_index: int,
    state_path: Path | None = None,
) -> list[int]:
    action_stone_ids = [
        action_stone_id(action) for action in action_sequence[:step_index]
    ]
    for state in plan_state_candidates(plan_dir, step_index, state_path):
        state_ids = state_stone_ids(state, step_index)
        if not state_ids:
            continue
        merged_ids = list(state_ids[:step_index])
        for stone_id in action_stone_ids:
            if len(merged_ids) >= step_index:
                break
            if stone_id not in merged_ids:
                merged_ids.append(stone_id)
        if len(merged_ids) >= step_index:
            print(
                f"  step order: using {state_label(state_path)} for "
                f"{min(len(state_ids), step_index)} / {step_index} placed stones"
            )
            return merged_ids
    return action_stone_ids


def state_pose_entries_from_plan_state(
    plan_dir: Path,
    action_sequence: list,
    step_index: int,
    target_offset: np.ndarray,
    fallback_entries: list[tuple[int, np.ndarray]],
    state_path: Path | None = None,
) -> tuple[list[tuple[int, np.ndarray]], dict[int, str]]:
    candidates = plan_state_candidates(plan_dir, step_index, state_path)
    if not candidates:
        return fallback_entries, {stone_id: "plan" for stone_id, _ in fallback_entries}

    action_stone_ids = [
        action_stone_id(action) for action in action_sequence[:step_index]
    ]
    action_stone_set = set(action_stone_ids)
    state_poses = {}
    for state in candidates:
        stone_set = getattr(state, "stone_set", None)
        stone_seq = getattr(state, "stone_seq", None)
        stone_poses = getattr(state, "stone_poses", None)
        if stone_set is None or stone_seq is None or stone_poses is None:
            continue
        for stone_idx in stone_seq:
            try:
                stone_id = int(stone_set[int(stone_idx)])
            except (TypeError, ValueError, IndexError):
                continue
            if stone_id not in action_stone_set or stone_id not in stone_poses:
                continue
            pose = np.asarray(stone_poses[stone_id], dtype=np.float64).copy()
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                continue
            pose[:2] += np.asarray(target_offset, dtype=np.float64)[:2]
            state_poses[stone_id] = pose
        if all(stone_id in state_poses for stone_id in action_stone_ids):
            break

    entries = []
    sources = {}
    fallback_by_stone = {stone_id: pose for stone_id, pose in fallback_entries}
    for stone_id in action_stone_ids:
        if stone_id in state_poses:
            entries.append((stone_id, state_poses[stone_id].copy()))
            sources[stone_id] = "plan_state"
        elif stone_id in fallback_by_stone:
            entries.append((stone_id, fallback_by_stone[stone_id].copy()))
            sources[stone_id] = "plan"
        else:
            print(f"  warning: no initial pose found for stone {stone_id}; skipping")

    n_state = sum(1 for source in sources.values() if source == "plan_state")
    if n_state:
        print(
            f"  init poses: using {state_label(state_path)} for "
            f"{n_state} / {len(entries)} stones"
        )
    return entries, sources


def missing_prior_step_indices(scene_dir: Path, step_index: int) -> list[int]:
    if step_index <= 1:
        return []
    exec_dir = exec_dir_from_scene_dir(scene_dir)
    available_steps = {
        step_index_from_scene_dir(path) for path in scene_scan_dirs(exec_dir)
    }
    return [step for step in range(1, step_index) if step not in available_steps]


def mark_missing_prior_step_sources(
    initial_pose_sources: dict[int, str],
    action_sequence: list,
    missing_steps: list[int],
) -> list[int]:
    fixed_stone_ids = []
    for step in missing_steps:
        if step <= 0 or step > len(action_sequence):
            continue
        stone_id = action_stone_id(action_sequence[step - 1])
        source = initial_pose_sources.get(stone_id)
        if source is None or source.startswith("prior"):
            continue
        if "fixed_missing_scene" not in source:
            initial_pose_sources[stone_id] = f"{source}+fixed_missing_scene"
        fixed_stone_ids.append(stone_id)
    return fixed_stone_ids


def mark_user_fixed_step_sources(
    initial_pose_sources: dict[int, str],
    action_sequence: list,
    step_index: int,
    fixed_steps: list[int],
) -> list[tuple[int, int]]:
    fixed = []
    for step in sorted(set(fixed_steps)):
        if step <= 0 or step > step_index or step > len(action_sequence):
            continue
        stone_id = action_stone_id(action_sequence[step - 1])
        source = initial_pose_sources.get(stone_id)
        if source is None or "user_fixed" in source:
            continue
        initial_pose_sources[stone_id] = f"{source}+user_fixed"
        fixed.append((step, stone_id))
    return fixed


def prior_poses_before_step(
    prior_poses_by_stone: dict[int, np.ndarray],
    action_sequence: list,
    step_index: int,
) -> dict[int, np.ndarray]:
    prior_stone_ids = {
        action_stone_id(action) for action in action_sequence[: max(step_index - 1, 0)]
    }
    return {
        int(stone_id): pose
        for stone_id, pose in prior_poses_by_stone.items()
        if int(stone_id) in prior_stone_ids
    }


def plan_log_root_from_scene_dir(scene_dir: Path) -> Path:
    for parent in scene_dir.parents:
        if parent.name.startswith("plan_"):
            return parent
    return exec_dir_from_scene_dir(scene_dir)


def latest_previous_identified_poses_from_logs(
    scene_dir: Path,
    output_prefix: str,
    action_sequence: list,
    step_index: int,
) -> dict[int, np.ndarray]:
    previous_stone_ids = {
        action_stone_id(action) for action in action_sequence[: max(step_index - 1, 0)]
    }
    if not previous_stone_ids:
        return {}

    records = []
    for candidate_dir in scene_scan_dirs(plan_log_root_from_scene_dir(scene_dir)):
        try:
            candidate_step = step_index_from_scene_dir(candidate_dir)
        except ValueError:
            continue
        if candidate_step >= step_index:
            continue
        record_path = candidate_dir / f"{output_prefix}.pkl"
        if record_path.exists():
            records.append((candidate_step, natural_key(candidate_dir), record_path))

    poses: dict[int, np.ndarray] = {}
    for _, _, record_path in sorted(records):
        try:
            with record_path.open("rb") as f:
                record = pickle.load(f)
        except Exception as exc:
            print(
                f"  warning: could not read previous sceneid record {record_path}: {exc}"
            )
            continue
        if not isinstance(record, dict):
            continue
        for stone_id, pose in record.get("optimal_poses", {}).items():
            stone_id = int(stone_id)
            if stone_id not in previous_stone_ids:
                continue
            try:
                poses[stone_id] = normalize_pose_quaternion(pose)
            except Exception as exc:
                print(
                    "  warning: ignoring previous sceneid pose "
                    f"{record_path} stone {stone_id}: {exc}"
                )
    return poses


def parse_int_list(value: str | list[int] | tuple[int, ...] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_step_key(value) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    match = re.fullmatch(r"(?:step[_-]?)?(\d+)", text)
    if match is None:
        raise ValueError(f"invalid step key for manual init pose: {value!r}")
    return int(match.group(1))


def normalize_pose_quaternion(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1).copy()
    if pose.shape != (7,):
        raise ValueError(f"pose must have 7 values [x,y,z,qx,qy,qz,qw], got {pose}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"pose contains non-finite values: {pose}")
    quat_norm = float(np.linalg.norm(pose[3:]))
    if quat_norm <= 1e-12:
        raise ValueError(f"pose quaternion is degenerate: {pose}")
    pose[3:] /= quat_norm
    return pose


def parse_pose_vector(value) -> np.ndarray:
    if isinstance(value, dict):
        if "pose" in value:
            return parse_pose_vector(value["pose"])
        position = value.get("position", value.get("xyz", None))
        quaternion = value.get("quaternion", value.get("quat", None))
        if position is not None and quaternion is not None:
            return normalize_pose_quaternion([*position, *quaternion])
    if isinstance(value, np.ndarray):
        return normalize_pose_quaternion(value)
    if isinstance(value, (list, tuple)):
        return normalize_pose_quaternion(value)

    text = str(value).strip()
    for delimiter in (";", "|"):
        text = text.replace(delimiter, ",")
    parts = [part for part in re.split(r"[\s,]+", text) if part]
    return normalize_pose_quaternion([float(part) for part in parts])


def manual_init_entries_from_data(data) -> dict[int, np.ndarray]:
    if isinstance(data, dict):
        if "steps" in data:
            data = data["steps"]
        elif "manual_init_poses" in data:
            data = data["manual_init_poses"]

    entries: dict[int, np.ndarray] = {}
    if isinstance(data, dict):
        for step_key, pose_value in data.items():
            try:
                step = parse_step_key(step_key)
            except ValueError:
                continue
            entries[step] = parse_pose_vector(pose_value)
        return entries

    if isinstance(data, (list, tuple)):
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"manual init list item must be a dict: {item!r}")
            step = parse_step_key(item.get("step", item.get("step_index", None)))
            entries[step] = parse_pose_vector(item.get("pose", item))
        return entries

    raise ValueError(
        "manual init data must be a dict or a list of {'step': int, 'pose': [...]}"
    )


def load_manual_init_pose_file(path: Path) -> dict[int, np.ndarray]:
    path = path.resolve()
    with path.open("rb") as f:
        if path.suffix.lower() in (".pkl", ".pickle"):
            data = pickle.load(f)
        else:
            data = json.load(f)
    return manual_init_entries_from_data(data)


def parse_manual_step_pose(value: str) -> tuple[int, np.ndarray]:
    text = str(value).strip()
    if "=" in text:
        step_text, pose_text = text.split("=", 1)
    elif ":" in text:
        step_text, pose_text = text.split(":", 1)
    else:
        raise ValueError(
            "--manual-init-step-pose must look like " "STEP:x,y,z,qx,qy,qz,qw"
        )
    return parse_step_key(step_text), parse_pose_vector(pose_text)


def collect_manual_init_poses(args: argparse.Namespace) -> dict[int, np.ndarray]:
    entries: dict[int, np.ndarray] = {}
    if args.manual_init_poses is not None:
        entries.update(load_manual_init_pose_file(args.manual_init_poses))
    for item in args.manual_init_step_pose or []:
        step, pose = parse_manual_step_pose(item)
        entries[step] = pose
    invalid_steps = [step for step in entries if step <= 0]
    if invalid_steps:
        raise ValueError(f"manual init step indices must be positive: {invalid_steps}")
    return entries


def apply_manual_initial_poses(
    planned_entries: list[tuple[int, np.ndarray]],
    initial_pose_sources: dict[int, str],
    action_sequence: list,
    step_index: int,
    manual_init_poses: dict[int, np.ndarray],
    fixed: bool,
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, int]]]:
    if not manual_init_poses:
        return planned_entries, []

    step_by_stone = {
        action_stone_id(action_sequence[action_i]): action_i + 1
        for action_i in range(step_index)
    }
    applied = []
    updated_entries = []
    source = "manual_fixed" if fixed else "manual"
    for stone_id, pose in planned_entries:
        action_step = step_by_stone.get(stone_id)
        if action_step in manual_init_poses:
            manual_pose = manual_init_poses[action_step].copy()
            updated_entries.append((stone_id, manual_pose))
            initial_pose_sources[stone_id] = source
            applied.append((action_step, stone_id))
        else:
            updated_entries.append((stone_id, pose))
    return updated_entries, applied


def save_manual_init_pose_file(
    path: Path,
    manual_init_poses: dict[int, np.ndarray],
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "steps": {
            str(step): normalize_pose_quaternion(pose).tolist()
            for step, pose in sorted(manual_init_poses.items())
        }
    }
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"  wrote manual init poses: {path}")


def set_mesh_pose_from_base(
    mesh: o3d.geometry.TriangleMesh,
    base_vertices: np.ndarray,
    base_normals: np.ndarray,
    pose: np.ndarray,
) -> None:
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    mesh.vertices = o3d.utility.Vector3dVector(base_vertices @ rotation.T + pose[:3])
    if base_normals.size:
        mesh.vertex_normals = o3d.utility.Vector3dVector(base_normals @ rotation.T)
    else:
        mesh.compute_vertex_normals()


def print_manual_gui_help(
    translation_step: float,
    rotation_step_deg: float,
) -> None:
    print(
        "\nManual init GUI controls:\n"
        "  N/P or 1-9: select next/previous or indexed stone\n"
        "  G: jump by terminal input; enter step number or s<stone_id>, e.g. s30\n"
        "  A/D: move -/+ world X\n"
        "  S/W: move -/+ world Y\n"
        "  F/R: move -/+ world Z\n"
        "  J/L: yaw -/+ around world Z\n"
        "  I/K: pitch +/- around world Y\n"
        "  U/O: roll -/+ around world X\n"
        "  M: mark selected pose as manual without moving\n"
        "  Z: reset selected pose to the pose shown when the GUI opened\n"
        "  C or Enter: continue with the edited poses\n"
        f"  translation step={translation_step:.4f} m, "
        f"rotation step={rotation_step_deg:.2f} deg"
    )


def print_manual_gui_index_table(editable: list[tuple[int, int, np.ndarray]]) -> None:
    print("  editable stones:")
    for idx, (step, stone_id, _) in enumerate(editable, start=1):
        print(f"    {idx:>2}: step {step:>2}, stone {stone_id}")


def run_manual_initialization_gui(
    scene_pcd: o3d.geometry.PointCloud,
    planned_entries: list[tuple[int, np.ndarray]],
    initial_pose_sources: dict[int, str],
    action_sequence: list,
    step_index: int,
    visual_meshes: dict[int, o3d.geometry.TriangleMesh],
    args: argparse.Namespace,
    scene_dir: Path,
) -> tuple[list[tuple[int, np.ndarray]], dict[int, np.ndarray], list[tuple[int, int]]]:
    editable = []
    step_by_stone = {
        action_stone_id(action_sequence[action_i]): action_i + 1
        for action_i in range(step_index)
    }
    for stone_id, pose in planned_entries:
        step = step_by_stone.get(stone_id)
        if step is None or stone_id not in visual_meshes:
            continue
        editable.append((step, stone_id, pose.copy()))

    if not editable:
        print("  manual init GUI skipped: no visual meshes for placed stones")
        return planned_entries, {}, []

    poses_by_stone = {stone_id: pose.copy() for stone_id, pose in planned_entries}
    original_poses_by_stone = {
        stone_id: pose.copy() for stone_id, pose in poses_by_stone.items()
    }
    original_sources_by_stone = {
        stone_id: initial_pose_sources.get(stone_id, "plan")
        for stone_id in poses_by_stone
    }
    dirty_steps: set[int] = set()
    selected = {"idx": max(0, len(editable) - 1)}
    mesh_state = {}

    scene_vis = o3d.geometry.PointCloud(scene_pcd)
    if not scene_vis.is_empty():
        normal_radius = float(getattr(args, "manual_init_gui_normal_radius", 0.25))
        normal_max_nn = int(getattr(args, "manual_init_gui_normal_max_nn", 30))
        scene_vis.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=normal_max_nn,
            )
        )
        scene_vis.normalize_normals()
    scene_vis.paint_uniform_color([0.45, 0.45, 0.45])
    geoms = [
        scene_vis,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.75),
    ]

    for step, stone_id, pose in editable:
        mesh = copy.deepcopy(visual_meshes[stone_id])
        mesh.compute_vertex_normals()
        base_vertices = np.asarray(mesh.vertices).copy()
        base_normals = np.asarray(mesh.vertex_normals).copy()
        set_mesh_pose_from_base(mesh, base_vertices, base_normals, pose)
        mesh_state[stone_id] = {
            "mesh": mesh,
            "base_vertices": base_vertices,
            "base_normals": base_normals,
            "step": step,
        }
        geoms.append(mesh)

    def current_entry() -> tuple[int, int]:
        step, stone_id, _ = editable[selected["idx"]]
        return step, stone_id

    def current_source(stone_id: int) -> str:
        return initial_pose_sources.get(stone_id, "plan")

    def manual_source_for(stone_id: int) -> str:
        if args.manual_init_fixed or is_fixed_initial_pose_source(
            current_source(stone_id)
        ):
            return "manual_fixed"
        return "manual"

    def paint_meshes(vis=None) -> None:
        for idx, (step, stone_id, _) in enumerate(editable):
            mesh = mesh_state[stone_id]["mesh"]
            if idx == selected["idx"]:
                color = [1.0, 0.45, 0.05]
            elif step in dirty_steps:
                color = [0.10, 0.70, 0.25]
            elif is_fixed_initial_pose_source(current_source(stone_id)):
                color = [0.65, 0.20, 0.90]
            else:
                color = [0.10, 0.35, 1.00]
            mesh.paint_uniform_color(color)
            if vis is not None:
                vis.update_geometry(mesh)

    def refresh_mesh(vis, stone_id: int) -> None:
        state = mesh_state[stone_id]
        set_mesh_pose_from_base(
            state["mesh"],
            state["base_vertices"],
            state["base_normals"],
            poses_by_stone[stone_id],
        )
        vis.update_geometry(state["mesh"])
        paint_meshes(vis)
        vis.update_renderer()

    def print_current_pose(prefix: str = "selected") -> None:
        step, stone_id = current_entry()
        pose = poses_by_stone[stone_id]
        source = current_source(stone_id)
        print(
            f"  {prefix}: step {step}, stone {stone_id}, source={source}, "
            f"pose={np.round(pose, 6).tolist()}"
        )

    def select_delta(vis, delta: int) -> bool:
        selected["idx"] = (selected["idx"] + delta) % len(editable)
        paint_meshes(vis)
        vis.update_renderer()
        print_current_pose()
        return False

    def select_index(vis, idx: int) -> bool:
        if idx < len(editable):
            selected["idx"] = idx
            paint_meshes(vis)
            vis.update_renderer()
            print_current_pose()
        return False

    def select_step_or_stone(vis) -> bool:
        value = input("  jump to step or stone id (e.g. 23 or s30): ").strip()
        if not value:
            return False
        target_idx = None
        if value.lower().startswith("s"):
            try:
                target_stone = int(value[1:])
            except ValueError:
                print(f"  invalid stone id: {value}")
                return False
            for idx, (_, stone_id, _) in enumerate(editable):
                if stone_id == target_stone:
                    target_idx = idx
                    break
        else:
            try:
                target_step = int(value)
            except ValueError:
                print(f"  invalid step: {value}")
                return False
            for idx, (step, _, _) in enumerate(editable):
                if step == target_step:
                    target_idx = idx
                    break
        if target_idx is None:
            print(f"  no editable entry for {value}")
            return False
        return select_index(vis, target_idx)

    def mark_dirty(stone_id: int) -> None:
        step = mesh_state[stone_id]["step"]
        dirty_steps.add(step)
        initial_pose_sources[stone_id] = manual_source_for(stone_id)

    def move_selected(vis, delta_xyz: np.ndarray) -> bool:
        _, stone_id = current_entry()
        poses_by_stone[stone_id][:3] += delta_xyz
        mark_dirty(stone_id)
        refresh_mesh(vis, stone_id)
        print_current_pose("moved")
        return False

    def rotate_selected(vis, axis: np.ndarray, sign: float) -> bool:
        _, stone_id = current_entry()
        pose = poses_by_stone[stone_id]
        angle = np.deg2rad(args.manual_init_gui_rotation_step_deg * sign)
        delta = Rotation.from_rotvec(axis * angle)
        rotation = delta * Rotation.from_quat(pose[3:])
        pose[3:] = rotation.as_quat()
        pose[:] = normalize_pose_quaternion(pose)
        mark_dirty(stone_id)
        refresh_mesh(vis, stone_id)
        print_current_pose("rotated")
        return False

    def mark_selected(vis) -> bool:
        _, stone_id = current_entry()
        mark_dirty(stone_id)
        paint_meshes(vis)
        vis.update_renderer()
        print_current_pose("marked")
        return False

    def reset_selected(vis) -> bool:
        _, stone_id = current_entry()
        poses_by_stone[stone_id] = original_poses_by_stone[stone_id].copy()
        step = mesh_state[stone_id]["step"]
        dirty_steps.discard(step)
        initial_pose_sources[stone_id] = original_sources_by_stone[stone_id]
        refresh_mesh(vis, stone_id)
        print_current_pose("reset")
        return False

    def close_window(vis) -> bool:
        vis.close()
        return False

    paint_meshes()
    print(f"\nOpening manual init GUI for {scene_dir}")
    print_manual_gui_help(
        args.manual_init_gui_translation_step,
        args.manual_init_gui_rotation_step_deg,
    )
    print_manual_gui_index_table(editable)
    print_current_pose()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    created = vis.create_window(
        window_name=f"manual init: {comparison_label(scene_dir)}"
    )
    if not created:
        print("  Open3D viewer unavailable; manual init GUI skipped.")
        vis.destroy_window()
        return planned_entries, {}, []

    for geom in geoms:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = args.manual_init_gui_point_size
        render_option.mesh_show_back_face = True
        render_option.light_on = True

    t_step = float(args.manual_init_gui_translation_step)
    r_axis_x = np.array([1.0, 0.0, 0.0])
    r_axis_y = np.array([0.0, 1.0, 0.0])
    r_axis_z = np.array([0.0, 0.0, 1.0])
    callbacks = {
        "N": lambda vis: select_delta(vis, 1),
        "P": lambda vis: select_delta(vis, -1),
        "A": lambda vis: move_selected(vis, np.array([-t_step, 0.0, 0.0])),
        "D": lambda vis: move_selected(vis, np.array([t_step, 0.0, 0.0])),
        "S": lambda vis: move_selected(vis, np.array([0.0, -t_step, 0.0])),
        "W": lambda vis: move_selected(vis, np.array([0.0, t_step, 0.0])),
        "F": lambda vis: move_selected(vis, np.array([0.0, 0.0, -t_step])),
        "R": lambda vis: move_selected(vis, np.array([0.0, 0.0, t_step])),
        "J": lambda vis: rotate_selected(vis, r_axis_z, -1.0),
        "L": lambda vis: rotate_selected(vis, r_axis_z, 1.0),
        "I": lambda vis: rotate_selected(vis, r_axis_y, 1.0),
        "K": lambda vis: rotate_selected(vis, r_axis_y, -1.0),
        "U": lambda vis: rotate_selected(vis, r_axis_x, -1.0),
        "O": lambda vis: rotate_selected(vis, r_axis_x, 1.0),
        "M": mark_selected,
        "G": select_step_or_stone,
        "Q": lambda vis: False,
        "Z": reset_selected,
        "C": close_window,
    }
    for key, callback in callbacks.items():
        vis.register_key_callback(ord(key), callback)
    vis.register_key_callback(13, close_window)
    for idx in range(min(9, len(editable))):
        vis.register_key_callback(
            ord(str(idx + 1)), lambda vis, i=idx: select_index(vis, i)
        )

    vis.run()
    vis.destroy_window()

    manual_poses_by_step = {
        step: poses_by_stone[stone_id].copy()
        for step, stone_id, _ in editable
        if step in dirty_steps
    }
    applied = []
    updated_entries = []
    for stone_id, pose in planned_entries:
        step = step_by_stone.get(stone_id)
        if step in manual_poses_by_step:
            updated_entries.append((stone_id, manual_poses_by_step[step].copy()))
            applied.append((step, stone_id))
        else:
            updated_entries.append((stone_id, pose))

    if applied:
        print(
            "  manual GUI init poses: "
            + ", ".join(
                f"step {step} -> stone {stone_id}" for step, stone_id in applied
            )
        )
    else:
        print("  manual GUI init poses: no edits")
    return updated_entries, manual_poses_by_step, applied


def print_manual_fix_gui_help() -> None:
    print(
        "\nManual fix GUI controls:\n"
        "  N/P or 1-9: select next/previous or indexed stone\n"
        "  G: jump by terminal input; enter step number or s<stone_id>, e.g. s30\n"
        "  M or Space: toggle selected step fixed/unfixed\n"
        "  C or Enter: continue with selected fixed steps"
    )


def run_manual_fix_gui(
    scene_pcd: o3d.geometry.PointCloud,
    planned_entries: list[tuple[int, np.ndarray]],
    initial_pose_sources: dict[int, str],
    action_sequence: list,
    step_index: int,
    visual_meshes: dict[int, o3d.geometry.TriangleMesh],
    args: argparse.Namespace,
    scene_dir: Path,
    initial_fixed_steps: list[int],
) -> list[int]:
    editable = []
    step_by_stone = {
        action_stone_id(action_sequence[action_i]): action_i + 1
        for action_i in range(step_index)
    }
    for stone_id, pose in planned_entries:
        step = step_by_stone.get(stone_id)
        if step is None or stone_id not in visual_meshes:
            continue
        editable.append((step, stone_id, pose.copy()))

    if not editable:
        print("  manual fix GUI skipped: no visual meshes for placed stones")
        return initial_fixed_steps

    selected_fixed_steps = {
        int(step) for step in initial_fixed_steps if 0 < int(step) <= step_index
    }
    selected = {"idx": max(0, len(editable) - 1)}
    mesh_state = {}

    scene_vis = o3d.geometry.PointCloud(scene_pcd)
    if not scene_vis.is_empty():
        scene_vis.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(getattr(args, "manual_init_gui_normal_radius", 0.25)),
                max_nn=int(getattr(args, "manual_init_gui_normal_max_nn", 30)),
            )
        )
        scene_vis.normalize_normals()
    scene_vis.paint_uniform_color([0.45, 0.45, 0.45])
    geoms = [
        scene_vis,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.75),
    ]

    for step, stone_id, pose in editable:
        mesh = copy.deepcopy(visual_meshes[stone_id])
        mesh.compute_vertex_normals()
        set_mesh_pose_from_base(
            mesh,
            np.asarray(mesh.vertices).copy(),
            np.asarray(mesh.vertex_normals).copy(),
            pose,
        )
        mesh_state[stone_id] = {"mesh": mesh, "step": step}
        geoms.append(mesh)

    def current_entry() -> tuple[int, int]:
        step, stone_id, _ = editable[selected["idx"]]
        return step, stone_id

    def current_source(stone_id: int) -> str:
        return initial_pose_sources.get(stone_id, "plan")

    def is_auto_locked_for_display(stone_id: int) -> bool:
        source = current_source(stone_id)
        if args.icp_prior_poses and is_previous_step_icp_source(source):
            return False
        return is_fixed_initial_pose_source(source)

    def is_fixed_for_display(step: int, stone_id: int) -> bool:
        return step in selected_fixed_steps or is_auto_locked_for_display(stone_id)

    def paint_meshes(vis=None) -> None:
        for idx, (step, stone_id, _) in enumerate(editable):
            mesh = mesh_state[stone_id]["mesh"]
            if idx == selected["idx"]:
                color = [1.0, 0.45, 0.05]
            elif is_fixed_for_display(step, stone_id):
                color = [0.65, 0.20, 0.90]
            else:
                color = [0.10, 0.35, 1.00]
            mesh.paint_uniform_color(color)
            if vis is not None:
                vis.update_geometry(mesh)

    def print_current(prefix: str = "selected") -> None:
        step, stone_id = current_entry()
        source = current_source(stone_id)
        user_fixed = step in selected_fixed_steps
        fixed = is_fixed_for_display(step, stone_id)
        print(
            f"  {prefix}: step {step}, stone {stone_id}, "
            f"source={source}, user_fixed={user_fixed}, fixed={fixed}"
        )

    def select_delta(vis, delta: int) -> bool:
        selected["idx"] = (selected["idx"] + delta) % len(editable)
        paint_meshes(vis)
        vis.update_renderer()
        print_current()
        return False

    def select_index(vis, idx: int) -> bool:
        if idx < len(editable):
            selected["idx"] = idx
            paint_meshes(vis)
            vis.update_renderer()
            print_current()
        return False

    def select_step_or_stone(vis) -> bool:
        value = input("  jump to step or stone id (e.g. 23 or s30): ").strip()
        if not value:
            return False
        target_idx = None
        if value.lower().startswith("s"):
            try:
                target_stone = int(value[1:])
            except ValueError:
                print(f"  invalid stone id: {value}")
                return False
            for idx, (_, stone_id, _) in enumerate(editable):
                if stone_id == target_stone:
                    target_idx = idx
                    break
        else:
            try:
                target_step = int(value)
            except ValueError:
                print(f"  invalid step: {value}")
                return False
            for idx, (step, _, _) in enumerate(editable):
                if step == target_step:
                    target_idx = idx
                    break
        if target_idx is None:
            print(f"  no editable entry for {value}")
            return False
        return select_index(vis, target_idx)

    def toggle_selected(vis) -> bool:
        step, _ = current_entry()
        if step in selected_fixed_steps:
            selected_fixed_steps.remove(step)
        else:
            selected_fixed_steps.add(step)
        paint_meshes(vis)
        vis.update_renderer()
        print_current("toggled")
        return False

    def close_window(vis) -> bool:
        vis.close()
        return False

    paint_meshes()
    print(f"\nOpening manual fix GUI for {scene_dir}")
    print_manual_fix_gui_help()
    print_manual_gui_index_table(editable)
    print_current()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    created = vis.create_window(
        window_name=f"manual fix: {comparison_label(scene_dir)}"
    )
    if not created:
        print("  Open3D viewer unavailable; manual fix GUI skipped.")
        vis.destroy_window()
        return sorted(selected_fixed_steps)

    for geom in geoms:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = args.manual_init_gui_point_size
        render_option.mesh_show_back_face = True
        render_option.light_on = True

    callbacks = {
        "N": lambda vis: select_delta(vis, 1),
        "P": lambda vis: select_delta(vis, -1),
        "M": toggle_selected,
        "G": select_step_or_stone,
        "C": close_window,
    }
    for key, callback in callbacks.items():
        vis.register_key_callback(ord(key), callback)
    vis.register_key_callback(32, toggle_selected)
    vis.register_key_callback(13, close_window)
    for idx in range(min(9, len(editable))):
        vis.register_key_callback(
            ord(str(idx + 1)), lambda vis, i=idx: select_index(vis, i)
        )

    vis.run()
    vis.destroy_window()

    print(f"  manual fixed steps: {sorted(selected_fixed_steps)}")
    return sorted(selected_fixed_steps)


def estimate_ground_height_ransac(
    pcd: o3d.geometry.PointCloud,
    reference_xy: np.ndarray,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    normal_z_min: float,
    min_inlier_ratio: float,
    max_abs_height: float,
) -> tuple[float, np.ndarray] | None:
    n_points = len(pcd.points)
    if n_points < ransac_n:
        print(
            "  ground height calibration skipped: "
            f"only {n_points} points available for ransac_n={ransac_n}"
        )
        return None

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
            "  ground height calibration skipped: "
            f"RANSAC plane normal={normal}, inliers={len(inliers)} / {n_points}"
        )
        return None
    if inlier_ratio < min_inlier_ratio:
        print(
            "  ground height calibration skipped: "
            f"RANSAC plane inlier ratio {inlier_ratio:.3f} < {min_inlier_ratio:.3f}"
        )
        return None

    a, b, c, d = np.asarray(plane_model, dtype=float)
    if abs(c) < 1e-9:
        print("  ground height calibration skipped: plane is nearly vertical")
        return None

    x_ref, y_ref = np.asarray(reference_xy, dtype=float)[:2]
    height = float(-(a * x_ref + b * y_ref + d) / c)
    if abs(height) > max_abs_height:
        print(
            "  ground height calibration skipped: "
            f"height={height:.4f} exceeds max_abs_height={max_abs_height:.4f}"
        )
        return None

    print(
        "  ground height calibration: "
        f"z={height:.4f} at xy={np.asarray(reference_xy).tolist()}, "
        f"normal={normal}, inliers={len(inliers)} / {n_points}"
    )
    return height, np.asarray(plane_model, dtype=np.float64)


def remove_ground_points_by_plane_model(
    pcd: o3d.geometry.PointCloud,
    plane_model: np.ndarray | None,
    distance_threshold: float,
    normal_z_min: float,
) -> o3d.geometry.PointCloud | None:
    if plane_model is None:
        return None

    points = np.asarray(pcd.points)
    n_points = len(points)
    if n_points == 0:
        return pcd

    plane = np.asarray(plane_model, dtype=np.float64).reshape(-1)
    if plane.shape[0] != 4 or not np.all(np.isfinite(plane)):
        print(f"ground removal by calibrated plane skipped: invalid plane {plane}")
        return None

    normal = plane[:3]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        print("ground removal by calibrated plane skipped: degenerate normal")
        return None

    unit_normal = normal / normal_norm
    if abs(float(unit_normal[2])) < normal_z_min:
        print(
            "ground removal by calibrated plane skipped: "
            f"normal={unit_normal}, normal_z_min={normal_z_min}"
        )
        return None

    distances = np.abs(points @ normal + float(plane[3])) / normal_norm
    ground_idx = np.flatnonzero(distances <= float(distance_threshold))
    if len(ground_idx) == 0:
        print(
            "removed ground by calibrated plane: 0 / "
            f"{n_points} points, normal={unit_normal}, d={plane[3]:.4f}"
        )
        return pcd

    filtered = pcd.select_by_index(ground_idx.tolist(), invert=True)
    print(
        "removed ground by calibrated plane: "
        f"{len(ground_idx)} / {n_points} points, "
        f"normal={unit_normal}, d={plane[3]:.4f}, "
        f"threshold={distance_threshold:.4f}, remaining={len(filtered.points)}"
    )
    return filtered


def apply_ground_height_to_plan_poses(
    planned_entries: list[tuple[int, np.ndarray]],
    ground_height_delta: float | None,
) -> list[tuple[int, np.ndarray]]:
    if ground_height_delta is None or abs(float(ground_height_delta)) <= 1e-12:
        return planned_entries

    adjusted = []
    for stone_id, pose in planned_entries:
        pose = pose.copy()
        pose[2] += float(ground_height_delta)
        adjusted.append((stone_id, pose))
    return adjusted


def plan_place_plane_height(plan_dir: Path) -> float:
    params_path = plan_dir / "planning_params.pkl"
    if not params_path.exists():
        return 0.0
    try:
        with params_path.open("rb") as f:
            params = pickle.load(f)
    except Exception as exc:
        print(f"  warning: could not read planning params {params_path}: {exc}")
        return 0.0
    try:
        height = float(params.get("place_plane_height", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(height):
        return 0.0
    return height


def ground_height_pose_delta(
    plan_dir: Path,
    ground_height: float | None,
) -> float | None:
    if ground_height is None:
        return None
    reference_height = plan_place_plane_height(plan_dir)
    delta = float(ground_height) - reference_height
    print(
        "  ground height pose delta: "
        f"scene z={float(ground_height):.4f}, "
        f"plan place_plane_height={reference_height:.4f}, dz={delta:.4f}"
    )
    return delta


def mark_ground_height_sources(
    initial_pose_sources: dict[int, str],
    planned_entries: list[tuple[int, np.ndarray]],
    ground_height_delta: float | None,
) -> None:
    if ground_height_delta is None or abs(float(ground_height_delta)) <= 1e-12:
        return
    for stone_id, _ in planned_entries:
        source = initial_pose_sources.get(stone_id)
        if source in {"plan", "plan_state"}:
            initial_pose_sources[stone_id] = f"{source}+ground"


def set_sceneid_ground_height(
    config: sceneid.Config, ground_height: float | None
) -> None:
    if ground_height is None:
        return
    if hasattr(config.obj, "ground_height"):
        config.obj.ground_height = float(ground_height)
        print(f"  sceneid ground height: z={ground_height:.4f}")
    else:
        print(
            "  sceneid ground height: diffsimpy binding has no "
            "obj.ground_height; rebuild ../diffsim to apply it inside sceneid"
        )


def apply_sceneid_args_to_config(
    config: sceneid.Config, args: argparse.Namespace
) -> None:
    config.n_threads = args.n_threads
    config.log_interval = args.log_interval
    config.tr.max_iter = args.max_iter
    config.tr.eps = args.trust_region_eps
    config.tr.delta_init = args.delta_init
    config.graph.max_iter = args.graph_max_iter
    config.obj.k_pcd = args.k_pcd
    config.obj.pcd_huber_delta = args.pcd_huber_delta
    config.obj.pcd_max_gap = args.pcd_max_gap
    config.obj.k_gap_c = args.k_gap_c
    config.obj.k_comp = args.k_comp


def transformation_matrix_to_pose(T: np.ndarray) -> np.ndarray:
    pose = np.empty(7, dtype=np.float64)
    pose[:3] = T[:3, 3]
    pose[3:] = Rotation.from_matrix(T[:3, :3]).as_quat()
    return pose


def rotation_delta_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    rot_a = Rotation.from_quat(pose_a[3:])
    rot_b = Rotation.from_quat(pose_b[3:])
    return float(np.rad2deg((rot_a.inv() * rot_b).magnitude()))


def is_fixed_initial_pose_source(source: str) -> bool:
    return (
        source.startswith("prior")
        or "fixed_missing_scene" in source
        or "user_fixed" in source
        or "manual_fixed" in source
    )


def is_locked_solution_pose_source(
    source: str,
    args: argparse.Namespace | None = None,
) -> bool:
    if "user_fixed" in source or "manual_fixed" in source:
        return True

    if source.startswith("prior_empty_icp_target") or source.startswith(
        "prior_low_icp_corr"
    ):
        return True

    if args is not None and getattr(args, "icp_prior_poses", False):
        if source == "prior" or source.startswith("prior+icp"):
            return False
        if "fixed_missing_scene" in source and source.endswith("+icp"):
            return False

    return is_fixed_initial_pose_source(source)


def is_icp_refinable_initial_pose_source(source: str) -> bool:
    return source.startswith("plan") or source.startswith("manual")


def is_previous_step_icp_source(source: str) -> bool:
    if "user_fixed" in source or "manual_fixed" in source:
        return False
    return source == "prior" or "fixed_missing_scene" in source


def icp_orientation_deltas(axes: str, step_deg: float) -> list[np.ndarray]:
    if step_deg <= 0.0:
        return [np.eye(3)]

    axes = "".join(axis for axis in axes.lower() if axis in "xyz")
    if not axes:
        return [np.eye(3)]

    angles = np.arange(0.0, 360.0, step_deg, dtype=float)
    angle_lists = [angles if axis in axes else np.array([0.0]) for axis in "xyz"]
    mesh = np.meshgrid(*angle_lists, indexing="ij")
    rots = []
    seen = set()
    for rx, ry, rz in zip(*(m.flatten() for m in mesh)):
        key = (round(float(rx), 6), round(float(ry), 6), round(float(rz), 6))
        if key in seen:
            continue
        seen.add(key)
        rots.append(Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix())
    return rots


def crop_scene_pcd_for_pose_icp(
    scene_pcd: o3d.geometry.PointCloud,
    stone_id: int,
    pose: np.ndarray,
    visual_meshes: dict[int, o3d.geometry.TriangleMesh],
    margin: float,
) -> o3d.geometry.PointCloud:
    if stone_id in visual_meshes:
        mesh = copy.deepcopy(visual_meshes[stone_id])
        mesh.transform(pose_to_transformation_matrix(pose))
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bbox.min_bound, dtype=float) - margin
        max_bound = np.asarray(bbox.max_bound, dtype=float) + margin
    else:
        min_bound = pose[:3] - margin
        max_bound = pose[:3] + margin

    return scene_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))


def filter_sceneid_entries_by_point_support(
    scene_pcd: o3d.geometry.PointCloud,
    planned_entries: list[tuple[int, np.ndarray]],
    initial_pose_sources: dict[int, str],
    visual_meshes: dict[int, o3d.geometry.TriangleMesh],
    args: argparse.Namespace,
) -> tuple[list[tuple[int, np.ndarray]], dict[int, dict], dict[int, dict]]:
    min_points = int(getattr(args, "sceneid_min_stone_points", 0) or 0)
    if min_points <= 0:
        return planned_entries, {}, {}

    margin = float(
        getattr(args, "sceneid_stone_point_margin", None)
        if getattr(args, "sceneid_stone_point_margin", None) is not None
        else getattr(args, "icp_crop_margin", 0.35)
    )
    if not np.isfinite(margin) or margin < 0.0:
        margin = float(getattr(args, "icp_crop_margin", 0.35))

    kept = []
    point_counts = {}
    skipped = {}
    for stone_id, pose in planned_entries:
        source = initial_pose_sources.get(stone_id, "unknown")
        nearby = crop_scene_pcd_for_pose_icp(
            scene_pcd,
            stone_id,
            pose,
            visual_meshes,
            margin,
        )
        point_count = len(nearby.points)
        kept_stone = point_count >= min_points
        point_counts[int(stone_id)] = {
            "points": int(point_count),
            "min_points": int(min_points),
            "margin": float(margin),
            "source": source,
            "kept": bool(kept_stone),
        }
        decision = "kept" if kept_stone else "skipped"
        print(
            f"  SceneID support stone {stone_id}: {point_count} nearby "
            f"non-ground scene points (min={min_points}, "
            f"margin={margin:.3f}m, source={source}, {decision})"
        )
        if point_count >= min_points:
            kept.append((stone_id, pose))
            continue

        skipped[int(stone_id)] = point_counts[int(stone_id)].copy()
        initial_pose_sources[stone_id] = f"{source}+skipped_low_scene_points"

    if skipped:
        print(
            "  SceneID low-point filter skipped "
            f"{len(skipped)} / {len(planned_entries)} stone(s): "
            f"{sorted(skipped)}"
        )
    return kept, point_counts, skipped


def source_pcd_for_stone(
    stone_id: int,
    stone_pcds: dict[int, o3d.geometry.PointCloud],
    visual_meshes: dict[int, o3d.geometry.TriangleMesh],
    n_mesh_points: int,
) -> o3d.geometry.PointCloud:
    if stone_id in stone_pcds and not stone_pcds[stone_id].is_empty():
        return copy.deepcopy(stone_pcds[stone_id])
    if stone_id in visual_meshes:
        return visual_meshes[stone_id].sample_points_uniformly(
            number_of_points=n_mesh_points
        )
    return o3d.geometry.PointCloud()


def icp_alignment_stats(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    transform: np.ndarray,
    voxel_size: float,
    correspondence_distance_scale: float,
) -> tuple[float, float, int]:
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    threshold = voxel_size * correspondence_distance_scale
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_down,
        target_down,
        threshold,
        transform,
    )
    return (
        float(evaluation.fitness),
        float(evaluation.inlier_rmse),
        int(len(evaluation.correspondence_set)),
    )


def icp_candidate_score(
    fitness: float,
    rmse: float,
    translation_from_seed: float,
    rotation_from_seed_deg: float,
    args: argparse.Namespace,
) -> float:
    rmse_weight = float(getattr(args, "icp_rmse_weight", 0.25))
    translation_weight = float(getattr(args, "icp_translation_weight", 1.0))
    rotation_weight = float(getattr(args, "icp_rotation_weight", 0.0))
    return float(
        fitness
        - rmse_weight * rmse
        - translation_weight * translation_from_seed
        - rotation_weight * (rotation_from_seed_deg / 180.0)
    )


def refine_plan_initial_poses_with_icp(
    scene_pcd: o3d.geometry.PointCloud,
    planned_entries: list[tuple[int, np.ndarray]],
    initial_pose_sources: dict[int, str],
    stone_pcds: dict[int, o3d.geometry.PointCloud],
    visual_meshes: dict[int, o3d.geometry.TriangleMesh],
    args: argparse.Namespace,
    prior_fallback_poses_by_stone: dict[int, np.ndarray] | None = None,
) -> list[tuple[int, np.ndarray]]:
    if args.no_icp_init:
        return planned_entries

    prior_fallback_poses_by_stone = prior_fallback_poses_by_stone or {}
    rots = icp_orientation_deltas(args.icp_axes, args.icp_angle_step_deg)
    refined_entries = []
    n_refined = 0
    n_prior_fallback = 0
    n_low_corr_fallback = 0
    for stone_id, pose in planned_entries:
        source = initial_pose_sources.get(stone_id, "plan")
        refine_previous_step_pose = (
            args.icp_prior_poses and is_previous_step_icp_source(source)
        )
        min_correspondences = int(args.icp_min_correspondences)
        min_fitness = float(args.icp_min_fitness)
        if refine_previous_step_pose:
            prior_min_correspondences = getattr(
                args,
                "icp_prior_min_correspondences",
                None,
            )
            if prior_min_correspondences is not None:
                min_correspondences = max(
                    min_correspondences,
                    int(prior_min_correspondences),
                )
            prior_min_fitness = getattr(args, "icp_prior_min_fitness", None)
            if prior_min_fitness is not None:
                min_fitness = max(min_fitness, float(prior_min_fitness))
        if is_fixed_initial_pose_source(source) and not refine_previous_step_pose:
            refined_entries.append((stone_id, pose))
            continue
        if (
            args.icp_plan_only
            and not refine_previous_step_pose
            and not is_icp_refinable_initial_pose_source(source)
        ):
            refined_entries.append((stone_id, pose))
            continue

        target_pcd = crop_scene_pcd_for_pose_icp(
            scene_pcd,
            stone_id,
            pose,
            visual_meshes,
            args.icp_crop_margin,
        )
        if (
            not args.no_icp_empty_target_prior_fallback
            and source.startswith("plan")
            and len(target_pcd.points) < args.icp_min_target_points
            and stone_id in prior_fallback_poses_by_stone
        ):
            fallback_pose = prior_fallback_poses_by_stone[stone_id].copy()
            dxyz = float(np.linalg.norm(fallback_pose[:3] - pose[:3]))
            drot = rotation_delta_deg(pose, fallback_pose)
            print(
                f"  ICP init stone {stone_id}: target crop has "
                f"{len(target_pcd.points)} < {args.icp_min_target_points} points; "
                "using latest previous identified pose "
                f"(dxyz={dxyz:.3f}m, drot={drot:.1f}deg)"
            )
            refined_entries.append((stone_id, fallback_pose))
            initial_pose_sources[stone_id] = f"prior_empty_icp_target_from_{source}"
            n_prior_fallback += 1
            continue

        source_pcd = source_pcd_for_stone(
            stone_id,
            stone_pcds,
            visual_meshes,
            args.icp_mesh_points,
        )
        if args.icp_source_voxel_size > 0.0:
            source_pcd = source_pcd.voxel_down_sample(args.icp_source_voxel_size)

        if (
            len(target_pcd.points) < args.icp_min_target_points
            or len(source_pcd.points) == 0
        ):
            print(
                f"  ICP init stone {stone_id}: skipped "
                f"(target={len(target_pcd.points)}, source={len(source_pcd.points)})"
            )
            refined_entries.append((stone_id, pose))
            continue

        init_T_base = pose_to_transformation_matrix(pose)
        final_voxel_size = float(args.icp_voxel_sizes[-1])
        correspondence_distance_scale = float(
            getattr(args, "icp_correspondence_distance_scale", 1.5)
        )
        init_fitness, init_rmse, init_correspondences = icp_alignment_stats(
            source_pcd,
            target_pcd,
            init_T_base,
            final_voxel_size,
            correspondence_distance_scale,
        )
        init_score = icp_candidate_score(
            init_fitness,
            init_rmse,
            0.0,
            0.0,
            args,
        )
        best = None
        best_rejected = None
        for rot_delta in rots:
            init_T = init_T_base.copy()
            init_T[:3, :3] = init_T_base[:3, :3] @ rot_delta
            target_T, history = multiscale_icp(
                source_pcd,
                target_pcd,
                init_T,
                voxel_sizes=args.icp_voxel_sizes,
                max_iters=args.icp_max_iters,
                max_correspondence_distance_scale=correspondence_distance_scale,
            )
            _, fitness, rmse, correspondences = history[-1]
            candidate_pose = transformation_matrix_to_pose(target_T)
            translation = float(np.linalg.norm(candidate_pose[:3] - pose[:3]))
            rotation = rotation_delta_deg(pose, candidate_pose)
            accepted = (
                correspondences >= min_correspondences
                and fitness >= min_fitness
                and translation <= args.icp_max_translation
            )
            score = icp_candidate_score(
                fitness,
                rmse,
                translation,
                rotation,
                args,
            )
            rejected = (
                score,
                candidate_pose,
                fitness,
                rmse,
                correspondences,
                translation,
                rotation,
            )
            if best_rejected is None or score > best_rejected[0]:
                best_rejected = rejected
            if accepted and (best is None or score > best[0]):
                best = (
                    score,
                    candidate_pose,
                    fitness,
                    rmse,
                    correspondences,
                    translation,
                    rotation,
                )

        min_score_gain = float(getattr(args, "icp_min_score_gain", 0.0))
        if best is not None and best[0] < init_score + min_score_gain:
            _, _, fitness, rmse, correspondences, translation, rotation = best
            print(
                f"  ICP init stone {stone_id}: kept initial pose "
                f"(init score={init_score:.4f}, fitness={init_fitness:.3f}, "
                f"rmse={init_rmse:.4f}, corr={init_correspondences}; "
                f"best ICP score={best[0]:.4f}, fitness={fitness:.3f}, "
                f"rmse={rmse:.4f}, corr={correspondences}, "
                f"dxyz={translation:.3f}m, drot={rotation:.1f}deg)"
            )
            refined_entries.append((stone_id, pose))
            continue

        if best is None:
            if (
                not args.no_icp_low_correspondence_prior_fallback
                and best_rejected is not None
                and (
                    best_rejected[4] < min_correspondences
                    or best_rejected[2] < min_fitness
                )
                and stone_id in prior_fallback_poses_by_stone
            ):
                fallback_pose = prior_fallback_poses_by_stone[stone_id].copy()
                _, _, fitness, rmse, correspondences, translation, rotation = (
                    best_rejected
                )
                dxyz = float(np.linalg.norm(fallback_pose[:3] - pose[:3]))
                drot = rotation_delta_deg(pose, fallback_pose)
                reject_reasons = []
                if correspondences < min_correspondences:
                    reject_reasons.append(
                        f"corr {correspondences} < {min_correspondences}"
                    )
                if fitness < min_fitness:
                    reject_reasons.append(f"fitness {fitness:.3f} < {min_fitness:.3f}")
                print(
                    f"  ICP init stone {stone_id}: best hypothesis rejected "
                    f"({', '.join(reject_reasons)}); fixing to latest previous "
                    "identified pose "
                    f"(best fitness={fitness:.3f}, rmse={rmse:.4f}, "
                    f"dxyz_best={translation:.3f}m, drot_best={rotation:.1f}deg, "
                    f"dxyz_prior={dxyz:.3f}m, drot_prior={drot:.1f}deg)"
                )
                refined_entries.append((stone_id, fallback_pose))
                initial_pose_sources[stone_id] = f"prior_low_icp_corr_from_{source}"
                n_low_corr_fallback += 1
                continue
            print(f"  ICP init stone {stone_id}: no accepted orientation hypothesis")
            refined_entries.append((stone_id, pose))
            continue

        _, refined_pose, fitness, rmse, correspondences, translation, rotation = best
        print(
            f"  ICP init stone {stone_id}: {source} -> refined, "
            f"score={best[0]:.4f} > init {init_score:.4f}, "
            f"fitness={fitness:.3f}, rmse={rmse:.4f}, corr={correspondences}, "
            f"dxyz={translation:.3f}m, drot={rotation:.1f}deg"
        )
        refined_entries.append((stone_id, refined_pose))
        initial_pose_sources[stone_id] = f"{source}+icp"
        n_refined += 1

    if n_refined:
        print(f"  ICP init refined {n_refined} / {len(planned_entries)} poses")
    if n_prior_fallback:
        print(
            "  ICP init used previous identified pose fallback for "
            f"{n_prior_fallback} / {len(planned_entries)} poses"
        )
    if n_low_corr_fallback:
        print(
            "  ICP init fixed low-correspondence poses to previous identified "
            f"configuration for {n_low_corr_fallback} / {len(planned_entries)} poses"
        )
    return refined_entries


def write_results(
    scene_dir: Path,
    output_prefix: str,
    plan_dir: Path,
    action_path: Path,
    step_index: int,
    scene_pcd_path: Path | None,
    sceneid_to_stone: dict[int, int],
    initial_poses_by_body: dict[int, np.ndarray],
    initial_pose_sources: dict[int, str],
    optimal_poses_by_stone: dict[int, np.ndarray],
    ground_height: float | None,
    ground_plane_model: np.ndarray | None,
    skipped_sceneid_solve: bool,
    elapsed: float,
    sceneid_stone_point_counts: dict[int, dict] | None = None,
    skipped_low_scene_points: dict[int, dict] | None = None,
):
    sceneid_stone_point_counts = sceneid_stone_point_counts or {}
    skipped_low_scene_points = skipped_low_scene_points or {}
    initial_poses_by_stone = {
        sceneid_to_stone[body_id]: pose.copy()
        for body_id, pose in initial_poses_by_body.items()
    }
    result = {
        "plan_dir": str(plan_dir),
        "action_sequence": str(action_path),
        "step_index": step_index,
        "scene_scan_dir": str(scene_dir),
        "scene_pcd_path": str(scene_pcd_path) if scene_pcd_path is not None else None,
        "stone_ids": sorted(optimal_poses_by_stone),
        "initial_poses": initial_poses_by_stone,
        "initial_pose_sources": initial_pose_sources,
        "optimal_poses": optimal_poses_by_stone,
        "ground_height": ground_height,
        "ground_plane_model": (
            ground_plane_model.copy() if ground_plane_model is not None else None
        ),
        "sceneid_stone_point_counts": sceneid_stone_point_counts,
        "skipped_low_scene_point_stones": skipped_low_scene_points,
        "skipped_sceneid_solve": skipped_sceneid_solve,
        "elapsed": elapsed,
    }

    pkl_path = scene_dir / f"{output_prefix}.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(result, f)

    txt_path = scene_dir / f"{output_prefix}.txt"
    with txt_path.open("w") as f:
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
        if sceneid_stone_point_counts:
            f.write("sceneid_stone_point_counts:\n")
            for stone_id in sorted(sceneid_stone_point_counts):
                item = sceneid_stone_point_counts[stone_id]
                f.write(
                    f"  stone {stone_id}: points={item.get('points')} "
                    f"min_points={item.get('min_points')} "
                    f"margin={item.get('margin')} "
                    f"source={item.get('source')} "
                    f"kept={item.get('kept')}\n"
                )
        if skipped_low_scene_points:
            f.write("skipped_low_scene_point_stones:\n")
            for stone_id in sorted(skipped_low_scene_points):
                item = skipped_low_scene_points[stone_id]
                f.write(
                    f"  stone {stone_id}: points={item.get('points')} "
                    f"min_points={item.get('min_points')} "
                    f"margin={item.get('margin')} "
                    f"source={item.get('source')}\n"
                )
        f.write(f"elapsed: {elapsed:.6f}\n")
        for stone_id in sorted(optimal_poses_by_stone):
            f.write(
                f"stone {stone_id}: init_source={initial_pose_sources[stone_id]} "
                f"init={initial_poses_by_stone[stone_id].tolist()} "
                f"optimal={optimal_poses_by_stone[stone_id].tolist()}\n"
            )

    print(f"  wrote: {pkl_path}")
    print(f"  wrote: {txt_path}")


def fix_locked_solution_poses(
    sceneid_to_stone: dict[int, int],
    initial_poses_by_body: dict[int, np.ndarray],
    initial_pose_sources: dict[int, str],
    solution_poses_by_body: dict,
    optimal_poses_by_stone: dict[int, np.ndarray],
    args: argparse.Namespace | None = None,
) -> tuple[dict, dict[int, np.ndarray]]:
    solution_poses_fixed = {
        int(body_id): pose for body_id, pose in solution_poses_by_body.items()
    }
    fixed_stats = []
    for body_id, stone_id in sceneid_to_stone.items():
        body_id = int(body_id)
        stone_id = int(stone_id)
        source = initial_pose_sources.get(stone_id, "")
        if not is_locked_solution_pose_source(source, args):
            continue
        fixed_pose = np.asarray(initial_poses_by_body[body_id], dtype=np.float64).copy()
        solved_pose = optimal_poses_by_stone.get(stone_id)
        if solved_pose is not None:
            dxyz = float(np.linalg.norm(np.asarray(solved_pose)[:3] - fixed_pose[:3]))
            drot = rotation_delta_deg(fixed_pose, np.asarray(solved_pose))
            fixed_stats.append((stone_id, dxyz, drot))
        optimal_poses_by_stone[stone_id] = fixed_pose
        solution_poses_fixed[body_id] = PoseVector(fixed_pose)

    if fixed_stats:
        max_dxyz = max(item[1] for item in fixed_stats)
        max_drot = max(item[2] for item in fixed_stats)
        stone_ids = [item[0] for item in fixed_stats]
        print(
            "  fixed locked poses: "
            f"{stone_ids} (max solver delta {max_dxyz:.4f}m, {max_drot:.2f}deg)"
        )
    return solution_poses_fixed, optimal_poses_by_stone


def action_stone_id(action) -> int:
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, dict):
        return int(action["stone_id"])
    return int(action.stone_id)


def action_pose(action) -> np.ndarray | None:
    pose = (
        action.get("pose", None)
        if isinstance(action, dict)
        else getattr(action, "pose", None)
    )
    if pose is None:
        return None
    return np.asarray(pose, dtype=np.float64).copy()


def action_init_pose(action, fallback_pose: np.ndarray) -> np.ndarray:
    init_pose = (
        action.get("init_pose", None)
        if isinstance(action, dict)
        else getattr(action, "init_pose", None)
    )
    if init_pose is None:
        return np.asarray(fallback_pose, dtype=np.float64).copy()
    return np.asarray(init_pose, dtype=np.float64).copy()


def load_existing_state_data(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"  warning: could not read existing state pkl {path}: {exc}")
        return {}


def stone_set_from_state_data(data: dict) -> list[int] | None:
    candidates = []
    if isinstance(data, dict):
        candidates.append(data.get("resume_state", None))
        for step in reversed(list(data.get("steps", []))):
            if not isinstance(step, dict):
                continue
            candidates.append(step.get("resume_state", None))
            candidates.append(step.get("raw_state", None))

    for state in candidates:
        stone_set = getattr(state, "stone_set", None)
        if stone_set is not None:
            return [int(stone_id) for stone_id in stone_set]
    return None


def reconstructed_stone_set(
    plan_dir: Path,
    action_sequence: list,
    placed_stone_ids: list[int],
    existing_data: dict,
) -> list[int]:
    stone_set = stone_set_from_state_data(existing_data)
    if stone_set is None:
        pose_data_path = plan_dir / "pose_data.pkl"
        if pose_data_path.exists():
            try:
                with pose_data_path.open("rb") as f:
                    pose_data = pickle.load(f)
                stone_set = [int(stone_id) for stone_id in pose_data.keys()]
            except Exception as exc:
                print(f"  warning: could not read pose data {pose_data_path}: {exc}")

    if stone_set is None:
        seen = set()
        stone_set = []
        for action in action_sequence:
            stone_id = action_stone_id(action)
            if stone_id not in seen:
                seen.add(stone_id)
                stone_set.append(stone_id)

    missing_placed = [
        stone_id for stone_id in placed_stone_ids if stone_id not in stone_set
    ]
    if missing_placed:
        stone_set = missing_placed + stone_set
    return [int(stone_id) for stone_id in stone_set]


def state_pose_from_sceneid_pose(
    world_pose: np.ndarray, target_offset: np.ndarray
) -> np.ndarray:
    pose = np.asarray(world_pose, dtype=np.float64).copy()
    pose[:2] -= np.asarray(target_offset, dtype=np.float64)[:2]
    return pose


def make_reconstructed_trajectories(
    placed_stone_ids: list[int],
    stone_poses: dict[int, np.ndarray],
) -> list[dict[int, StoneTrajectory]]:
    trajectories = []
    for step_i in range(len(placed_stone_ids)):
        step_trajectories = {}
        for stone_id in placed_stone_ids[: step_i + 1]:
            traj = StoneTrajectory(stone_id)
            traj.add_pose(stone_poses[stone_id].copy())
            traj.vel_integral = 0.0
            step_trajectories[stone_id] = traj
        trajectories.append(step_trajectories)
    return trajectories


def build_reconstructed_state(
    plan_dir: Path,
    action_sequence: list,
    step_index: int,
    optimal_world_poses: dict[int, np.ndarray],
    target_offset: np.ndarray,
    mark_pose_identified: bool,
    existing_data: dict,
    placed_stone_ids: list[int] | None = None,
) -> State:
    prefix = action_sequence[:step_index]
    if placed_stone_ids is None:
        placed_stone_ids = [action_stone_id(action) for action in prefix]
    stone_set = reconstructed_stone_set(
        plan_dir, action_sequence, placed_stone_ids, existing_data
    )
    stone_idx_by_id = {int(stone_id): idx for idx, stone_id in enumerate(stone_set)}
    missing = [
        stone_id for stone_id in placed_stone_ids if stone_id not in stone_idx_by_id
    ]
    if missing:
        raise ValueError(
            f"placed stone ids missing from reconstructed stone_set: {missing}"
        )

    stone_seq = [stone_idx_by_id[stone_id] for stone_id in placed_stone_ids]
    stone_poses = {}
    pose_identified_stone_ids = set()
    action_by_stone = {action_stone_id(action): action for action in prefix}
    for stone_id in placed_stone_ids:
        if stone_id in optimal_world_poses:
            stone_poses[stone_id] = state_pose_from_sceneid_pose(
                optimal_world_poses[stone_id],
                target_offset,
            )
            pose_identified_stone_ids.add(stone_id)
            continue

        action = action_by_stone.get(stone_id)
        if action is None:
            raise ValueError(f"no action fallback for stone {stone_id}")
        planned_pose = action_pose(action)
        if planned_pose is None:
            raise ValueError(f"action for stone {stone_id} has no pose fallback")
        stone_poses[stone_id] = planned_pose

    action_history = []
    for stone_id in placed_stone_ids:
        pose = stone_poses[stone_id]
        action = action_by_stone.get(stone_id)
        action_history.append(
            Action(
                stone_idx=stone_idx_by_id[stone_id],
                stone_id=stone_id,
                pose=pose.copy(),
                init_pose=(
                    action_init_pose(action, pose)
                    if action is not None
                    else pose.copy()
                ),
            )
        )

    return State(
        stone_set=stone_set,
        stone_seq=stone_seq,
        stone_poses=stone_poses,
        trajectories=make_reconstructed_trajectories(placed_stone_ids, stone_poses),
        action_history=action_history,
        terminated=False,
        failed=False,
        contact_points=[],
        pose_identified_stone_ids=(
            pose_identified_stone_ids if mark_pose_identified else set()
        ),
    )


def write_reconstructed_state_pkl(
    record: dict,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    plan_dir = Path(record["plan_dir"])
    action_path = plan_dir / "action_sequence.pkl"
    action_sequence = load_action_sequence(action_path)
    step_index = int(record["step_index"])
    state_path_record = record.get("state_path", None)
    plan_state_path = (
        Path(state_path_record)
        if state_path_record is not None
        else plan_dir / DEFAULT_PLAN_STATE_NAME
    )
    placed_stone_ids = record.get("placed_stone_ids", None)
    if placed_stone_ids is not None:
        placed_stone_ids = [int(stone_id) for stone_id in placed_stone_ids]
    else:
        placed_stone_ids = state_backed_step_sequence(
            plan_dir,
            action_sequence,
            step_index,
            plan_state_path,
        )
    target_offset = (
        np.zeros(2)
        if args.no_offset
        else target_offset_for_action_sequence(action_path)
    )

    existing_data = load_existing_state_data(output_path)
    if not existing_data and output_path.resolve() != plan_state_path.resolve():
        existing_data = load_existing_state_data(plan_state_path)
    state = build_reconstructed_state(
        plan_dir,
        action_sequence,
        step_index,
        record["optimal_poses"],
        target_offset,
        mark_pose_identified=not args.no_mark_pose_identified,
        existing_data=existing_data,
        placed_stone_ids=placed_stone_ids,
    )

    output_data = dict(existing_data)
    output_data.setdefault("stone_meshes", {})
    output_data.setdefault("stone_ply_meshes", {})
    output_data.setdefault("target_wall_meshes", [])
    output_data.setdefault("target_wall_cfg", {})
    output_data.setdefault("mesh_source", "unknown")
    output_data["steps"] = list(output_data.get("steps", []))
    output_data["steps"].append(
        {
            "step": step_index,
            "succeeded": True,
            "scene": {
                "stone_seq": [
                    int(stone_id) for stone_id in placed_stone_ids
                ],
                "stone_poses": {
                    int(stone_id): pose.copy()
                    for stone_id, pose in state.stone_poses.items()
                },
                "pose_identified_stone_ids": sorted(state.pose_identified_stone_ids),
            },
            "candidates": [],
            "raw_state": copy.deepcopy(state),
            "resume_state": copy.deepcopy(state),
            "reconstructed_from_logs": True,
            "source_scene_scan_dir": str(record["scene_dir"]),
        }
    )
    output_data["resume_state"] = copy.deepcopy(state)
    output_data["resume_step"] = step_index
    output_data["reconstructed_from_logs"] = {
        "scene_scan_dir": str(record["scene_dir"]),
        "plan_dir": str(plan_dir),
        "step_index": step_index,
        "ground_height": record["ground_height"],
        "ground_plane_model": (
            record["ground_plane_model"].copy()
            if record["ground_plane_model"] is not None
            else None
        ),
        "pose_identified_stone_ids": sorted(state.pose_identified_stone_ids),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(output_data, f)

    print(
        "  reconstructed state: "
        f"{output_path} ({len(state.stone_seq)} placed, "
        f"pose_identified={sorted(state.pose_identified_stone_ids)})"
    )


def identify_scene_dir(
    scene_dir: Path,
    args: argparse.Namespace,
    prior_poses_by_stone: dict[int, np.ndarray] | None = None,
    calibrated_ground_height: float | None = None,
    calibrated_ground_plane_model: np.ndarray | None = None,
) -> tuple[bool, dict[int, np.ndarray], float | None, np.ndarray | None]:
    print(f"\nScene scan: {scene_dir}")
    prior_poses_by_stone = prior_poses_by_stone or {}
    step_index = step_index_from_scene_dir(scene_dir)
    plan_dir = resolve_plan_dir(scene_dir, args.plan_dir)
    action_path = plan_dir / "action_sequence.pkl"
    action_sequence = load_action_sequence(action_path)
    step_sequence = state_backed_step_sequence(plan_dir, action_sequence, step_index)
    if len(step_sequence) < step_index:
        raise ValueError(
            f"{scene_dir}: step {step_index} exceeds available placed stones "
            f"from state.pkl/action_sequence.pkl ({len(step_sequence)})"
        )

    scene_pcd, scene_pcd_path = read_or_merge_scene_pcd(
        scene_dir,
        args.remerge,
        args.raw,
        args.merge_voxel_size,
        args.merged_name,
    )
    print(
        f"  plan: {plan_dir.name}, step={step_index}, "
        f"placed stones={step_index}, scene points={len(scene_pcd.points)}"
    )

    target_offset = (
        np.zeros(2)
        if args.no_offset
        else target_offset_for_action_sequence(action_path)
    )
    asset_dir = args.asset_dir or asset_dir_for_action_sequence(
        action_path, "assets/stone"
    )

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
    planned_entries_plan, plan_pose_sources = state_pose_entries_from_plan_state(
        plan_dir,
        step_sequence,
        step_index,
        target_offset,
        action_entries_plan,
    )
    if not args.no_ground_height_init:
        if calibrated_ground_height is None:
            reference_xy = np.mean(
                np.vstack([pose[:2] for _, pose in planned_entries_plan]), axis=0
            )
            ground_height_pcd = scene_pcd
            if not args.no_ground_height_crop:
                try:
                    ground_height_pcd = crop_scene_pcd_near_expected_poses(
                        scene_pcd,
                        scene_pcd_path or scene_dir,
                        planned_entries_plan,
                        visual_meshes,
                        args.ground_height_crop_margin,
                        args.ground_height_crop_max_margin,
                    )
                except Exception as exc:
                    print(
                        "  ground height crop skipped: "
                        f"{exc}; using full scene PCD for calibration"
                    )
            ground_estimate = estimate_ground_height_ransac(
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
                calibrated_ground_height, calibrated_ground_plane_model = (
                    ground_estimate
                )
        else:
            print(
                f"  ground height calibration: reusing z={calibrated_ground_height:.4f}"
            )
    pose_ground_delta = ground_height_pose_delta(
        plan_dir,
        None if args.no_ground_height_init else calibrated_ground_height,
    )
    planned_entries_plan = apply_ground_height_to_plan_poses(
        planned_entries_plan,
        pose_ground_delta,
    )
    prior_poses_for_step = prior_poses_before_step(
        prior_poses_by_stone,
        step_sequence,
        step_index,
    )
    icp_prior_fallback_poses = latest_previous_identified_poses_from_logs(
        scene_dir,
        args.output_prefix,
        step_sequence,
        step_index,
    )
    icp_prior_fallback_poses.update(
        {stone_id: pose.copy() for stone_id, pose in prior_poses_for_step.items()}
    )
    planned_entries, initial_pose_sources = apply_recursive_initial_poses(
        planned_entries_plan,
        prior_poses_for_step,
        enabled=not args.no_recursive_init,
        base_sources=plan_pose_sources,
    )
    mark_ground_height_sources(
        initial_pose_sources,
        planned_entries,
        pose_ground_delta,
    )
    planned_entries, manual_applied = apply_manual_initial_poses(
        planned_entries,
        initial_pose_sources,
        step_sequence,
        step_index,
        args.manual_init_by_step,
        args.manual_init_fixed,
    )
    if manual_applied:
        print(
            "  manual init poses: "
            + ", ".join(
                f"step {step} -> stone {stone_id}" for step, stone_id in manual_applied
            )
            + (" (fixed)" if args.manual_init_fixed else " (ICP/sceneid refinable)")
        )
    fixed_missing_steps = []
    fixed_missing_stone_ids = []
    if not args.no_recursive_init:
        if not args.no_fix_missing_prior_steps:
            fixed_missing_steps.extend(
                missing_prior_step_indices(scene_dir, step_index)
            )
        fixed_missing_steps.extend(
            step for step in getattr(args, "fix_planned_steps", []) if step < step_index
        )
        fixed_missing_steps = sorted(set(fixed_missing_steps))
        fixed_missing_stone_ids = mark_missing_prior_step_sources(
            initial_pose_sources,
            step_sequence,
            fixed_missing_steps,
        )
    if fixed_missing_stone_ids:
        print(
            "  fixing planned poses for prior steps without usable sceneid: "
            f"steps={fixed_missing_steps}; stones="
            f"{fixed_missing_stone_ids}"
        )
    if args.manual_init_gui:
        planned_entries, manual_gui_poses, _ = run_manual_initialization_gui(
            scene_pcd,
            planned_entries,
            initial_pose_sources,
            step_sequence,
            step_index,
            visual_meshes,
            args,
            scene_dir,
        )
        if args.manual_init_gui_output is not None:
            saved_manual_poses = dict(args.manual_init_by_step)
            saved_manual_poses.update(manual_gui_poses)
            save_manual_init_pose_file(args.manual_init_gui_output, saved_manual_poses)
        args.manual_init_by_step.update(manual_gui_poses)
    if args.manual_fix_gui:
        args.fix_initial_steps = run_manual_fix_gui(
            scene_pcd,
            planned_entries,
            initial_pose_sources,
            step_sequence,
            step_index,
            visual_meshes,
            args,
            scene_dir,
            args.fix_initial_steps,
        )
    user_fixed_steps = mark_user_fixed_step_sources(
        initial_pose_sources,
        step_sequence,
        step_index,
        args.fix_initial_steps,
    )
    if user_fixed_steps:
        print(
            "  fixing user-specified initialized poses: "
            + ", ".join(
                f"step {step} -> stone {stone_id}"
                for step, stone_id in user_fixed_steps
            )
        )
    n_prior = sum(1 for source in initial_pose_sources.values() if source == "prior")
    n_state = sum(
        1 for source in initial_pose_sources.values() if source.startswith("plan_state")
    )
    n_plan = len(planned_entries) - n_prior - n_state
    if n_prior or n_state:
        parts = []
        if n_prior:
            parts.append(f"{n_prior} prior poses")
        if n_state:
            parts.append(f"{n_state} state.pkl poses")
        if n_plan:
            parts.append(f"{n_plan} action_sequence poses")
        print("  recursive init: using " + ", ".join(parts))
    else:
        print(
            "  recursive init: using action_sequence poses for all "
            f"{len(planned_entries)} stones"
        )

    if not args.no_crop:
        scene_pcd = crop_scene_pcd_near_expected_poses(
            scene_pcd,
            scene_pcd_path or scene_dir,
            planned_entries,
            visual_meshes,
            args.crop_margin,
            args.crop_max_margin,
        )
    visualization_pcd = o3d.geometry.PointCloud(scene_pcd)
    if not args.no_ground_removal:
        filtered_by_model = (
            None
            if args.no_ground_height_init
            else remove_ground_points_by_plane_model(
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

    (
        planned_entries,
        sceneid_stone_point_counts,
        skipped_low_scene_points,
    ) = filter_sceneid_entries_by_point_support(
        scene_pcd,
        planned_entries,
        initial_pose_sources,
        visual_meshes,
        args,
    )
    if not planned_entries:
        print("  SceneID skipped: no initialized stones have enough nearby scene points")
        write_results(
            scene_dir,
            args.output_prefix,
            plan_dir,
            action_path,
            step_index,
            scene_pcd_path,
            {},
            {},
            initial_pose_sources,
            {},
            None if args.no_ground_height_init else calibrated_ground_height,
            None if args.no_ground_height_init else calibrated_ground_plane_model,
            args.skip_sceneid_solve,
            0.0,
            sceneid_stone_point_counts,
            skipped_low_scene_points,
        )
        return (
            False,
            {},
            calibrated_ground_height,
            calibrated_ground_plane_model,
        )

    planned_entries = refine_plan_initial_poses_with_icp(
        scene_pcd,
        planned_entries,
        initial_pose_sources,
        stone_pcds,
        visual_meshes,
        args,
        icp_prior_fallback_poses,
    )

    scene_pcd, points = downsample_and_select_points(
        scene_pcd,
        scene_pcd_path or scene_dir,
        args.voxel_size,
        args.max_points,
    )
    if args.visualize_solver_pcd:
        visualization_pcd = scene_pcd
    print(f"  scene points passed to sceneid: {points.shape[1]}")

    if args.skip_sceneid_solve:
        sceneid_to_stone = {}
        initial_poses = {}
        solution_poses = {}
        optimal_poses_by_stone = {}
        for body_id, (stone_id, pose) in enumerate(planned_entries, start=1):
            pose = pose.copy()
            sceneid_to_stone[body_id] = stone_id
            initial_poses[body_id] = pose
            solution_poses[body_id] = PoseVector(pose)
            optimal_poses_by_stone[stone_id] = pose
        elapsed = 0.0
        print("  skipped diffsimpy sceneid solve; using initialized poses as output")
    else:
        context, config = get_sceneid(
            ground_height=(
                None if args.no_ground_height_init else calibrated_ground_height
            )
        )
        apply_sceneid_args_to_config(config, args)
        set_sceneid_ground_height(
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
        optimal_poses_by_stone = pose_map_from_solution(solution, sceneid_to_stone)

    solution_poses, optimal_poses_by_stone = fix_locked_solution_poses(
        sceneid_to_stone,
        initial_poses,
        initial_pose_sources,
        solution_poses,
        optimal_poses_by_stone,
        args,
    )

    print(f"  solve elapsed: {elapsed:.3f}s")
    for stone_id in sorted(optimal_poses_by_stone):
        print(f"    stone {stone_id}: {optimal_poses_by_stone[stone_id]}")

    write_results(
        scene_dir,
        args.output_prefix,
        plan_dir,
        action_path,
        step_index,
        scene_pcd_path,
        sceneid_to_stone,
        initial_poses,
        initial_pose_sources,
        optimal_poses_by_stone,
        None if args.no_ground_height_init else calibrated_ground_height,
        None if args.no_ground_height_init else calibrated_ground_plane_model,
        args.skip_sceneid_solve,
        elapsed,
        sceneid_stone_point_counts,
        skipped_low_scene_points,
    )

    if args.visualize:
        print(f"  visualization scene points: {len(visualization_pcd.points)}")
        visualize_result(
            visualization_pcd,
            visual_meshes,
            sceneid_to_stone,
            initial_poses,
            solution_poses,
            None if args.no_ground_height_init else calibrated_ground_plane_model,
        )
    return (
        True,
        optimal_poses_by_stone,
        calibrated_ground_height,
        calibrated_ground_plane_model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify placed stone poses in execution logs using sceneid, with "
            "the corresponding plan's planned placement poses as initialization."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Log root, step_X directory, or scene_scan directory.",
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=None,
        help="Override plan directory containing action_sequence.pkl.",
    )
    parser.add_argument("--asset-dir", default=None)
    parser.add_argument("--output-prefix", default="sceneid_scene_poses")
    parser.add_argument("--merged-name", default="scene_scan_merged_for_sceneid.ply")
    parser.add_argument("--remerge", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--merge-voxel-size", type=float, default=0.0)
    parser.add_argument(
        "--compare-pcd-steps",
        nargs="?",
        const="all",
        default=None,
        help=(
            "Color and compare merged scene PCDs without running sceneid. "
            "Use comma-separated step indices such as 4,6, or omit the value "
            "to compare all scene_scan directories under the root."
        ),
    )
    parser.add_argument(
        "--compare-pcd-output",
        type=Path,
        default=None,
        help="Optional path to write the colored comparison PCD.",
    )
    parser.add_argument(
        "--compare-pcd-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open an Open3D viewer for --compare-pcd-steps.",
    )
    parser.add_argument(
        "--compare-pcd-bboxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show colored axis-aligned bounding boxes in the comparison viewer.",
    )
    parser.add_argument(
        "--compare-pcd-point-size",
        type=float,
        default=2.0,
        help="Open3D point size used by --compare-pcd-steps.",
    )
    parser.add_argument(
        "--compare-pcd-max-points",
        type=int,
        default=0,
        help=(
            "Maximum displayed/saved points per compared step. "
            "Use 0 to keep all points."
        ),
    )
    parser.add_argument(
        "--compare-pcd-ground-removal",
        action="store_true",
        help="Apply RANSAC ground removal to each compared PCD before coloring.",
    )
    parser.add_argument(
        "--no-recursive-init",
        action="store_true",
        help="Always initialize from planned poses instead of prior identified step poses.",
    )
    parser.add_argument(
        "--recursive-cache-scope",
        choices=("plan", "exec"),
        default="plan",
        help=(
            "Reuse identified poses across the whole plan by default, so a later "
            "exec can continue from earlier exec scene scans. Use 'exec' for the "
            "old per-exec cache behavior."
        ),
    )
    parser.add_argument(
        "--no-fix-missing-prior-steps",
        action="store_true",
        help=(
            "If an earlier step in the same exec has no scene_scan, allow that "
            "step's stone to be optimized instead of fixing it at the planned pose."
        ),
    )
    parser.add_argument(
        "--fix-planned-steps",
        default="",
        help=(
            "Comma-separated step indices whose placed stones should be fixed at "
            "their planned poses in later sceneid steps, e.g. 5."
        ),
    )
    parser.add_argument(
        "--fix-initial-steps",
        default="",
        help=(
            "Comma-separated placed step indices to lock at their initialized "
            "poses for this scene scan. The initialized pose may come from "
            "state.pkl, recursive prior, manual input, or action_sequence.pkl."
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
        "--manual-init-poses",
        type=Path,
        default=None,
        help=(
            "JSON or pickle file with step-indexed manual initial poses in scene "
            "coordinates. Supports {'steps': {'5': [x,y,z,qx,qy,qz,qw]}} or "
            "{'5': [x,y,z,qx,qy,qz,qw]}."
        ),
    )
    parser.add_argument(
        "--manual-init-step-pose",
        action="append",
        default=[],
        metavar="STEP:POSE",
        help=(
            "Manual initial pose for the stone placed at STEP, e.g. "
            "'5:x,y,z,qx,qy,qz,qw'. Repeat for multiple steps."
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
        "--manual-init-gui",
        action="store_true",
        help=(
            "Open an Open3D keyboard pose editor before crop/ICP/sceneid. "
            "Edited poses are used as manual initialization for this run."
        ),
    )
    parser.add_argument(
        "--manual-init-gui-output",
        type=Path,
        default=None,
        help=(
            "Optional JSON file to write edited manual GUI poses so they can be "
            "reused later with --manual-init-poses."
        ),
    )
    parser.add_argument(
        "--manual-init-gui-translation-step",
        type=float,
        default=0.01,
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
    parser.add_argument("--no-offset", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--crop-margin", type=float, default=0.5)
    parser.add_argument("--crop-max-margin", type=float, default=3.0)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--no-ground-removal", action="store_true")
    parser.add_argument("--ground-distance-threshold", type=float, default=0.1)
    parser.add_argument("--ground-ransac-n", type=int, default=3)
    parser.add_argument("--ground-num-iterations", type=int, default=1000)
    parser.add_argument("--ground-normal-z-min", type=float, default=0.7)
    parser.add_argument("--ground-min-inlier-ratio", type=float, default=0.02)
    parser.add_argument(
        "--no-ground-height-init",
        action="store_true",
        help=(
            "Do not estimate a ground-plane z offset from the first step. "
            "By default plan-seeded poses get this z offset before ICP/sceneid."
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
        "--no-icp-init",
        action="store_true",
        help="Skip Open3D ICP initialization before sceneid.",
    )
    parser.add_argument(
        "--skip-sceneid-solve",
        action="store_true",
        help="Stop after ground/ICP initialization and write those poses as output.",
    )
    parser.add_argument(
        "--icp-plan-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run ICP only for poses initialized from the plan or manual overrides. "
            "Recursive prior poses from earlier steps are not ICP-refined unless "
            "--icp-prior-poses is set."
        ),
    )
    parser.add_argument(
        "--icp-prior-poses",
        action="store_true",
        help=(
            "Also run ICP on previous-step stones, including recursive prior "
            "poses and fixed missing-step plan/state poses. Manual-fixed poses "
            "remain locked."
        ),
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
    parser.add_argument("--icp-min-target-points", type=int, default=50)
    parser.add_argument("--icp-min-correspondences", type=int, default=20)
    parser.add_argument("--icp-min-fitness", type=float, default=0.0)
    parser.add_argument(
        "--icp-min-score-gain",
        type=float,
        default=0.0,
        help=(
            "Require the best ICP hypothesis score to improve over the initial "
            "pose score by at least this amount. This prevents ICP from moving "
            "away from an already good initialization without evidence."
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
            "degrees / 180. Defaults to 0 so 90-degree orientation hypotheses "
            "remain usable unless explicitly regularized."
        ),
    )
    parser.add_argument(
        "--icp-prior-min-correspondences",
        type=int,
        default=None,
        help=(
            "Minimum final-scale correspondences for ICP-refined recursive "
            "prior poses. Defaults to --icp-min-correspondences."
        ),
    )
    parser.add_argument(
        "--icp-prior-min-fitness",
        type=float,
        default=None,
        help=(
            "Minimum final-scale fitness for ICP-refined recursive prior poses. "
            "Defaults to --icp-min-fitness."
        ),
    )
    parser.add_argument("--icp-max-translation", type=float, default=0.4)
    parser.add_argument("--icp-rmse-weight", type=float, default=0.25)
    parser.add_argument(
        "--sceneid-min-stone-points",
        type=int,
        default=1000,
        help=(
            "Skip a stone from SceneID when its initialized mesh bbox, expanded "
            "by --sceneid-stone-point-margin, contains fewer non-ground scene "
            "points. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--sceneid-stone-point-margin",
        type=float,
        default=0.35,
        help="Margin around each initialized stone mesh bbox for the SceneID point-count filter.",
    )
    parser.add_argument(
        "--no-icp-empty-target-prior-fallback",
        action="store_true",
        help=(
            "Disable fallback to the latest previous identified pose when a "
            "plan-initialized stone has too few nearby scene points for ICP."
        ),
    )
    parser.add_argument(
        "--no-icp-low-correspondence-prior-fallback",
        action="store_true",
        help=(
            "Disable fixing a stone to its latest previous identified pose when "
            "all ICP orientation hypotheses have too few correspondences."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--graph-max-iter", type=int, default=100)
    parser.add_argument("--trust-region-eps", type=float, default=0.1)
    parser.add_argument("--delta-init", type=float, default=0.125)
    parser.add_argument("--k-pcd", type=float, default=5.0)
    parser.add_argument("--pcd-huber-delta", type=float, default=0.02)
    parser.add_argument("--pcd-max-gap", type=float, default=0.15)
    parser.add_argument("--k-gap-c", type=float, default=30.0)
    parser.add_argument("--k-comp", type=float, default=0.0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--visualize-solver-pcd",
        action="store_true",
        help=(
            "Visualize the final filtered scene PCD passed to sceneid. "
            "By default --visualize shows the cropped ground-filtered PCD."
        ),
    )
    parser.add_argument(
        "--reconstruct-state",
        action="store_true",
        help=(
            "After identifying logs, write a generate_sequence.py resume-ready "
            f"{DEFAULT_RECONSTRUCTED_STATE_NAME} for the latest successful scene "
            "in each plan."
        ),
    )
    parser.add_argument(
        "--state-output",
        type=Path,
        default=None,
        help=(
            "Optional output path for --reconstruct-state. Only valid when the "
            "log root resolves to one plan; otherwise each plan writes its own "
            f"{DEFAULT_RECONSTRUCTED_STATE_NAME}."
        ),
    )
    parser.add_argument(
        "--overwrite-state",
        action="store_true",
        help=(
            "With --reconstruct-state and no --state-output, write state.pkl "
            f"instead of {DEFAULT_RECONSTRUCTED_STATE_NAME}. This replaces the "
            "plan's primary debug/resume state pointer."
        ),
    )
    parser.add_argument(
        "--no-mark-pose-identified",
        action="store_true",
        help=(
            "Do not label reconstructed placed stones as scene-pose-identified "
            "in the resume State."
        ),
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
    args.fix_planned_steps = parse_int_list(args.fix_planned_steps)
    args.fix_initial_steps = parse_int_list(args.fix_initial_steps)
    args.manual_init_by_step = collect_manual_init_poses(args)
    if args.compare_pcd_steps is None:
        args.compare_pcd_step_indices = None
    elif str(args.compare_pcd_steps).strip().lower() in ("", "all"):
        args.compare_pcd_step_indices = None
    else:
        args.compare_pcd_step_indices = set(parse_int_list(args.compare_pcd_steps))
    if args.compare_pcd_max_points < 0:
        raise ValueError("--compare-pcd-max-points must be >= 0")
    if args.manual_init_gui_translation_step <= 0.0:
        raise ValueError("--manual-init-gui-translation-step must be > 0")
    if args.manual_init_gui_rotation_step_deg <= 0.0:
        raise ValueError("--manual-init-gui-rotation-step-deg must be > 0")
    if args.manual_init_gui_point_size <= 0.0:
        raise ValueError("--manual-init-gui-point-size must be > 0")
    if args.sceneid_min_stone_points < 0:
        raise ValueError("--sceneid-min-stone-points must be >= 0")
    if args.sceneid_stone_point_margin < 0.0:
        raise ValueError("--sceneid-stone-point-margin must be >= 0")
    if len(args.icp_voxel_sizes) != len(args.icp_max_iters):
        raise ValueError("--icp-voxel-sizes and --icp-max-iters must have same length")
    if not args.icp_voxel_sizes:
        raise ValueError("--icp-voxel-sizes must contain at least one value")
    if any(value <= 0.0 for value in args.icp_voxel_sizes):
        raise ValueError("--icp-voxel-sizes values must be > 0")
    if any(value <= 0 for value in args.icp_max_iters):
        raise ValueError("--icp-max-iters values must be > 0")
    if args.icp_correspondence_distance_scale <= 0.0:
        raise ValueError("--icp-correspondence-distance-scale must be > 0")
    if args.icp_translation_weight < 0.0:
        raise ValueError("--icp-translation-weight must be >= 0")
    if args.icp_rotation_weight < 0.0:
        raise ValueError("--icp-rotation-weight must be >= 0")
    if (
        args.icp_prior_min_correspondences is not None
        and args.icp_prior_min_correspondences < 0
    ):
        raise ValueError("--icp-prior-min-correspondences must be >= 0")
    if args.icp_prior_min_fitness is not None and args.icp_prior_min_fitness < 0.0:
        raise ValueError("--icp-prior-min-fitness must be >= 0")
    return args


if __name__ == "__main__":
    args = parse_args()
    dirs = scene_scan_dirs(args.root.resolve())
    if not dirs:
        raise FileNotFoundError(f"no scene_scan directories found under {args.root}")

    if args.compare_pcd_steps is not None:
        n_compared = compare_scene_pcds(dirs, args)
        print(f"\nCompared {n_compared} scene_scan directories.")
        sys.exit(0)

    print(f"Recursive pose cache scope: {args.recursive_cache_scope}")

    n_ok = 0
    prior_by_cache: dict[Path, dict[int, np.ndarray]] = {}
    ground_height_by_cache: dict[Path, float | None] = {}
    ground_plane_by_cache: dict[Path, np.ndarray | None] = {}
    identified_records = []
    for scene_dir in dirs:
        exec_dir = exec_dir_from_scene_dir(scene_dir)
        cache_key = recursive_cache_key(scene_dir, args)
        prior_poses = prior_by_cache.setdefault(cache_key, {})
        try:
            ok, optimal_poses, ground_height, ground_plane_model = identify_scene_dir(
                scene_dir,
                args,
                prior_poses,
                ground_height_by_cache.get(cache_key),
                ground_plane_by_cache.get(cache_key),
            )
            if ok:
                n_ok += 1
                plan_dir = resolve_plan_dir(scene_dir, args.plan_dir)
                identified_records.append(
                    {
                        "scene_dir": scene_dir,
                        "exec_dir": exec_dir,
                        "plan_dir": plan_dir,
                        "step_index": step_index_from_scene_dir(scene_dir),
                        "optimal_poses": {
                            int(stone_id): pose.copy()
                            for stone_id, pose in optimal_poses.items()
                        },
                        "ground_height": ground_height,
                        "ground_plane_model": (
                            ground_plane_model.copy()
                            if ground_plane_model is not None
                            else None
                        ),
                    }
                )
                if cache_key not in ground_height_by_cache:
                    ground_height_by_cache[cache_key] = ground_height
                    ground_plane_by_cache[cache_key] = ground_plane_model
                if not args.no_recursive_init:
                    prior_poses.update(
                        {
                            stone_id: pose.copy()
                            for stone_id, pose in optimal_poses.items()
                        }
                    )
        except Exception as exc:
            print(f"  failed: {exc}")

    print(f"\nIdentified {n_ok} / {len(dirs)} scene_scan directories.")

    if args.reconstruct_state:
        latest_by_plan = {}
        for record in identified_records:
            plan_dir = Path(record["plan_dir"])
            current = latest_by_plan.get(plan_dir)
            if current is None:
                latest_by_plan[plan_dir] = record
                continue
            current_key = (
                int(current["step_index"]),
                natural_key(Path(current["scene_dir"])),
            )
            record_key = (
                int(record["step_index"]),
                natural_key(Path(record["scene_dir"])),
            )
            if record_key > current_key:
                latest_by_plan[plan_dir] = record

        if args.state_output is not None and len(latest_by_plan) > 1:
            raise ValueError(
                "--state-output can only be used when identified scenes belong "
                "to one plan"
            )

        for plan_dir, record in sorted(
            latest_by_plan.items(), key=lambda item: natural_key(item[0])
        ):
            output_path = (
                args.state_output.resolve()
                if args.state_output is not None
                else plan_dir
                / (
                    DEFAULT_PLAN_STATE_NAME
                    if args.overwrite_state
                    else DEFAULT_RECONSTRUCTED_STATE_NAME
                )
            )
            write_reconstructed_state_pkl(record, args, output_path)
