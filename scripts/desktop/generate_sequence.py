#!/usr/bin/env python3
import argparse
import datetime
import os
import shutil
import sys
import threading
import time
from omegaconf import OmegaConf
import pickle
import copy
import numpy as np
from scipy.spatial.transform import Rotation

from agent import IntegratedPlanner
from agent.env.components.contexts import (
    environment_ground_height,
    get_diffsim,
    set_environment_ground_height,
)
from agent.env.components.action.floor_fill import (
    LEGACY_GROUND_FILL_REJECT_STACKED_KEY,
    LOWER_FLOOR_FILL_REJECT_STACKED_KEY,
    active_floor_context,
    active_layer_fill_metrics,
    lower_floor_fill_reject_stacked,
    migrate_lower_floor_fill_reject_stacked,
)
from agent.config_views import support_config
from agent.mcts.utils import solved_action_pose
from scripts.debug.mcts_map_images import state_score_map_debug
from model import get_stone_model, get_excavator_model
from planning import (
    get_planner,
    motion_failure_can_retry_regrasp_xy,
    motion_failure_detail,
    motion_failure_summary,
    motion_failure_stage,
    motion_result_summary,
    regrasp_planning,
    regrasp_position_candidates,
    split_regrasp_place_paths,
    trajectory_visualization_with_target,
    generate_path_with_opening_angle,
    Q_HOME,
)
from utils import get_unique_dir, resolve_thread_count

STONE_RENDER_MESH_SOURCE = "dsf"  # "dsf" or "mesh"
TARGET_STRUCTURE_OFFSET = np.array([-6.0, 3.0])
REGRASP_XY_POS = np.array([-2.0, -5.0])
PICK_PLANE_HEIGHT = 0.0
PLACE_PLANE_HEIGHT = 0.0
PLACE_Z_OFFSET = 0.20
INHAND_REPLAN_MODE = "regrasp"  # "direct" or "regrasp"
PLACE_HEIGHT_BOX_MARGIN = 0.25
PLACE_HEIGHT_BOX_MIN_THICKNESS = 0.04
DEFAULT_GROUND_BASE_COLOR = [0.5, 0.7, 0.3, 1.0]
GROUND_FILL_STACK_REJECT_MARGIN = 0.08
STONE_REJECTION_THRESHOLD = 2
INITIAL_PREVIEW_TIMEOUT_SECONDS = 60.0

N_MOVE = 15
N_GRASP = 10
N_OPENING_ANGLE = 10

# ---------------------------------------------------------------------------
# Debug-state helpers — output is readable by scripts/debug/candidate_viewer.py
# ---------------------------------------------------------------------------


def _serializable_info(info) -> dict:
    if not info:
        return {}
    out = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            out[key] = value.astype(float).tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, (float, int, bool, str)) or value is None:
            out[key] = value
        else:
            try:
                out[key] = float(value)
            except Exception:
                out[key] = str(value)
    return out


def _candidate_diagnostics(action) -> tuple[dict, list]:
    diagnostics = dict(getattr(action, "diagnostics", {}) or {})
    pose_solve_contacts = diagnostics.pop("pose_solve_contacts", []) or []
    return _serializable_info(diagnostics), list(pose_solve_contacts)


def _pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float).reshape(-1)
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    return T


def _mesh_vertices_at_pose(mesh, pose: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        return np.empty((0, 3), dtype=float)
    T = _pose_to_matrix(pose)
    return vertices @ T[:3, :3].T + T[:3, 3]


def _target_xy_bounds(cfg) -> tuple[np.ndarray, np.ndarray]:
    target = cfg.environment.target
    origin = np.asarray(target.get("origin", [0.0, 0.0])[:2], dtype=float)
    half = np.asarray([float(target.width), float(target.length)], dtype=float) / 2.0
    return origin - half, origin + half


def _aabb_xy_occupancy(
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    lower: np.ndarray,
    upper: np.ndarray,
    grid_size: int,
) -> float:
    if not aabbs or np.any(upper <= lower):
        return 0.0
    grid_size = max(int(grid_size), 4)
    xs = np.linspace(lower[0], upper[0], grid_size)
    ys = np.linspace(lower[1], upper[1], grid_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    occupied = np.zeros(xx.shape, dtype=bool)
    for aabb_lower, aabb_upper in aabbs:
        aabb_lower = np.maximum(np.asarray(aabb_lower, dtype=float), lower)
        aabb_upper = np.minimum(np.asarray(aabb_upper, dtype=float), upper)
        if np.any(aabb_upper < aabb_lower):
            continue
        occupied |= (
            (xx >= aabb_lower[0])
            & (xx <= aabb_upper[0])
            & (yy >= aabb_lower[1])
            & (yy <= aabb_upper[1])
        )
    return float(np.mean(occupied))


def _ground_fill_status(state, stone_meshes, cfg) -> dict:
    floor_fill_cfg = cfg.environment.action.planar.get("floor_fill", {})
    support = support_config(cfg)
    ground_z = support.ground_z
    bottom_tol = float(
        floor_fill_cfg.get(
            "lower_floor_bottom_z_tolerance",
            support.z_tolerance,
        )
    )
    fill_ratio = float(floor_fill_cfg.get("lower_floor_fill_ratio", 0.90))
    grid_size = int(floor_fill_cfg.get("lower_floor_occupancy_grid", 32))
    lower, upper = _target_xy_bounds(cfg)

    ground_aabbs = []
    ground_tops = []
    for placed_idx in getattr(state, "stone_seq", []) or []:
        try:
            stone_id = int(state.stone_set[int(placed_idx)])
        except (TypeError, ValueError, IndexError):
            continue
        pose = state.stone_poses.get(stone_id)
        mesh = stone_meshes.get(stone_id)
        if pose is None or mesh is None:
            continue
        vertices = _mesh_vertices_at_pose(mesh, pose)
        if vertices.size == 0:
            continue
        bottom = float(np.min(vertices[:, 2]))
        top = float(np.max(vertices[:, 2]))
        if bottom > ground_z + bottom_tol:
            continue
        ground_tops.append(top)
        ground_aabbs.append(
            (
                np.min(vertices[:, :2], axis=0),
                np.max(vertices[:, :2], axis=0),
            )
        )

    occupancy = _aabb_xy_occupancy(ground_aabbs, lower, upper, grid_size)
    return {
        "done": occupancy >= fill_ratio,
        "occupancy": occupancy,
        "required": fill_ratio,
        "ground_top": max(ground_tops) if ground_tops else ground_z,
        "ground_z": ground_z,
    }


def _candidate_stacks_before_ground_fill(
    state,
    target_id: int,
    candidate_pose: np.ndarray,
    stone_meshes,
    cfg,
) -> tuple[bool, dict]:
    floor_fill_cfg = cfg.environment.action.planar.get("floor_fill", {})
    if not lower_floor_fill_reject_stacked(floor_fill_cfg):
        return False, {}
    if len(getattr(state, "stone_seq", []) or []) == 0:
        return False, {}

    status = _ground_fill_status(state, stone_meshes, cfg)
    if status["done"]:
        return False, status

    mesh = stone_meshes.get(int(target_id))
    if mesh is None:
        return False, status
    vertices = _mesh_vertices_at_pose(mesh, candidate_pose)
    if vertices.size == 0:
        return False, status
    candidate_bottom = float(np.min(vertices[:, 2]))
    margin = float(
        floor_fill_cfg.get(
            "ground_fill_stack_reject_margin",
            GROUND_FILL_STACK_REJECT_MARGIN,
        )
    )
    reject = candidate_bottom > status["ground_top"] + margin
    status["candidate_bottom"] = candidate_bottom
    status["margin"] = margin
    return reject, status


def _candidate_rejects_active_floor(
    env,
    state,
    target_id: int,
    candidate_pose: np.ndarray,
) -> tuple[bool, dict]:
    floor_fill_cfg = env.cfg.action.planar.get("floor_fill", {})
    if not lower_floor_fill_reject_stacked(floor_fill_cfg):
        return False, {}

    layer = active_floor_context(env.inventory, state)
    if layer is None:
        return False, {}

    stone_set = np.asarray(env.inventory.stone_set, dtype=int).reshape(-1)
    matches = np.flatnonzero(stone_set == int(target_id))
    if len(matches) == 0:
        return False, {}
    stone_idx = int(matches[0])

    metrics = active_layer_fill_metrics(
        env.inventory,
        layer,
        stone_idx,
        candidate_pose,
    )
    min_contact = max(
        int(floor_fill_cfg.get("u_shape_min_frontier_contact_cells", 1)),
        0,
    )
    status = {
        "fixed_scene_active_layer_fill_score": float(metrics["fill_score"]),
        "fixed_scene_active_layer_above": bool(metrics["above_active_layer"]),
        "fixed_scene_active_layer_contact_cells": int(metrics["contact_cells"]),
        "fixed_scene_active_layer_min_contact_cells": int(min_contact),
        "fixed_scene_active_layer_unfilled_cells": int(metrics["unfilled_cells"]),
        "fixed_scene_active_layer_overlap_cells": int(metrics["overlap_cells"]),
        "fixed_scene_active_layer_occupancy": float(layer["occupancy"]),
    }

    if bool(metrics["above_active_layer"]):
        status["active_floor_reject_reason"] = "above_active_layer"
        return True, status
    if (
        min_contact > 0
        and float(metrics["fill_score"]) > 0.0
        and int(metrics["contact_cells"]) < min_contact
    ):
        status["active_floor_reject_reason"] = "disconnected_active_layer"
        return True, status
    return False, status


def _node_score(node) -> float:
    info = getattr(node, "info", None) or {}
    prior_score = info.get("final_validation_prior_score", None)
    if prior_score is not None:
        try:
            prior_score = float(prior_score)
        except (TypeError, ValueError):
            prior_score = -np.inf
        if np.isfinite(prior_score):
            return prior_score
    if node.failed:
        return -np.inf
    score = float(node.q_value_init)
    if node.is_simulated and node.visits > 0 and np.isfinite(node.q_value):
        score = node.q_value / max(node.visits, 1.0) + float(node.q_value_init)
    elif node.is_simulated:
        reward = 0.0 if node.reward is None else float(node.reward)
        score = reward + float(node.q_value_init) + float(node.value_init)
    return score


def _candidate_final_pose(node) -> np.ndarray | None:
    """Pose the candidate stone reaches in its latest stored simulation."""
    state = node.state
    action = node.action
    if state is None or action is None:
        return None
    pose = getattr(state, "stone_poses", {}).get(int(action.stone_id))
    if pose is None:
        return None
    arr = np.asarray(pose, dtype=float)
    if arr.shape[0] >= 7 and np.all(np.isfinite(arr[:7])):
        return arr[:7].copy()
    return None


def _same_candidate_node(node, selected_node) -> bool:
    if selected_node is None:
        return False
    if node is selected_node:
        return True
    action = getattr(node, "action", None)
    selected_action = getattr(selected_node, "action", None)
    if action is None or selected_action is None:
        return False
    if int(action.stone_id) != int(selected_action.stone_id):
        return False
    if int(action.stone_idx) != int(selected_action.stone_idx):
        return False
    return np.allclose(
        np.asarray(action.pose, dtype=float),
        np.asarray(selected_action.pose, dtype=float),
        equal_nan=True,
    )


def _candidate_trajectory(node) -> list:
    if node.state is None or not getattr(node.state, "trajectories", None):
        return []
    action = node.action
    if action is None:
        return []
    stone_id = int(action.stone_id)
    latest = node.state.trajectories[-1]
    traj = latest.get(stone_id, None)
    if traj is None:
        return []
    poses = []
    for pose in getattr(traj, "poses", []):
        arr = np.asarray(pose, dtype=float)
        if arr.shape[0] >= 7 and np.all(np.isfinite(arr[:7])):
            poses.append(arr[:7].copy())
    return poses


def _candidate_scene_motion(node) -> dict | None:
    value = getattr(node, "_final_validation_scene_motion", None)
    if not isinstance(value, dict):
        return None
    poses = []
    for pose in value.get("trajectory", []):
        pose = np.asarray(pose, dtype=float)
        if pose.shape[0] >= 7 and np.all(np.isfinite(pose[:7])):
            poses.append(pose[:7].copy())
    return {
        "stone_id": int(value.get("stone_id", -1)),
        "velocity_integral": float(value.get("velocity_integral", 0.0)),
        "trajectory": poses,
    }


def _candidate_contact_points(node) -> list:
    state = node.state
    action = node.action
    if state is None or action is None:
        return []
    contacts = []
    for contact in getattr(state, "contact_points", []) or []:
        item = {}
        for key, value in contact.items():
            if isinstance(value, np.ndarray):
                item[key] = value.astype(float).copy()
            elif isinstance(value, np.generic):
                item[key] = value.item()
            else:
                item[key] = value
        if (
            item.get("stone_idx_1") == action.stone_idx
            or item.get("stone_idx_2") == action.stone_idx
        ):
            contacts.append(item)
    return contacts


def _pose_to_matrix_array(pose):
    if hasattr(pose, "as_matrix"):
        matrix = pose.as_matrix()
    else:
        matrix = pose
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape == (4, 4) and np.all(np.isfinite(matrix)):
        return matrix.copy()
    return None


def _matrix_rotation_error_deg(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    rot_delta = matrix_a[:3, :3].T @ matrix_b[:3, :3]
    return float(np.rad2deg(Rotation.from_matrix(rot_delta).magnitude()))


def _global_pose_from_state_pose(
    pose: np.ndarray, target_structure_offset
) -> np.ndarray:
    out = np.asarray(pose, dtype=float).reshape(-1)[:7].copy()
    out[:2] += np.asarray(target_structure_offset, dtype=float).reshape(2)
    return out


def _place_config_from_state_pose(
    stone_config,
    pose: np.ndarray,
    target_structure_offset,
):
    global_pose = _global_pose_from_state_pose(pose, target_structure_offset)
    if global_pose.shape[0] != 7 or not np.all(np.isfinite(global_pose[:7])):
        raise ValueError("state pose must be a finite 7-vector")
    config = copy.deepcopy(stone_config)
    config.pose.setPosition(global_pose[:3])
    config.pose.setOrientation(global_pose[3:])
    return config


def _config_pose_vector(config) -> np.ndarray:
    pose = config.pose
    if hasattr(pose, "vectorized"):
        return np.asarray(pose.vectorized(), dtype=float).reshape(-1)[:7].copy()
    return np.concatenate(
        [
            np.asarray(pose.position(), dtype=float).reshape(3),
            np.asarray(pose.orientation(), dtype=float).reshape(4),
        ]
    )


def _long_sim_motion_scene_check(
    parent_state,
    final_state,
    target_id: int,
    place_config,
    scene_configs: dict,
    target_structure_offset,
) -> dict:
    position_tol = 1e-4
    rotation_tol_deg = 0.1
    stone_set = np.asarray(parent_state.stone_set, dtype=int).reshape(-1)
    placed_ids = [
        int(stone_set[int(idx)])
        for idx in getattr(parent_state, "stone_seq", []) or []
        if 0 <= int(idx) < len(stone_set)
    ]

    scene_max_pos = 0.0
    scene_max_rot = 0.0
    scene_worst_pos_id = None
    scene_worst_rot_id = None
    compared = 0
    for stone_id in placed_ids:
        config = scene_configs.get(stone_id)
        final_pose = getattr(final_state, "stone_poses", {}).get(stone_id)
        if config is None or final_pose is None:
            continue
        scene_matrix = _pose_to_matrix_array(config.pose)
        if scene_matrix is None:
            continue
        final_global_pose = _global_pose_from_state_pose(
            final_pose,
            target_structure_offset,
        )
        final_matrix = _pose_to_matrix(final_global_pose)
        pos_err = float(np.linalg.norm(scene_matrix[:3, 3] - final_matrix[:3, 3]))
        rot_err = _matrix_rotation_error_deg(scene_matrix, final_matrix)
        compared += 1
        if pos_err > scene_max_pos:
            scene_max_pos = pos_err
            scene_worst_pos_id = stone_id
        if rot_err > scene_max_rot:
            scene_max_rot = rot_err
            scene_worst_rot_id = stone_id

    target_pose = getattr(final_state, "stone_poses", {}).get(int(target_id))
    target_pos_err = None
    target_rot_err = None
    if target_pose is not None:
        target_matrix = _pose_to_matrix(
            _global_pose_from_state_pose(target_pose, target_structure_offset)
        )
        place_matrix = _pose_to_matrix_array(place_config.pose)
        if place_matrix is not None:
            target_pos_err = float(
                np.linalg.norm(place_matrix[:3, 3] - target_matrix[:3, 3])
            )
            target_rot_err = _matrix_rotation_error_deg(place_matrix, target_matrix)

    scene_matches = scene_max_pos <= position_tol and scene_max_rot <= rotation_tol_deg
    target_matches = (
        target_pos_err is not None
        and target_rot_err is not None
        and target_pos_err <= position_tol
        and target_rot_err <= rotation_tol_deg
    )
    diagnostics = {
        "motion_scene_long_sim_compared": int(compared),
        "motion_scene_long_sim_max_position_error": float(scene_max_pos),
        "motion_scene_long_sim_max_rotation_error_deg": float(scene_max_rot),
        "motion_scene_long_sim_worst_position_stone_id": scene_worst_pos_id,
        "motion_scene_long_sim_worst_rotation_stone_id": scene_worst_rot_id,
        "motion_scene_matches_long_sim": bool(scene_matches),
        "target_config_long_sim_position_error": target_pos_err,
        "target_config_long_sim_rotation_error_deg": target_rot_err,
        "target_config_matches_long_sim": bool(target_matches),
    }

    if not scene_matches:
        print(
            "[WARN] Motion scene differs from long simulation scene: "
            f"max_pos={scene_max_pos:.4f}m stone={scene_worst_pos_id}, "
            f"max_rot={scene_max_rot:.2f}deg stone={scene_worst_rot_id}"
        )
    if not target_matches:
        print(
            "[WARN] Target place config differs from long simulation target: "
            f"pos_err={target_pos_err}, rot_err_deg={target_rot_err}"
        )
    return diagnostics


def _resettle_place_config_fixed_scene(
    cfg,
    parent_state,
    target_id: int,
    place_config,
    scene_configs: dict,
    target_structure_offset,
) -> tuple[np.ndarray | None, dict]:
    """Settle the target with the motion-planning scene fixed.

    MCTS long simulation may relax existing stones, but motion planning uses
    the fixed scene in ``scene_configs``. This re-settle keeps those scene stones
    frozen and updates only the target pose before it is handed to the planner.
    """
    settle_steps = max(int(cfg.environment.sim.get("settle_n_step", 0)), 1)
    dt = float(cfg.environment.sim.dt)
    ground_height = environment_ground_height(cfg.environment)
    sim, _ = get_diffsim(False, ground_height=ground_height)

    stone_set = np.asarray(parent_state.stone_set, dtype=int).reshape(-1)
    placed_ids = [
        int(stone_set[int(idx)])
        for idx in getattr(parent_state, "stone_seq", []) or []
        if 0 <= int(idx) < len(stone_set)
    ]

    frozen_body_ids = []
    for stone_id in placed_ids:
        config = scene_configs.get(stone_id)
        if config is None:
            continue
        body_id = sim.add_body(copy.deepcopy(config))
        sim.freeze_body(body_id)
        frozen_body_ids.append(body_id)

    target_body_id = sim.add_body(copy.deepcopy(place_config))
    start_pose = _config_pose_vector(place_config)
    start_time = time.perf_counter()
    for _ in range(settle_steps):
        sim.step(dt)
    elapsed = time.perf_counter() - start_time
    settled_pose = np.asarray(
        sim.state().pose(target_body_id).vectorized(), dtype=float
    )

    diagnostics = {
        "fixed_scene_resettle_enabled": True,
        "fixed_scene_resettle_steps": int(settle_steps),
        "fixed_scene_resettle_bodies": int(len(frozen_body_ids)),
        "fixed_scene_resettle_elapsed": float(elapsed),
        "fixed_scene_resettle_finite": bool(np.all(np.isfinite(settled_pose))),
    }
    if not np.all(np.isfinite(settled_pose)):
        diagnostics["fixed_scene_resettle_valid"] = False
        diagnostics["fixed_scene_resettle_reject_reason"] = "nonfinite"
        return None, diagnostics

    position_delta = float(np.linalg.norm(settled_pose[:3] - start_pose[:3]))
    xy_delta = float(np.linalg.norm(settled_pose[:2] - start_pose[:2]))
    rotation_delta = float(
        (
            Rotation.from_quat(start_pose[3:7]).inv()
            * Rotation.from_quat(settled_pose[3:7])
        ).magnitude()
    )
    diagnostics.update(
        {
            "fixed_scene_resettle_position_delta": position_delta,
            "fixed_scene_resettle_xy_delta": xy_delta,
            "fixed_scene_resettle_valid": True,
            "fixed_scene_resettle_rotation_delta_deg": float(
                np.rad2deg(rotation_delta)
            ),
        }
    )
    place_config.pose.setPosition(settled_pose[:3])
    place_config.pose.setOrientation(settled_pose[3:7])
    print(
        "[INFO] Re-settled target against fixed scene: "
        f"stone={target_id}, pos_delta={position_delta:.4f}m, "
        f"rot_delta={np.rad2deg(rotation_delta):.2f}deg, "
        f"steps={settle_steps}, elapsed={elapsed:.2f}s"
    )
    local_pose = settled_pose.copy()
    local_pose[:2] -= np.asarray(target_structure_offset, dtype=float).reshape(2)
    return local_pose, diagnostics


def _serializable_pose_path(path) -> list:
    out = []
    for pose in path or []:
        matrix = _pose_to_matrix_array(pose)
        if matrix is not None:
            out.append(matrix)
    return out


def _serializable_grasp(grasp) -> dict | None:
    pose = _pose_to_matrix_array(getattr(grasp, "pose", None))
    if pose is None:
        return None
    try:
        opening_angle = float(getattr(grasp, "opening_angle", 0.0))
    except (TypeError, ValueError):
        opening_angle = 0.0
    return {"pose": pose, "opening_angle": opening_angle}


def _serializable_failed_grasps(result, place_pick_transform=None) -> list:
    """Grasp poses rejected during grasp generation (for debug visualization).

    The planner returns these in the pick frame. When ``place_pick_transform``
    (``T_place_pick = place_pose @ pick_pose^-1``) is given, each grasp is mapped
    into the place frame so it renders at the candidate placement.
    """
    out = []
    for grasp in getattr(result, "failed_grasps", []) or []:
        item = _serializable_grasp(grasp)
        if item is None:
            continue
        if place_pick_transform is not None:
            item["pose"] = place_pick_transform @ item["pose"]
        out.append(item)
    return out


def _serializable_motion_result(result) -> dict:
    q_path_sequence = []
    for path in getattr(result, "q_path_sequence", []) or []:
        q_path = []
        for q in path:
            arr = np.asarray(q, dtype=float)
            if arr.ndim == 1 and arr.shape[0] >= 6 and np.all(np.isfinite(arr[:6])):
                q_path.append(arr[:6].copy())
        q_path_sequence.append(q_path)

    target_path_sequence = [
        _serializable_pose_path(path)
        for path in (getattr(result, "target_path_sequence", []) or [])
    ]
    settled_target_path_sequence = [
        _serializable_pose_path(path)
        for path in (getattr(result, "settled_target_path_sequence", []) or [])
    ]
    grasp_sequence = [
        item
        for item in (
            _serializable_grasp(grasp)
            for grasp in (getattr(result, "grasp_sequence", []) or [])
        )
        if item is not None
    ]
    scores = []
    for score in getattr(result, "scores", []) or []:
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            pass

    return {
        "is_feasible": bool(getattr(result, "is_feasible", False)),
        "q_path_sequence": q_path_sequence,
        "target_path_sequence": target_path_sequence,
        "settled_target_path_sequence": settled_target_path_sequence,
        "grasp_sequence": grasp_sequence,
        "scores": scores,
        "failure_stage": str(getattr(result, "failure_stage", "") or ""),
        "failure_detail": str(getattr(result, "failure_detail", "") or ""),
    }


def _candidate_best_sequence(node) -> list:
    sequence = getattr(node, "_debug_best_sequence", None)
    if not sequence:
        return []
    out = []
    for item in sequence:
        if not isinstance(item, dict):
            continue
        pose = np.asarray(item.get("pose", []), dtype=float)
        solved_pose = np.asarray(item.get("solved_pose", pose), dtype=float)
        settled_pose = np.asarray(item.get("settled_pose", pose), dtype=float)
        init_pose = np.asarray(item.get("init_pose", []), dtype=float)
        if pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
            continue
        out.append(
            {
                "stone_id": int(item.get("stone_id", -1)),
                "stone_idx": int(item.get("stone_idx", -1)),
                "pose": pose[:7].copy(),
                "solved_pose": (
                    solved_pose[:7].copy()
                    if solved_pose.shape[0] >= 7
                    and np.all(np.isfinite(solved_pose[:7]))
                    else pose[:7].copy()
                ),
                "settled_pose": (
                    settled_pose[:7].copy()
                    if settled_pose.shape[0] >= 7
                    and np.all(np.isfinite(settled_pose[:7]))
                    else pose[:7].copy()
                ),
                "init_pose": (
                    init_pose[:7].copy()
                    if init_pose.shape[0] >= 7 and np.all(np.isfinite(init_pose[:7]))
                    else pose[:7].copy()
                ),
                "reward": item.get("reward", None),
                "q_value": float(item.get("q_value", 0.0)),
                "q_value_init": float(item.get("q_value_init", 0.0)),
                "value_init": float(item.get("value_init", 0.0)),
                "visits": int(item.get("visits", 0)),
                "failed": bool(item.get("failed", False)),
                "info": _serializable_info(item.get("info", {})),
            }
        )
    return out


def _pose_identified_stone_ids(state) -> set[int]:
    return {
        int(stone_id)
        for stone_id in getattr(state, "pose_identified_stone_ids", set()) or set()
    }


def _target_wall_debug_payload(env) -> tuple[dict, list]:
    wall_meshes = []
    for geom in env.inventory.target_wall.geometries:
        m = geom.get_mesh()
        wall_meshes.append(
            (
                np.asarray(m.vertices, dtype=float).copy(),
                np.asarray(m.triangles, dtype=int).copy(),
            )
        )

    wall = env.inventory.target_wall
    wall_cfg = {
        "width": float(wall.width),
        "length": float(wall.length),
        "height": float(wall.height),
    }
    return wall_cfg, wall_meshes


def _init_debug_data(env) -> dict:
    stone_meshes = {}
    for idx in range(len(env.inventory.stone_set)):
        stone = env.inventory.stones[idx]
        all_v, all_t, offset = [], [], 0
        for geom in stone.geometries:
            m = geom.get_mesh()
            v = np.asarray(m.vertices, dtype=float)
            t = np.asarray(m.triangles, dtype=int)
            all_v.append(v)
            all_t.append(t + offset)
            offset += len(v)
        stone_meshes[int(stone.id)] = (
            np.concatenate(all_v, axis=0),
            np.concatenate(all_t, axis=0),
        )

    wall_cfg, wall_meshes = _target_wall_debug_payload(env)
    return {
        "target_wall_cfg": wall_cfg,
        "target_wall_meshes": wall_meshes,
        "stone_meshes": stone_meshes,
        "stone_ply_meshes": {},
        "mesh_source": "dsf",
        "steps": [],
    }


def _mesh_arrays(mesh):
    return (
        np.asarray(mesh.vertices, dtype=float).copy(),
        np.asarray(mesh.triangles, dtype=int).copy(),
    )


def _backfill_debug_stone_meshes(debug_data: dict, stone_meshes: dict) -> None:
    stored = debug_data.setdefault("stone_meshes", {})
    added = []
    for stone_id, mesh in stone_meshes.items():
        sid = int(stone_id)
        if sid in stored:
            continue
        stored[sid] = _mesh_arrays(mesh)
        added.append(sid)
    debug_data.setdefault("stone_ply_meshes", {})
    debug_data["mesh_source"] = "dsf"
    if added:
        print(f"[INFO] Backfilled debug stone meshes for ids: {sorted(added)}")


def _refresh_debug_target_wall(debug_data: dict, env, reason: str = "") -> None:
    wall_cfg, wall_meshes = _target_wall_debug_payload(env)
    old_cfg = debug_data.get("target_wall_cfg", {})
    debug_data["target_wall_cfg"] = wall_cfg
    debug_data["target_wall_meshes"] = wall_meshes
    if old_cfg != wall_cfg:
        suffix = f" ({reason})" if reason else ""
        print(
            "[INFO] Updated debug state target wall model"
            f"{suffix}: {old_cfg!r} -> {wall_cfg!r}"
        )


def _set_debug_coordinate_metadata(debug_data: dict, target_structure_offset) -> None:
    debug_data["target_structure_offset"] = np.asarray(
        target_structure_offset, dtype=float
    ).copy()
    debug_data["pose_coordinate_frame"] = "planner"


def _active_floor_debug(env, state) -> dict | None:
    layer = active_floor_context(env.inventory, state)
    if layer is None:
        return None

    floor_fill_cfg = env.cfg.action.planar.get("floor_fill", {})
    required = float(floor_fill_cfg.get("lower_floor_fill_ratio", 0.90))
    occupied = np.asarray(layer["occupied"], dtype=bool)
    xx = np.asarray(layer["xx"], dtype=float)
    yy = np.asarray(layer["yy"], dtype=float)
    if occupied.shape != xx.shape or occupied.shape != yy.shape:
        return None

    return {
        "kind": "active_floor",
        "ambiguous": True,
        "support_z": float(layer["support_z"]),
        "bottom_tol": float(layer["bottom_tol"]),
        "occupancy": float(layer["occupancy"]),
        "required": required,
        "grid_shape": tuple(int(x) for x in occupied.shape),
        "x_coords": xx[0, :].astype(float).copy(),
        "y_coords": yy[:, 0].astype(float).copy(),
        "occupied": occupied.copy(),
    }


def _action_sampling_debug(env, mcts=None) -> dict:
    builder = getattr(env, "action_builder", None)
    if builder is None:
        return {}

    result = {
        "last_rejection_counts": dict(
            getattr(builder, "last_rejection_counts", {}) or {}
        ),
        "last_planar_candidates": copy.deepcopy(
            getattr(builder, "last_planar_candidates", []) or []
        ),
        "recent_planar_candidate_calls": copy.deepcopy(
            getattr(builder, "recent_planar_candidate_calls", []) or []
        ),
    }
    if mcts is not None:
        result["root_planar_candidate_calls"] = copy.deepcopy(
            getattr(mcts, "_last_root_planar_candidate_calls", []) or []
        )
    return result


def _node_world_y(node, target_structure_offset) -> float | None:
    action = getattr(node, "action", None)
    pose = getattr(action, "pose", None) if action is not None else None
    if (
        pose is None
        and action is not None
        and getattr(action, "stone_id", None) is not None
    ):
        state = getattr(node, "state", None)
        pose = getattr(state, "stone_poses", {}).get(action.stone_id, None)
    if pose is None:
        return None
    try:
        local_y = float(np.asarray(pose, dtype=float)[1])
        world_y = local_y + float(np.asarray(target_structure_offset, dtype=float)[1])
        return world_y if np.isfinite(world_y) else None
    except Exception:
        return None


def _append_debug_step(
    debug_data,
    step,
    succeeded,
    state,
    nodes,
    env,
    resume_state=None,
    attempt=None,
    failure_reason=None,
    rejected_stone_ids=None,
    selected_node=None,
    selected_action_idx=None,
    selected_regrasp_xy=None,
    selected_motion_mode=None,
    mcts=None,
) -> None:
    recoverable_state = resume_state if resume_state is not None else state
    stone_id_seq = [int(env.inventory.stones[idx].id) for idx in state.stone_seq]
    stone_poses = {
        int(k): np.asarray(v, dtype=float).copy() for k, v in state.stone_poses.items()
    }
    pose_identified_ids = sorted(_pose_identified_stone_ids(recoverable_state))
    cand_dicts = []
    selected_candidate = None
    for rank, node in enumerate(nodes, start=1):
        if node.action is None or node.action.stone_idx < 0:
            continue
        a = node.action
        final_pose = _candidate_final_pose(node)
        info, pose_solve_contacts = _candidate_diagnostics(a)
        info.update(_serializable_info(node.info))
        cand = {
            "rank": rank,
            "score": _node_score(node),
            "stone_id": int(a.stone_id),
            "stone_idx": int(a.stone_idx),
            "pose": np.asarray(a.pose, dtype=float).copy(),
            "solved_pose": solved_action_pose(a).copy(),
            "settled_pose": np.asarray(a.pose, dtype=float).copy(),
            "init_pose": np.asarray(a.init_pose, dtype=float).copy(),
            "failed": bool(node.failed),
            "motion_failed": bool(getattr(node, "_motion_failed", False)),
            "simulated": bool(node.is_simulated),
            "validated": bool(node.is_simulated and node.reward is not None),
            "reward": None if node.reward is None else float(node.reward),
            "q_value": float(node.q_value),
            "q_value_init": float(node.q_value_init),
            "value_init": float(node.value_init),
            "visits": int(node.visits),
            "trajectory": _candidate_trajectory(node),
            "velocity_integrals": (
                {} if node.state is None else node.state.latest_velocity_integrals()
            ),
            "scene_motion": _candidate_scene_motion(node),
            "info": info,
            "contact_points": _candidate_contact_points(node),
            "pose_solve_contacts": pose_solve_contacts,
            "best_sequence": _candidate_best_sequence(node),
            "failed_grasps": list(getattr(node, "_failed_grasps", []) or []),
        }
        if final_pose is not None:
            cand["final_pose"] = final_pose
        if selected_candidate is None and _same_candidate_node(node, selected_node):
            cand["selected"] = True
            selected_candidate = {
                "rank": rank,
                "action_idx": (
                    None if selected_action_idx is None else int(selected_action_idx)
                ),
                "score": cand["score"],
                "stone_id": cand["stone_id"],
                "stone_idx": cand["stone_idx"],
                "pose": cand["pose"].copy(),
                "solved_pose": cand["solved_pose"].copy(),
                "settled_pose": cand["settled_pose"].copy(),
                "init_pose": cand["init_pose"].copy(),
            }
            if final_pose is not None:
                selected_candidate["final_pose"] = final_pose.copy()
            if selected_regrasp_xy is not None:
                selected_candidate["selected_regrasp_xy_pos"] = np.asarray(
                    selected_regrasp_xy, dtype=float
                ).copy()
            if selected_motion_mode is not None:
                selected_candidate["motion_mode"] = str(selected_motion_mode)
        cand_dicts.append(cand)
    action_sampling = _action_sampling_debug(env, mcts)
    step_record = {
        "step": step,
        "attempt": None if attempt is None else int(attempt),
        "succeeded": succeeded,
        "failure_reason": failure_reason,
        "rejected_stone_ids": [
            int(stone_id) for stone_id in (rejected_stone_ids or [])
        ],
        "scene": {
            "stone_seq": stone_id_seq,
            "stone_poses": stone_poses,
            "pose_identified_stone_ids": pose_identified_ids,
        },
        "score_map": state_score_map_debug(env, state),
        "floor_fill": _active_floor_debug(env, state),
        "action_sampling": action_sampling,
        "candidates": cand_dicts,
        "raw_state": copy.deepcopy(state),
        "resume_state": copy.deepcopy(recoverable_state),
    }
    if selected_candidate is not None:
        step_record["selected_candidate"] = selected_candidate
    debug_data["steps"].append(step_record)
    # Top-level pointer so _load_resume_state can find the latest state quickly.
    debug_data["resume_state"] = copy.deepcopy(recoverable_state)
    debug_data["resume_step"] = step


def _no_candidate_failure_reason(state, env, rejected_stone_ids=None) -> str:
    stone_set = [int(x) for x in np.asarray(env.inventory.stone_set).reshape(-1)]
    placed_indices = {
        int(x) for x in np.asarray(getattr(state, "stone_seq", [])).reshape(-1)
    }
    remaining = [
        (idx, stone_id)
        for idx, stone_id in enumerate(stone_set)
        if idx not in placed_indices
    ]
    if not remaining:
        return "all_stones_placed"

    banned = {int(stone_id) for stone_id in env.cfg.action.get("banned_stone_ids", [])}
    rejected = {int(stone_id) for stone_id in (rejected_stone_ids or [])}
    available = [
        stone_id
        for _, stone_id in remaining
        if stone_id not in banned and stone_id not in rejected
    ]
    if available:
        return "no_mcts_candidates"
    if all(stone_id in banned for _, stone_id in remaining):
        return "no_available_stones_banned"
    if rejected and all(
        stone_id in banned or stone_id in rejected for _, stone_id in remaining
    ):
        return "no_available_stones_execution_rejected"
    return "no_available_stones"


def _save_debug_data(debug_data, path) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(debug_data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _load_debug_data_or_init(path: str, env) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and isinstance(data.get("steps", []), list):
                return data
            print(f"[WARN] Resume debug data has unexpected format: {path}")
        except Exception as exc:
            print(f"[WARN] Could not load resume debug data from {path}: {exc}")
    return _init_debug_data(env)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a stacking sequence.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="YAML",
        help=(
            "Config YAML to use instead of the resume/source plan config. "
            "The effective config is still saved into the output session."
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PLAN_DIR",
        help=(
            "Resume from a previous plan directory. Loads state.pkl by default; "
            "use --resume-state-pkl for a reconstructed SceneID state file."
        ),
    )
    p.add_argument(
        "--resume-state-pkl",
        type=str,
        default=None,
        metavar="PKL",
        help=(
            "State pickle to load when using --resume. Relative paths are "
            "resolved inside PLAN_DIR. When set, output is branched to a new "
            "session directory instead of modifying PLAN_DIR. Defaults to "
            "state.pkl in PLAN_DIR for in-place resume."
        ),
    )
    p.add_argument(
        "--resume-start-step",
        type=int,
        default=None,
        metavar="STEP",
        help=(
            "When resuming, start replanning at this 1-based step. "
            "Saved actions, motions, and debug steps from STEP onward are "
            "overwritten."
        ),
    )
    p.add_argument(
        "--branch-start-step",
        type=int,
        default=None,
        metavar="STEP",
        help=(
            "When resuming, write output to a new branch and start replanning "
            "at this 1-based step. The source plan is left unchanged."
        ),
    )
    p.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override an OmegaConf config value after loading config.yml. "
            "May be repeated, e.g. "
            "--config-override environment.reward.place_stability.n_noise=1"
        ),
    )
    p.add_argument(
        "--target-structure-offset",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help=(
            "Override the target structure XY offset for this generated "
            "sequence. The value is saved into planning_params.pkl."
        ),
    )
    p.add_argument(
        "--place-z-offset",
        type=float,
        default=None,
        metavar="Z",
        help=(
            "Add a Z offset, in meters, to generated place_config poses before "
            "motion planning. Defaults to PLACE_Z_OFFSET."
        ),
    )
    p.add_argument(
        "--regrasp-xy-pos",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help=(
            "Override the base regrasp XY position. Derived regrasp candidates "
            "are recomputed and saved into planning_params.pkl."
        ),
    )
    p.add_argument(
        "--simulate-sceneid-stones",
        action="store_true",
        help=(
            "Simulate scene-pose-identified stones in planning instead of "
            "freezing them, with body error_reduction_ratio 0.0 so contacts "
            "among the (interpenetrating) reconstructed stones get no "
            "penetration correction while contacts with new stones keep the "
            "default rate. Sets environment.sim.simulate_pose_identified in "
            "the effective saved config."
        ),
    )
    p.add_argument(
        "--ground-first",
        action="store_true",
        help=(
            "Reject placements above the active lower floor until its occupancy "
            "reaches environment.action.planar.floor_fill.lower_floor_fill_ratio."
        ),
    )
    args = p.parse_args(argv)
    if args.resume_state_pkl is not None and args.resume is None:
        p.error("--resume-state-pkl requires --resume")
    if args.resume_start_step is not None:
        if args.resume is None:
            p.error("--resume-start-step requires --resume")
        if args.resume_start_step < 1:
            p.error("--resume-start-step must be >= 1")
    if args.branch_start_step is not None:
        if args.resume is None:
            p.error("--branch-start-step requires --resume")
        if args.branch_start_step < 1:
            p.error("--branch-start-step must be >= 1")
        if args.resume_start_step is not None:
            p.error("--branch-start-step cannot be combined with --resume-start-step")
    if args.config is not None:
        args.config = _resolve_config_arg(args.config)
        if not os.path.exists(args.config):
            p.error(f"--config path does not exist: {args.config}")
    for override in args.config_override:
        if "=" not in override:
            p.error(f"--config-override expects KEY=VALUE, got {override!r}")
    if args.place_z_offset is not None and not np.isfinite(float(args.place_z_offset)):
        p.error("--place-z-offset must be finite")
    return args


def _load_resume_planning_params(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            params = pickle.load(f)
    except Exception as exc:
        print(f"[WARN] Could not load resumed planning params from {path}: {exc}")
        return {}
    if not isinstance(params, dict):
        print(f"[WARN] Resumed planning params has unexpected format: {path}")
        return {}
    return params


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _resume_place_plane_height(params: dict) -> float | None:
    if not isinstance(params, dict):
        return None
    return _finite_float(params.get("place_plane_height", None))


def _debug_floor_support_height(state_pkl: str) -> float | None:
    if not os.path.exists(state_pkl):
        return None
    try:
        with open(state_pkl, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    heights = []
    for step in data.get("steps", []) or []:
        floor = step.get("floor_fill", {}) if isinstance(step, dict) else {}
        height = _finite_float(floor.get("support_z", None))
        if height is not None and abs(height) > 1e-9:
            heights.append(height)
    if not heights:
        return None
    return heights[-1]


def _resolve_resume_place_plane_height(
    resume_params: dict,
    state_pkl: str,
    sceneid_ground_height: float | None,
    default_height: float,
) -> float:
    if sceneid_ground_height is not None:
        return float(sceneid_ground_height)

    resumed_height = _resume_place_plane_height(resume_params)
    debug_height = _debug_floor_support_height(state_pkl)
    if debug_height is not None and (
        resumed_height is None or abs(resumed_height) <= 1e-9
    ):
        print(
            "[INFO] Recovered resumed place_plane_height from debug floor "
            f"support_z: {debug_height:.4f}"
        )
        return float(debug_height)
    if resumed_height is not None:
        print("[INFO] Using resumed place_plane_height: " f"{resumed_height:.4f}")
        return float(resumed_height)
    return float(default_height)


def _resolve_config_arg(config: str) -> str:
    path = os.path.normpath(os.path.expanduser(config))
    if os.path.exists(path):
        return path
    candidates = []
    if not path.endswith((".yml", ".yaml")):
        candidates.append(f"{path}.yml")
        candidates.append(f"{path}.yaml")
    basename = os.path.basename(path)
    candidates.append(os.path.join("agent", "configs", basename))
    if not basename.endswith((".yml", ".yaml")):
        candidates.append(os.path.join("agent", "configs", f"{basename}.yml"))
        candidates.append(os.path.join("agent", "configs", f"{basename}.yaml"))
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            return candidate
    return path


def _plan_date_from_name(plan_name: str) -> str | None:
    parts = plan_name.split("_")
    if len(parts) >= 3 and parts[0] == "plan" and parts[1].isdigit():
        return parts[1]
    return None


def _new_plan_dir(sessions_dir: str, suffix: str = "") -> str:
    date_str = datetime.datetime.now().strftime("%y%m%d")
    dated_sessions_dir = os.path.join(sessions_dir, date_str)
    os.makedirs(dated_sessions_dir, exist_ok=True)
    suffix = str(suffix or "").strip("_")
    if not suffix:
        return get_unique_dir(dated_sessions_dir, prefix=f"plan_{date_str}")

    counter = 1
    while True:
        candidate = os.path.join(
            dated_sessions_dir,
            f"plan_{date_str}_{counter}_{suffix}",
        )
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _resolve_plan_dir(plan_dir: str) -> str:
    path = os.path.normpath(os.path.expanduser(plan_dir))
    candidates = [path]

    plan_name = os.path.basename(path)
    parent = os.path.dirname(path)
    if not plan_name.startswith("plan_"):
        plan_name = f"plan_{plan_name}"
    date_str = _plan_date_from_name(plan_name)
    if date_str is not None:
        if parent:
            candidates.append(os.path.join(parent, date_str, plan_name))
        candidates.append(os.path.join("sessions", date_str, plan_name))
    candidates.append(os.path.join("sessions", plan_name))

    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if os.path.isdir(candidate):
            return candidate
    return path


def _resolve_resume_state_pkl(plan_dir: str, resume_state_pkl: str | None) -> str:
    if resume_state_pkl is None:
        return os.path.join(plan_dir, "state.pkl")
    path = os.path.expanduser(resume_state_pkl)
    if not os.path.isabs(path):
        path = os.path.join(plan_dir, path)
    return os.path.normpath(path)


def _load_resume_state(state_pkl: str, env):
    """Load a resume state pickle and return (resume_state, start_step)."""
    with open(state_pkl, "rb") as f:
        data = pickle.load(f)

    # Prefer the top-level resume_state written after every step.
    resume_state = data.get("resume_state", None)
    if resume_state is None:
        steps = data.get("steps", [])
        if not steps:
            raise ValueError(f"No steps found in {state_pkl}")
        last = steps[-1]
        resume_state = last.get("resume_state", last.get("raw_state", None))
    if resume_state is None:
        raise ValueError(f"No resume_state found in {state_pkl}")

    resume_state = copy.deepcopy(resume_state)
    resume_state.pose_identified_stone_ids = _pose_identified_stone_ids(resume_state)
    start_step = len(resume_state.stone_seq)
    return resume_state, start_step


def _sceneid_resume_ground_height(state_pkl: str) -> float | None:
    if not os.path.exists(state_pkl):
        return None
    try:
        with open(state_pkl, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read sceneid resume metadata from {state_pkl}: {exc}")
        return None

    metadata = data.get("reconstructed_from_logs", {})
    if not isinstance(metadata, dict):
        return None

    ground_height = metadata.get("ground_height", None)
    if ground_height is None:
        return None

    try:
        ground_height = float(ground_height)
    except (TypeError, ValueError):
        print(f"[WARN] Ignoring non-numeric sceneid ground_height: {ground_height!r}")
        return None
    if not np.isfinite(ground_height):
        print(f"[WARN] Ignoring non-finite sceneid ground_height: {ground_height!r}")
        return None
    return ground_height


def _apply_config_overrides(cfg, overrides: list[str]):
    if not overrides:
        return cfg
    override_cfg = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.merge(cfg, override_cfg)
    print("[INFO] Applied config override(s):")
    for override in overrides:
        print(f"  - {override}")
    return cfg


def _set_action_score_excavator_xy(cfg, target_structure_offset: np.ndarray) -> None:
    planar_cfg = cfg.environment.action.planar
    if "score" not in planar_cfg or planar_cfg.score is None:
        planar_cfg.score = {}
    score_cfg = planar_cfg.score
    excavator_xy = -np.asarray(target_structure_offset, dtype=float).reshape(2)
    score_cfg.excavator_xy = [float(excavator_xy[0]), float(excavator_xy[1])]
    print(
        "[INFO] Using local excavator xy for action scoring: "
        f"{score_cfg.excavator_xy}"
    )


def _resolve_config_path(args, source_config_path: str) -> str:
    if args.config is not None:
        print(f"[INFO] Using config file override: {args.config}")
        return args.config
    if os.path.exists(source_config_path):
        return source_config_path
    return "agent/configs/config.yml"


def _config_overrides_key(overrides: list[str], key: str) -> bool:
    for override in overrides or []:
        override_key = override.split("=", 1)[0].strip()
        if override_key == key:
            return True
    return False


def _plain_log_value(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain_log_value(item) for item in value]
    return value


def _record_run_config_change(
    planning_params: dict,
    name: str,
    old_value,
    new_value,
    reason: str,
) -> None:
    old_plain = _plain_log_value(old_value)
    new_plain = _plain_log_value(new_value)
    if old_plain == new_plain:
        return
    entry = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": str(name),
        "old": old_plain,
        "new": new_plain,
        "reason": str(reason),
    }
    planning_params.setdefault("run_config_changes", []).append(entry)
    print(
        "[INFO] Run config change: " f"{name} {old_plain!r} -> {new_plain!r} ({reason})"
    )


def _cfg_pose_data_path(cfg) -> str:
    return os.path.normpath(
        os.path.expanduser(str(cfg.environment.action.pose_data_path))
    )


def _select_pose_data_load_path(
    args, cfg, source_pose_data_path: str
) -> tuple[str, str]:
    cfg_pose_data_path = _cfg_pose_data_path(cfg)
    if args.config is not None or _config_overrides_key(
        args.config_override,
        "environment.action.pose_data_path",
    ):
        return cfg_pose_data_path, "config override"
    if args.resume is not None and os.path.exists(source_pose_data_path):
        return source_pose_data_path, "resume session"
    return cfg_pose_data_path, "config"


def _load_pose_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pose data file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _same_existing_file(path_a: str, path_b: str) -> bool:
    try:
        return (
            os.path.exists(path_a)
            and os.path.exists(path_b)
            and os.path.samefile(
                path_a,
                path_b,
            )
        )
    except OSError:
        return False


def _write_pose_data_copy(poses, source_path: str, save_path: str) -> str:
    if _same_existing_file(source_path, save_path):
        return save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(poses, f)
    return save_path


def _resume_source_metadata(
    source_plan_dir: str,
    source_state_pkl: str,
    branched: bool,
) -> dict:
    return {
        "plan_dir": os.path.abspath(source_plan_dir),
        "state_pkl": os.path.abspath(source_state_pkl),
        "branched_to_new_plan": bool(branched),
    }


def _update_resume_planning_params(
    params_path: str,
    place_plane_height: float,
    inhand_replan_mode: str,
    target_structure_offset: np.ndarray,
    regrasp_xy_pos: np.ndarray,
    regrasp_xy_candidates,
    ground_first: bool | None = None,
    lower_floor_fill_reject_stacked_enabled: bool | None = None,
    mcts_max_depth: int | None = None,
    run_config_changes: list[dict] | None = None,
    poses=None,
    pose_data_path: str | None = None,
    pose_data_source_path: str | None = None,
    refresh_poses: bool = False,
) -> None:
    if not os.path.exists(params_path):
        return
    try:
        with open(params_path, "rb") as f:
            params = pickle.load(f)
    except Exception as exc:
        print(f"[WARN] Could not update {params_path}: {exc}")
        return
    if not isinstance(params, dict):
        return

    changed = False
    old_height = params.get("place_plane_height", None)
    height_changed = True
    if old_height is not None:
        try:
            height_changed = abs(float(old_height) - float(place_plane_height)) >= 1e-9
        except (TypeError, ValueError):
            pass
    if height_changed:
        params["place_plane_height"] = float(place_plane_height)
        changed = True

    old_mode = params.get("inhand_replan_mode", None)
    if old_mode != inhand_replan_mode:
        params["inhand_replan_mode"] = inhand_replan_mode
        changed = True

    old_offset = params.get("target_structure_offset", None)
    offset = np.asarray(target_structure_offset, dtype=float).reshape(2)
    offset_changed = True
    if old_offset is not None:
        try:
            old_offset_arr = np.asarray(old_offset, dtype=float).reshape(2)
            offset_changed = not np.allclose(old_offset_arr, offset)
        except (TypeError, ValueError):
            pass
    if offset_changed:
        params["target_structure_offset"] = offset.copy()
        changed = True

    old_regrasp_xy = params.get("regrasp_xy_pos", None)
    regrasp_xy = np.asarray(regrasp_xy_pos, dtype=float).reshape(2)
    regrasp_xy_changed = True
    if old_regrasp_xy is not None:
        try:
            old_regrasp_xy_arr = np.asarray(old_regrasp_xy, dtype=float).reshape(2)
            regrasp_xy_changed = not np.allclose(old_regrasp_xy_arr, regrasp_xy)
        except (TypeError, ValueError):
            pass
    if regrasp_xy_changed:
        params["regrasp_xy_pos"] = regrasp_xy.copy()
        params["regrasp_xy_candidates"] = [
            np.asarray(candidate, dtype=float).copy()
            for candidate in regrasp_xy_candidates
        ]
        changed = True

    pose_data_changed = False
    if pose_data_path is not None:
        old_pose_data_path = params.get("pose_data_path", None)
        if old_pose_data_path != pose_data_path:
            params["pose_data_path"] = pose_data_path
            pose_data_changed = True
            changed = True
    if pose_data_source_path is not None:
        old_pose_data_source_path = params.get("pose_data_source_path", None)
        if old_pose_data_source_path != pose_data_source_path:
            params["pose_data_source_path"] = pose_data_source_path
            changed = True
    if poses is not None and (
        refresh_poses or pose_data_changed or "poses" not in params
    ):
        params["poses"] = poses
        changed = True

    old_ground_first = params.get("ground_first", None)
    if ground_first is not None and old_ground_first != bool(ground_first):
        params["ground_first"] = bool(ground_first)
        changed = True

    old_lower_floor_fill_reject_stacked = params.get(
        LOWER_FLOOR_FILL_REJECT_STACKED_KEY,
        params.get(LEGACY_GROUND_FILL_REJECT_STACKED_KEY, None),
    )
    if LEGACY_GROUND_FILL_REJECT_STACKED_KEY in params:
        if LOWER_FLOOR_FILL_REJECT_STACKED_KEY not in params:
            params[LOWER_FLOOR_FILL_REJECT_STACKED_KEY] = bool(
                params[LEGACY_GROUND_FILL_REJECT_STACKED_KEY]
            )
        del params[LEGACY_GROUND_FILL_REJECT_STACKED_KEY]
        changed = True
    if (
        lower_floor_fill_reject_stacked_enabled is not None
        and params.get(LOWER_FLOOR_FILL_REJECT_STACKED_KEY, None)
        != bool(lower_floor_fill_reject_stacked_enabled)
    ):
        params[LOWER_FLOOR_FILL_REJECT_STACKED_KEY] = bool(
            lower_floor_fill_reject_stacked_enabled
        )
        changed = True

    old_mcts_max_depth = params.get("mcts_max_depth", None)
    if mcts_max_depth is not None:
        try:
            old_depth_matches = int(old_mcts_max_depth) == int(mcts_max_depth)
        except (TypeError, ValueError):
            old_depth_matches = False
        if not old_depth_matches:
            params["mcts_max_depth"] = int(mcts_max_depth)
            changed = True

    if run_config_changes:
        params.setdefault("run_config_changes", []).extend(run_config_changes)
        changed = True

    if not changed:
        return

    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    updates = []
    if height_changed:
        updates.append(f"place_plane_height {old_height} -> {place_plane_height:.4f}")
    if old_mode != inhand_replan_mode:
        updates.append(f"inhand_replan_mode {old_mode!r} -> {inhand_replan_mode!r}")
    if offset_changed:
        updates.append(
            "target_structure_offset " f"{old_offset!r} -> {offset.tolist()}"
        )
    if regrasp_xy_changed:
        updates.append("regrasp_xy_pos " f"{old_regrasp_xy!r} -> {regrasp_xy.tolist()}")
    if pose_data_changed or refresh_poses:
        updates.append(f"pose_data_path -> {pose_data_path}")
    if ground_first is not None and old_ground_first != bool(ground_first):
        updates.append(f"ground_first {old_ground_first!r} -> {bool(ground_first)!r}")
    if (
        lower_floor_fill_reject_stacked_enabled is not None
        and old_lower_floor_fill_reject_stacked
        != bool(lower_floor_fill_reject_stacked_enabled)
    ):
        updates.append(
            "lower_floor_fill_reject_stacked "
            f"{old_lower_floor_fill_reject_stacked!r} -> "
            f"{bool(lower_floor_fill_reject_stacked_enabled)!r}"
        )
    if mcts_max_depth is not None:
        try:
            old_depth_matches = int(old_mcts_max_depth) == int(mcts_max_depth)
        except (TypeError, ValueError):
            old_depth_matches = False
        if not old_depth_matches:
            updates.append(
                f"mcts_max_depth {old_mcts_max_depth!r} -> {int(mcts_max_depth)}"
            )
    if run_config_changes:
        updates.append(f"run_config_changes +{len(run_config_changes)}")
    print("[INFO] Updated resumed planning_params.pkl: " + ", ".join(updates))


def _action_pose_by_stone_id(action_sequence) -> dict:
    poses = {}
    for action in action_sequence:
        if "stone_id" not in action or "pose" not in action:
            continue
        poses[int(action["stone_id"])] = np.asarray(action["pose"], dtype=float).copy()
    return poses


def _backup_resume_trim_inputs(save_dir: str, paths: list[str]) -> str | None:
    existing = [path for path in paths if os.path.exists(path)]
    if not existing:
        return None

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(save_dir, f"resume_trim_backup_{stamp}")
    os.makedirs(backup_dir, exist_ok=False)
    for path in existing:
        shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
    print(f"[INFO] Backed up pre-trim resume files to {backup_dir}")
    return backup_dir


def _trim_resume_sequences(
    action_sequence: list,
    motion_sequence: list,
    motion_result_sequence: list,
    resume_step: int,
) -> tuple[list, list, list]:
    resume_step = max(0, int(resume_step))
    if len(motion_result_sequence) < len(motion_sequence):
        motion_result_sequence = list(motion_result_sequence) + [None] * (
            len(motion_sequence) - len(motion_result_sequence)
        )

    old_lengths = (
        len(action_sequence),
        len(motion_sequence),
        len(motion_result_sequence),
    )
    action_sequence = list(action_sequence[:resume_step])
    motion_sequence = list(motion_sequence[:resume_step])
    motion_result_sequence = list(motion_result_sequence[:resume_step])
    new_lengths = (
        len(action_sequence),
        len(motion_sequence),
        len(motion_result_sequence),
    )
    if old_lengths != new_lengths:
        print(
            "[INFO] Trimmed resumed sequences to resume_step "
            f"{resume_step}: action/motion/motion_result "
            f"{old_lengths} -> {new_lengths}"
        )
    return action_sequence, motion_sequence, motion_result_sequence


def _resume_state_prefix_entries(
    resume_state, resume_step: int
) -> list[tuple[int, np.ndarray]]:
    resume_step = max(0, int(resume_step))
    stone_set = [int(stone_id) for stone_id in resume_state.stone_set]
    entries = []
    for stone_idx in list(getattr(resume_state, "stone_seq", []) or [])[:resume_step]:
        stone_id = int(stone_set[int(stone_idx)])
        pose = getattr(resume_state, "stone_poses", {}).get(stone_id)
        if pose is None:
            raise ValueError(
                "Resume state is missing placed-stone pose for stone " f"{stone_id}"
            )
        arr = np.asarray(pose, dtype=float).reshape(-1)
        if arr.shape[0] < 7 or not np.all(np.isfinite(arr[:7])):
            raise ValueError(
                "Resume state has invalid placed-stone pose for stone "
                f"{stone_id}: {pose!r}"
            )
        entries.append((stone_id, arr[:7].copy()))
    return entries


def _sync_resume_sequences_to_state_prefix(
    action_sequence: list,
    motion_sequence: list,
    motion_result_sequence: list,
    resume_state,
    resume_step: int,
) -> tuple[list, list, list]:
    """Make saved prefix actions match the resume state's placed sequence."""
    prefix_entries = _resume_state_prefix_entries(resume_state, resume_step)
    if not prefix_entries:
        return action_sequence, motion_sequence, motion_result_sequence

    if len(motion_result_sequence) < len(motion_sequence):
        motion_result_sequence = list(motion_result_sequence) + [None] * (
            len(motion_sequence) - len(motion_result_sequence)
        )

    max_len = max(
        len(action_sequence),
        len(motion_sequence),
        len(motion_result_sequence),
    )
    records = []
    for idx in range(max_len):
        action = (
            copy.deepcopy(action_sequence[idx]) if idx < len(action_sequence) else None
        )
        stone_id = None
        if isinstance(action, dict) and "stone_id" in action:
            try:
                stone_id = int(action["stone_id"])
            except (TypeError, ValueError):
                stone_id = None
        records.append(
            {
                "index": idx,
                "stone_id": stone_id,
                "action": action,
                "motion": motion_sequence[idx] if idx < len(motion_sequence) else None,
                "motion_result": (
                    motion_result_sequence[idx]
                    if idx < len(motion_result_sequence)
                    else None
                ),
            }
        )

    used = set()

    def take_record(stone_id: int):
        for rec in records:
            if rec["index"] in used:
                continue
            if rec["stone_id"] == int(stone_id):
                used.add(rec["index"])
                return rec
        return None

    expected_ids = [stone_id for stone_id, _ in prefix_entries]
    old_prefix_ids = [
        int(action["stone_id"])
        for action in action_sequence[: len(expected_ids)]
        if isinstance(action, dict) and "stone_id" in action
    ]

    new_actions = []
    new_motions = []
    new_motion_results = []
    inserted_prefix_ids = []
    for stone_id, pose in prefix_entries:
        rec = take_record(stone_id)
        if rec is None or not isinstance(rec["action"], dict):
            action = {"stone_id": int(stone_id)}
            motion = None
            motion_result = None
            inserted_prefix_ids.append(int(stone_id))
        else:
            action = copy.deepcopy(rec["action"])
            motion = rec["motion"]
            motion_result = rec["motion_result"]
        action["stone_id"] = int(stone_id)
        action["pose"] = pose.copy()
        new_actions.append(action)
        new_motions.append(motion)
        new_motion_results.append(motion_result)

    prefix_id_set = set(expected_ids)
    skipped_duplicate_ids = []
    for rec in records:
        if rec["index"] in used or not isinstance(rec["action"], dict):
            continue
        if rec["stone_id"] in prefix_id_set:
            skipped_duplicate_ids.append(int(rec["stone_id"]))
            continue
        new_actions.append(copy.deepcopy(rec["action"]))
        new_motions.append(rec["motion"])
        new_motion_results.append(rec["motion_result"])

    changed = (
        old_prefix_ids != expected_ids
        or len(new_actions) != len(action_sequence)
        or inserted_prefix_ids
        or skipped_duplicate_ids
    )
    if changed:
        print(
            "[INFO] Reconciled resumed action prefix with resume_state: "
            f"prefix_len={len(expected_ids)}, "
            f"old_prefix_tail={old_prefix_ids[-5:]}, "
            f"new_prefix_tail={expected_ids[-5:]}, "
            f"inserted={inserted_prefix_ids}, "
            f"skipped_duplicates={skipped_duplicate_ids}"
        )

    return new_actions, new_motions, new_motion_results


def _truncate_resume_state(resume_state, resume_step: int) -> None:
    resume_step = int(resume_step)
    current_step = len(getattr(resume_state, "stone_seq", []) or [])
    if resume_step < 0:
        raise ValueError(f"resume_step must be >= 0, got {resume_step}")
    if resume_step > current_step:
        raise ValueError(
            "Cannot resume from requested start step: loaded state only has "
            f"{current_step} placed stones, requested {resume_step}."
        )
    if resume_step == current_step:
        return

    old_seq = [int(idx) for idx in resume_state.stone_seq]
    kept_seq = old_seq[:resume_step]
    kept_idx_set = set(kept_seq)
    stone_set = [int(stone_id) for stone_id in resume_state.stone_set]
    kept_ids = {stone_set[idx] for idx in kept_seq}

    resume_state.stone_seq = kept_seq
    resume_state.stone_poses = {
        int(stone_id): pose
        for stone_id, pose in getattr(resume_state, "stone_poses", {}).items()
        if int(stone_id) in kept_ids
    }
    resume_state.trajectories = list(
        (getattr(resume_state, "trajectories", []) or [])[:resume_step]
    )
    resume_state.action_history = list(
        (getattr(resume_state, "action_history", []) or [])[:resume_step]
    )

    contacts = []
    for contact in getattr(resume_state, "contact_points", []) or []:
        if not isinstance(contact, dict):
            continue
        idxs = [
            contact.get("stone_idx_1", None),
            contact.get("stone_idx_2", None),
        ]
        if all(idx is None or int(idx) in kept_idx_set for idx in idxs):
            contacts.append(contact)
    resume_state.contact_points = contacts
    resume_state.pose_identified_stone_ids = {
        int(stone_id)
        for stone_id in getattr(resume_state, "pose_identified_stone_ids", set())
        if int(stone_id) in kept_ids
    }
    resume_state.terminated = False
    resume_state.failed = False
    print(
        "[INFO] Truncated resume state to "
        f"{resume_step} placed stones for replanning from step {resume_step + 1}."
    )


def _trim_debug_data_steps(debug_data: dict, resume_step: int) -> None:
    if not isinstance(debug_data, dict):
        return
    steps = debug_data.get("steps", [])
    if not isinstance(steps, list):
        return
    kept = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        try:
            step = int(item.get("step"))
        except (TypeError, ValueError):
            continue
        if step <= int(resume_step):
            kept.append(item)
    if len(kept) != len(steps):
        print(
            "[INFO] Trimmed debug steps for resume override: "
            f"{len(steps)} -> {len(kept)}"
        )
    debug_data["steps"] = kept


def _fill_missing_resume_stone_poses(resume_state, action_sequence) -> None:
    action_poses = _action_pose_by_stone_id(action_sequence)
    recovered = []
    missing = []
    for stone_idx in resume_state.stone_seq:
        stone_id = int(resume_state.stone_set[int(stone_idx)])
        if stone_id in resume_state.stone_poses:
            continue
        pose = action_poses.get(stone_id)
        if pose is None:
            missing.append(stone_id)
            continue
        resume_state.stone_poses[stone_id] = pose
        recovered.append(stone_id)

    if recovered:
        print(
            "[INFO] Recovered missing resume stone poses from action_sequence: "
            f"{recovered}"
        )
    if missing:
        raise ValueError(
            "Resume state is missing placed-stone poses and action_sequence has "
            f"no fallback poses for stone ids: {missing}"
        )


def _remap_resume_state_to_current_inventory(resume_state, current_stone_set) -> bool:
    old_stone_set = [int(stone_id) for stone_id in resume_state.stone_set]
    old_stone_seq = [int(idx) for idx in resume_state.stone_seq]
    placed_ids = [old_stone_set[idx] for idx in old_stone_seq]

    current_ids = []
    for stone_id in current_stone_set:
        stone_id = int(stone_id)
        if stone_id not in current_ids:
            current_ids.append(stone_id)

    old_pending_ids = [
        stone_id for stone_id in old_stone_set if stone_id not in placed_ids
    ]
    pending_ids = list(old_pending_ids)
    for stone_id in current_ids:
        if stone_id not in placed_ids and stone_id not in pending_ids:
            pending_ids.append(stone_id)
    new_stone_set = placed_ids + pending_ids

    if old_stone_set == new_stone_set and old_stone_seq == list(range(len(placed_ids))):
        return False

    old_idx_to_new_idx = {
        old_idx: new_idx for new_idx, old_idx in enumerate(old_stone_seq)
    }
    resume_state.stone_set = np.asarray(new_stone_set)
    resume_state.stone_seq = list(range(len(placed_ids)))

    for new_idx, action in enumerate(getattr(resume_state, "action_history", []) or []):
        if new_idx >= len(placed_ids):
            break
        if hasattr(action, "stone_idx"):
            action.stone_idx = new_idx
        if hasattr(action, "stone_id"):
            action.stone_id = int(placed_ids[new_idx])

    for contact in getattr(resume_state, "contact_points", []) or []:
        if not isinstance(contact, dict):
            continue
        for key in ("stone_idx_1", "stone_idx_2"):
            old_idx = contact.get(key, None)
            if old_idx is None:
                continue
            contact[key] = old_idx_to_new_idx.get(int(old_idx), None)

    print(
        "[INFO] Remapped resume state to refreshed field inventory: "
        f"{len(placed_ids)} placed, {len(pending_ids)} pending, "
        f"{len(new_stone_set)} total stones."
    )
    print(f"[INFO] Pending refreshed stone ids: {pending_ids}")
    return True


def _resume_scene_pose_dict(resume_state) -> dict[int, np.ndarray]:
    poses = {}
    for stone_idx in getattr(resume_state, "stone_seq", []) or []:
        stone_id = int(resume_state.stone_set[int(stone_idx)])
        pose = resume_state.stone_poses.get(stone_id, None)
        if pose is None:
            continue
        arr = np.asarray(pose, dtype=float)
        if arr.ndim == 1 and arr.shape[0] >= 7 and np.all(np.isfinite(arr[:7])):
            poses[stone_id] = arr[:7].copy()
    return poses


SUPPRESSED_NATIVE_LOG_PATTERNS = (
    "WARNING: Using soft CircularBuffer",
    "FEngine (64 bits) created",
    "EGL(1.5)",
    "OpenGL(4.1)",
)


class _NativeStderrFilter:
    """Filter noisy native renderer startup logs while preserving other stderr."""

    def __enter__(self):
        self._stderr_fd = os.dup(2)
        self._read_fd, write_fd = os.pipe()
        self._pipe_write_fd = write_fd
        os.dup2(write_fd, 2)

        self._thread = threading.Thread(target=self._forward_stderr, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        sys.stderr.flush()
        os.dup2(self._stderr_fd, 2)
        os.close(self._pipe_write_fd)
        os.close(self._stderr_fd)
        self._thread.join(timeout=1.0)

    def _forward_stderr(self):
        with os.fdopen(self._read_fd, "rb", closefd=True) as pipe:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode(errors="replace")
                if any(pattern in line for pattern in SUPPRESSED_NATIVE_LOG_PATTERNS):
                    continue
                try:
                    os.write(self._stderr_fd, raw_line)
                except OSError:
                    break


def _make_place_height_box_mesh(
    wall_display_meshes,
    pick_plane_height: float,
    place_plane_height: float,
):
    import open3d as o3d

    points = []
    for mesh in wall_display_meshes.values():
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim == 2 and vertices.shape[1] >= 3 and vertices.size > 0:
            points.append(vertices[:, :3])
    if not points:
        return None

    pts = np.concatenate(points, axis=0)
    if not np.all(np.isfinite(pts)):
        return None

    xy_min = pts[:, :2].min(axis=0) - PLACE_HEIGHT_BOX_MARGIN
    xy_max = pts[:, :2].max(axis=0) + PLACE_HEIGHT_BOX_MARGIN
    extent_xy = xy_max - xy_min
    if np.any(extent_xy <= 0.0):
        return None

    z_top = float(place_plane_height)
    thickness = max(
        abs(float(place_plane_height) - float(pick_plane_height)),
        PLACE_HEIGHT_BOX_MIN_THICKNESS,
    )
    z_bottom = z_top - thickness

    mesh = o3d.geometry.TriangleMesh.create_box(
        float(extent_xy[0]), float(extent_xy[1]), float(thickness)
    )
    mesh.translate((float(xy_min[0]), float(xy_min[1]), float(z_bottom)))
    mesh.compute_vertex_normals()
    return mesh


def _make_default_ground_material(rendering):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = DEFAULT_GROUND_BASE_COLOR
    return mat


def _build_initial_scene_geometries(
    wall_display_meshes,
    scene_meshes,
    regrasp_xy_candidates,
    place_height_box_mesh=None,
):
    """Return list of (name, geometry, material) for the initial scene overview."""
    import open3d as o3d
    from open3d.visualization import rendering

    items = []

    ground = o3d.geometry.TriangleMesh.create_box(30.0, 30.0, 0.02)
    ground.translate([-15.0, -15.0, -0.02])
    ground.compute_vertex_normals()
    items.append(("ground", ground, _make_default_ground_material(rendering)))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    frame.compute_vertex_normals()
    mat_frame = rendering.MaterialRecord()
    mat_frame.shader = "defaultLit"
    mat_frame.base_color = [1.0, 1.0, 1.0, 1.0]
    items.append(("coord_frame", frame, mat_frame))

    mat_wall = rendering.MaterialRecord()
    mat_wall.shader = "defaultLitTransparency"
    mat_wall.has_alpha = True
    mat_wall.base_color = [0.45, 0.55, 0.85, 0.35]
    for name, mesh in wall_display_meshes.items():
        m = copy.deepcopy(mesh)
        m.compute_vertex_normals()
        items.append((f"wall_{name}", m, mat_wall))

    if place_height_box_mesh is not None:
        box = copy.deepcopy(place_height_box_mesh)
        box.compute_vertex_normals()
        items.append(
            (
                "sceneid_place_height_box",
                box,
                _make_default_ground_material(rendering),
            )
        )

    mat_stone = rendering.MaterialRecord()
    mat_stone.shader = "defaultLit"
    mat_stone.base_color = [0.72, 0.55, 0.38, 1.0]
    for stone_id, mesh in scene_meshes.items():
        m = copy.deepcopy(mesh)
        m.compute_vertex_normals()
        items.append((f"stone_{stone_id}", m, mat_stone))

    for j, rxy in enumerate(regrasp_xy_candidates):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
        sphere.translate([float(rxy[0]), float(rxy[1]), 0.25])
        sphere.compute_vertex_normals()
        mat_rg = rendering.MaterialRecord()
        mat_rg.shader = "defaultLit"
        mat_rg.base_color = [0.1, 0.85, 0.2, 1.0] if j == 0 else [0.4, 0.70, 0.4, 1.0]
        items.append((f"regrasp_{j}", sphere, mat_rg))

    return items


def _render_initial_scene(
    save_dir,
    wall_display_meshes,
    scene_meshes,
    regrasp_xy_candidates,
    place_height_box_mesh,
    camera_center,
    camera_position,
):
    """Render a static overview of the initial scene to preview.png."""
    import open3d as o3d
    from open3d.visualization import rendering

    w, h = 1280, 720
    renderer = rendering.OffscreenRenderer(w, h)
    sc = renderer.scene
    sc.set_background([0.92, 0.92, 0.92, 1.0])
    sc.scene.enable_sun_light(True)
    sc.scene.set_sun_light([0.577, 0.577, -0.577], [1.0, 1.0, 1.0], 100000)

    for name, geom, mat in _build_initial_scene_geometries(
        wall_display_meshes,
        scene_meshes,
        regrasp_xy_candidates,
        place_height_box_mesh,
    ):
        sc.add_geometry(name, geom, mat)

    renderer.setup_camera(60.0, camera_center, camera_position, [0.0, 0.0, 1.0])

    try:
        img = renderer.render_to_image()
        preview_path = os.path.join(save_dir, "preview.png")
        o3d.io.write_image(preview_path, img)
        print(f"[INFO] Initial scene preview saved to {preview_path}")
    except Exception as exc:
        print(f"[WARN] Could not render initial scene preview: {exc}")
    finally:
        del renderer  # release Filament resources before gui.Application initialises


def _show_initial_scene_interactive(
    wall_display_meshes,
    scene_meshes,
    regrasp_xy_candidates,
    place_height_box_mesh,
    camera_center,
    camera_position,
    pose_identified_ids=None,
    simulate_checked=False,
):
    """Open a blocking Open3D preview. Returns on user close or timeout.

    When `pose_identified_ids` is non-empty, a side panel lists those stacked
    stones with an "unfreeze (simulate)" checkbox each; checked stones are
    highlighted orange. Returns the sorted list of checked stone ids (or None
    when no panel was shown / the preview failed).
    """
    import open3d.visualization.gui as gui
    from open3d.visualization import rendering

    pose_identified_ids = sorted(int(i) for i in (pose_identified_ids or []))
    selection = {sid: bool(simulate_checked) for sid in pose_identified_ids}

    try:
        app = gui.Application.instance
        app.initialize()

        win = app.create_window(
            "Initial Scene Preview — close window or wait 60s to start planning",
            1280,
            720,
        )
        scene_widget = gui.SceneWidget()
        scene_widget.scene = rendering.Open3DScene(win.renderer)
        win.add_child(scene_widget)

        panel = None
        panel_width = 0
        if pose_identified_ids:
            em = win.theme.font_size
            panel_width = int(13 * em)

            def _stone_material(simulated: bool):
                mat = rendering.MaterialRecord()
                mat.shader = "defaultLit"
                mat.base_color = (
                    [0.95, 0.55, 0.15, 1.0] if simulated else [0.72, 0.55, 0.38, 1.0]
                )
                return mat

            def _make_on_checked(sid):
                def _on_checked(checked):
                    selection[sid] = bool(checked)
                    scene_widget.scene.modify_geometry_material(
                        f"stone_{sid}", _stone_material(bool(checked))
                    )

                return _on_checked

            panel = gui.Vert(0.25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
            panel.add_child(gui.Label("Unfreeze (simulate)"))
            panel.add_child(gui.Label("stacked stones:"))
            id_list = gui.ScrollableVert(0.25 * em)
            for sid in pose_identified_ids:
                cb = gui.Checkbox(f"stone {sid}")
                cb.checked = bool(selection[sid])
                cb.set_on_checked(_make_on_checked(sid))
                id_list.add_child(cb)
            panel.add_child(id_list)
            win.add_child(panel)

        def on_layout(ctx):
            r = win.content_rect
            if panel is not None:
                scene_widget.frame = gui.Rect(
                    r.x, r.y, r.width - panel_width, r.height
                )
                panel.frame = gui.Rect(
                    r.get_right() - panel_width, r.y, panel_width, r.height
                )
            else:
                scene_widget.frame = r

        win.set_on_layout(on_layout)

        sc = scene_widget.scene
        sc.set_background([0.92, 0.92, 0.92, 1.0])
        sc.scene.enable_sun_light(True)
        sc.scene.set_sun_light([0.577, 0.577, -0.577], [1.0, 1.0, 1.0], 100000)

        for name, geom, mat in _build_initial_scene_geometries(
            wall_display_meshes,
            scene_meshes,
            regrasp_xy_candidates,
            place_height_box_mesh,
        ):
            sc.add_geometry(name, geom, mat)

        for sid in pose_identified_ids:
            if selection.get(sid):
                sc.modify_geometry_material(f"stone_{sid}", _stone_material(True))

        bounds = sc.bounding_box
        scene_widget.setup_camera(60.0, bounds, bounds.get_center())
        sc.camera.look_at(
            np.asarray(camera_center, dtype=float),
            np.asarray(camera_position, dtype=float),
            np.array([0.0, 0.0, 1.0]),
        )

        preview_done = threading.Event()
        preview_timed_out = {"value": False}

        def close_after_timeout():
            if preview_done.wait(INITIAL_PREVIEW_TIMEOUT_SECONDS):
                return

            def close_window():
                if preview_done.is_set():
                    return
                preview_timed_out["value"] = True
                win.close()

            try:
                app.post_to_main_thread(win, close_window)
            except Exception:
                pass

        if INITIAL_PREVIEW_TIMEOUT_SECONDS > 0.0:
            timer = threading.Thread(
                target=close_after_timeout,
                name="initial-preview-timeout",
                daemon=True,
            )
            timer.start()
            print(
                "[INFO] Initial scene preview auto-closes after "
                f"{INITIAL_PREVIEW_TIMEOUT_SECONDS:.0f}s."
            )

        try:
            app.run()
        finally:
            preview_done.set()

        if preview_timed_out["value"]:
            print("[INFO] Initial scene preview timed out, starting planning.")
        else:
            print("[INFO] Interactive preview closed, starting planning.")
        if not pose_identified_ids:
            return None
        return sorted(sid for sid, checked in selection.items() if checked)
    except Exception as exc:
        print(f"[WARN] Could not open interactive scene preview: {exc}")
        return None


def _select_stone_render_meshes(dsf_meshes, meshes):
    if STONE_RENDER_MESH_SOURCE == "dsf":
        return dsf_meshes
    if STONE_RENDER_MESH_SOURCE == "mesh":
        missing = sorted(set(dsf_meshes) - set(meshes))
        if missing:
            print(
                "[generate_sequence] High-poly mesh missing for stone ids "
                f"{missing}; falling back to DSF mesh for those stones."
            )
        return {sid: meshes.get(sid, dsf_meshes[sid]) for sid in dsf_meshes}
    raise ValueError(
        "STONE_RENDER_MESH_SOURCE must be either 'dsf' or 'mesh', got "
        f"{STONE_RENDER_MESH_SOURCE!r}"
    )


def _format_elapsed(seconds):
    if seconds < 60.0:
        return f"{seconds:.2f}s"

    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:05.2f}s"
    return f"{minutes:d}m {sec:05.2f}s"


def _print_timing(label, start_time):
    elapsed = time.perf_counter() - start_time
    print(f"[Timing] {label}: {_format_elapsed(elapsed)}")
    return elapsed


def main(argv=None):
    args = _parse_args(argv)
    total_start = time.perf_counter()
    save_dir = None

    try:
        ## Parameters ##
        q_home = Q_HOME
        target_structure_offset = np.asarray(
            TARGET_STRUCTURE_OFFSET, dtype=float
        ).copy()
        n_move = N_MOVE
        n_grasp = N_GRASP
        n_opening_angle = N_OPENING_ANGLE
        regrasp_xy_pos = np.asarray(REGRASP_XY_POS, dtype=float).copy()
        regrasp_xy_candidates = None
        max_num_regrasp_solutions = 1
        pick_plane_height = float(PICK_PLANE_HEIGHT)
        place_plane_height = float(PLACE_PLANE_HEIGHT)
        place_z_offset = float(PLACE_Z_OFFSET)
        inhand_replan_mode = str(INHAND_REPLAN_MODE).strip().lower()
        if inhand_replan_mode not in {"direct", "regrasp"}:
            raise ValueError("INHAND_REPLAN_MODE must be either 'direct' or 'regrasp'")

        setup_start = time.perf_counter()
        sessions_dir = "sessions"
        resume_state_pkl = None
        source_plan_dir = None
        branch_from_resume_state = False
        branch_to_new_plan = False
        sceneid_ground_height = None

        if args.resume is not None:
            source_plan_dir = _resolve_plan_dir(args.resume)
            if not os.path.isdir(source_plan_dir):
                raise FileNotFoundError(
                    f"Resume directory not found: {source_plan_dir}"
                )
            print(f"Resuming from {source_plan_dir}")
            resume_state_pkl = _resolve_resume_state_pkl(
                source_plan_dir, args.resume_state_pkl
            )
            if not os.path.exists(resume_state_pkl):
                raise FileNotFoundError(
                    f"Resume state pickle not found: {resume_state_pkl}"
                )
            if os.path.basename(resume_state_pkl) != "state.pkl":
                print(f"[INFO] Using resume state pickle: {resume_state_pkl}")
            branch_from_resume_state = args.resume_state_pkl is not None
            branch_to_new_plan = (
                branch_from_resume_state or args.branch_start_step is not None
            )
            if branch_to_new_plan:
                save_dir = _new_plan_dir(sessions_dir)
                os.makedirs(save_dir, exist_ok=True)
                print(
                    "[INFO] Writing resumed branch to new plan directory: "
                    f"{save_dir}"
                )
            else:
                save_dir = source_plan_dir
            sceneid_ground_height = _sceneid_resume_ground_height(resume_state_pkl)
            if sceneid_ground_height is not None:
                place_plane_height = sceneid_ground_height
                print(
                    "[INFO] Using sceneid ground_height as resumed "
                    f"place_plane_height: {place_plane_height:.4f}"
                )
        else:
            save_dir = _new_plan_dir(sessions_dir)
            os.makedirs(save_dir, exist_ok=True)
            resume_state_pkl = os.path.join(save_dir, "state.pkl")

        source_data_dir = source_plan_dir if source_plan_dir is not None else save_dir

        planning_params = {
            "q_home": q_home,
            "target_structure_offset": target_structure_offset,
            "n_move": n_move,
            "n_grasp": n_grasp,
            "n_opening_angle": n_opening_angle,
            "regrasp_xy_pos": regrasp_xy_pos,
            "regrasp_xy_candidates": regrasp_xy_candidates,
            "max_num_regrasp_solutions": max_num_regrasp_solutions,
            "pick_plane_height": pick_plane_height,
            "place_plane_height": place_plane_height,
            "place_z_offset": place_z_offset,
            "inhand_replan_mode": inhand_replan_mode,
            "ground_first": bool(args.ground_first),
        }
        if args.config is not None:
            planning_params["config_path"] = os.path.abspath(args.config)
        if args.config_override:
            planning_params["config_overrides"] = list(args.config_override)
        if source_plan_dir is not None:
            planning_params["resumed_from"] = _resume_source_metadata(
                source_plan_dir,
                resume_state_pkl,
                branch_to_new_plan,
            )
        if args.resume_start_step is not None:
            planning_params["resume_start_step_override"] = int(args.resume_start_step)
        if args.branch_start_step is not None:
            planning_params["branch_start_step"] = int(args.branch_start_step)

        video_dir = get_unique_dir(os.path.join(save_dir, "videos"), prefix="motion")
        os.makedirs(video_dir, exist_ok=True)

        action_save_path = os.path.join(save_dir, "action_sequence.pkl")
        motion_save_path = os.path.join(save_dir, "motion_sequence.pkl")
        motion_result_save_path = os.path.join(save_dir, "motion_result_sequence.pkl")
        params_save_path = os.path.join(save_dir, "planning_params.pkl")
        config_save_path = os.path.join(save_dir, "config.yml")
        pose_data_save_path = os.path.join(save_dir, "pose_data.pkl")
        motion_log_path = os.path.join(save_dir, "motion_log.txt")
        source_action_save_path = os.path.join(source_data_dir, "action_sequence.pkl")
        source_motion_save_path = os.path.join(source_data_dir, "motion_sequence.pkl")
        source_motion_result_save_path = os.path.join(
            source_data_dir, "motion_result_sequence.pkl"
        )
        source_config_path = os.path.join(source_data_dir, "config.yml")
        source_pose_data_path = os.path.join(source_data_dir, "pose_data.pkl")

        resume_planning_params = (
            _load_resume_planning_params(
                os.path.join(source_data_dir, "planning_params.pkl")
            )
            if args.resume is not None
            else {}
        )
        if args.resume is not None:
            place_plane_height = _resolve_resume_place_plane_height(
                resume_planning_params,
                resume_state_pkl,
                sceneid_ground_height,
                place_plane_height,
            )
            planning_params["place_plane_height"] = place_plane_height
        if args.target_structure_offset is None:
            resumed_offset = resume_planning_params.get("target_structure_offset", None)
            if resumed_offset is not None:
                try:
                    target_structure_offset = np.asarray(
                        resumed_offset,
                        dtype=float,
                    ).reshape(2)
                    print(
                        "[INFO] Using resumed target structure offset: "
                        f"{target_structure_offset.tolist()}"
                    )
                except (TypeError, ValueError):
                    print(
                        "[WARN] Ignoring invalid resumed target_structure_offset: "
                        f"{resumed_offset!r}"
                    )
        else:
            target_structure_offset = np.asarray(
                args.target_structure_offset,
                dtype=float,
            ).reshape(2)
            print(
                "[INFO] Using target structure offset override: "
                f"{target_structure_offset.tolist()}"
            )

        if args.place_z_offset is None:
            resumed_place_z_offset = _finite_float(
                resume_planning_params.get("place_z_offset", None)
            )
            if resumed_place_z_offset is not None:
                place_z_offset = resumed_place_z_offset
                print("[INFO] Using resumed place z offset: " f"{place_z_offset:.4f}")
        else:
            place_z_offset = float(args.place_z_offset)
            print("[INFO] Using place z offset override: " f"{place_z_offset:.4f}")

        if args.regrasp_xy_pos is None:
            resumed_regrasp_xy = resume_planning_params.get("regrasp_xy_pos", None)
            if resumed_regrasp_xy is not None:
                try:
                    regrasp_xy_pos = np.asarray(
                        resumed_regrasp_xy,
                        dtype=float,
                    ).reshape(2)
                    print(
                        "[INFO] Using resumed regrasp xy: " f"{regrasp_xy_pos.tolist()}"
                    )
                except (TypeError, ValueError):
                    print(
                        "[WARN] Ignoring invalid resumed regrasp_xy_pos: "
                        f"{resumed_regrasp_xy!r}"
                    )
            resumed_candidates = resume_planning_params.get(
                "regrasp_xy_candidates",
                None,
            )
            if resumed_candidates is not None:
                try:
                    regrasp_xy_candidates = [
                        np.asarray(candidate, dtype=float).reshape(2)
                        for candidate in resumed_candidates
                    ]
                    print(
                        "[INFO] Using resumed regrasp xy candidates: "
                        f"{len(regrasp_xy_candidates)}"
                    )
                except (TypeError, ValueError):
                    print("[WARN] Ignoring invalid resumed regrasp_xy_candidates")
                    regrasp_xy_candidates = None
        else:
            regrasp_xy_pos = np.asarray(args.regrasp_xy_pos, dtype=float).reshape(2)
            print("[INFO] Using regrasp xy override: " f"{regrasp_xy_pos.tolist()}")

        if regrasp_xy_candidates is None or args.regrasp_xy_pos is not None:
            regrasp_xy_candidates = regrasp_position_candidates(regrasp_xy_pos)
        planning_params["target_structure_offset"] = target_structure_offset
        planning_params["regrasp_xy_pos"] = regrasp_xy_pos
        planning_params["regrasp_xy_candidates"] = regrasp_xy_candidates
        planning_params["place_z_offset"] = float(place_z_offset)

        camera_center = [0, 0, 0]
        camera_position = [6.0, 7.0, 4.0]
        ################

        cfg_path = _resolve_config_path(args, source_config_path)
        cfg = OmegaConf.load(cfg_path)
        cfg = _apply_config_overrides(cfg, args.config_override)
        floor_fill_cfg = cfg.environment.action.planar.floor_fill
        hard_gate_key = (
            "environment.action.planar.floor_fill."
            f"{LOWER_FLOOR_FILL_REJECT_STACKED_KEY}"
        )
        legacy_hard_gate_key = (
            "environment.action.planar.floor_fill."
            f"{LEGACY_GROUND_FILL_REJECT_STACKED_KEY}"
        )
        hard_gate_overridden = any(
            _config_overrides_key(args.config_override, key)
            for key in (hard_gate_key, legacy_hard_gate_key)
        )
        if _config_overrides_key(args.config_override, legacy_hard_gate_key):
            loaded_lower_floor_fill_reject_stacked = bool(
                floor_fill_cfg.get(LEGACY_GROUND_FILL_REJECT_STACKED_KEY, False)
            )
            floor_fill_cfg[LOWER_FLOOR_FILL_REJECT_STACKED_KEY] = (
                loaded_lower_floor_fill_reject_stacked
            )
            del floor_fill_cfg[LEGACY_GROUND_FILL_REJECT_STACKED_KEY]
        else:
            loaded_lower_floor_fill_reject_stacked = (
                migrate_lower_floor_fill_reject_stacked(floor_fill_cfg)
            )
        if args.ground_first:
            floor_fill_cfg[LOWER_FLOOR_FILL_REJECT_STACKED_KEY] = True
            _record_run_config_change(
                planning_params,
                hard_gate_key,
                loaded_lower_floor_fill_reject_stacked,
                True,
                "--ground-first",
            )
            if not _config_overrides_key(
                args.config_override,
                "environment.action.planar.floor_fill.lower_floor_fill_ratio",
            ):
                old_ratio = float(floor_fill_cfg.get("lower_floor_fill_ratio", 0.90))
                floor_fill_cfg.lower_floor_fill_ratio = max(
                    old_ratio,
                    0.95,
                )
                _record_run_config_change(
                    planning_params,
                    "environment.action.planar.floor_fill.lower_floor_fill_ratio",
                    old_ratio,
                    float(floor_fill_cfg.lower_floor_fill_ratio),
                    "--ground-first default",
                )
            if not _config_overrides_key(
                args.config_override,
                "environment.action.planar.floor_fill.lower_floor_occupancy_grid",
            ):
                old_grid = int(floor_fill_cfg.get("lower_floor_occupancy_grid", 32))
                floor_fill_cfg.lower_floor_occupancy_grid = max(
                    old_grid,
                    64,
                )
                _record_run_config_change(
                    planning_params,
                    "environment.action.planar.floor_fill.lower_floor_occupancy_grid",
                    old_grid,
                    int(floor_fill_cfg.lower_floor_occupancy_grid),
                    "--ground-first default",
                )
            if not _config_overrides_key(
                args.config_override,
                "algorithm.mcts.max_depth",
            ):
                old_max_depth = int(cfg.algorithm.mcts.max_depth)
                # max_depth counts the deepest node depth (root = 0), so 1 means
                # root children only: rank immediate placements, no lookahead.
                cfg.algorithm.mcts.max_depth = 1
                _record_run_config_change(
                    planning_params,
                    "algorithm.mcts.max_depth",
                    old_max_depth,
                    int(cfg.algorithm.mcts.max_depth),
                    "--ground-first default",
                )
            print(
                "[INFO] Ground-first mode enabled: rejecting candidates above "
                "the active lower floor until it is filled "
                f"(ratio={float(floor_fill_cfg.lower_floor_fill_ratio):.2f}, "
                f"grid={int(floor_fill_cfg.lower_floor_occupancy_grid)}, "
                f"mcts_max_depth={int(cfg.algorithm.mcts.max_depth)})."
            )
        elif loaded_lower_floor_fill_reject_stacked and not hard_gate_overridden:
            floor_fill_cfg[LOWER_FLOOR_FILL_REJECT_STACKED_KEY] = False
            _record_run_config_change(
                planning_params,
                hard_gate_key,
                True,
                False,
                "current run omits --ground-first",
            )
            print(
                "[INFO] Ground-first mode disabled for this run: cleared "
                "resumed lower_floor_fill_reject_stacked hard gate."
            )
        elif hard_gate_overridden:
            print(
                "[INFO] Using explicit lower_floor_fill_reject_stacked override: "
                f"{lower_floor_fill_reject_stacked(floor_fill_cfg)}"
            )
        planning_params["mcts_max_depth"] = int(cfg.algorithm.mcts.max_depth)
        planning_params.pop(LEGACY_GROUND_FILL_REJECT_STACKED_KEY, None)
        planning_params[LOWER_FLOOR_FILL_REJECT_STACKED_KEY] = bool(
            lower_floor_fill_reject_stacked(floor_fill_cfg)
        )
        if args.simulate_sceneid_stones:
            OmegaConf.update(
                cfg,
                "environment.sim.simulate_pose_identified",
                True,
                force_add=True,
            )
        planning_params["simulate_pose_identified"] = bool(
            cfg.environment.sim.get("simulate_pose_identified", False)
        )
        _set_action_score_excavator_xy(cfg, target_structure_offset)
        old_env_ground_height = environment_ground_height(cfg.environment)
        set_environment_ground_height(cfg.environment, place_plane_height)
        if abs(old_env_ground_height - float(place_plane_height)) > 1e-9:
            print(
                "[INFO] Updating environment support ground_z for planning: "
                f"{old_env_ground_height:.4f} -> {place_plane_height:.4f}"
            )

        pose_data_load_path, pose_data_source = _select_pose_data_load_path(
            args,
            cfg,
            source_pose_data_path,
        )
        poses = _load_pose_data(pose_data_load_path)
        pose_data_refresh_from_config = (
            args.config is not None
            or _config_overrides_key(
                args.config_override,
                "environment.action.pose_data_path",
            )
        )
        copy_pose_data_to_session = (
            args.resume is None
            or branch_to_new_plan
            or pose_data_refresh_from_config
        )
        pose_data_runtime_path = pose_data_load_path
        if copy_pose_data_to_session:
            pose_data_runtime_path = _write_pose_data_copy(
                poses,
                pose_data_load_path,
                pose_data_save_path,
            )
        cfg.environment.action.pose_data_path = pose_data_runtime_path
        planning_params["poses"] = poses
        planning_params["pose_data_path"] = pose_data_runtime_path
        if not _same_existing_file(pose_data_load_path, pose_data_runtime_path):
            planning_params["pose_data_source_path"] = pose_data_load_path
        print(
            "[INFO] Using pose_data from "
            f"{pose_data_load_path} ({pose_data_source}); "
            f"planner path: {pose_data_runtime_path}"
        )

        integrated_planner = IntegratedPlanner(cfg, parallel=False, use_ros=False)

        debug_pkl_path = os.path.join(save_dir, "state.pkl")
        debug_data = (
            _load_debug_data_or_init(resume_state_pkl, integrated_planner.env)
            if args.resume is not None and not branch_from_resume_state
            else _init_debug_data(integrated_planner.env)
        )
        _refresh_debug_target_wall(
            debug_data,
            integrated_planner.env,
            reason="effective config",
        )
        if source_plan_dir is not None:
            debug_data["resumed_from"] = _resume_source_metadata(
                source_plan_dir,
                resume_state_pkl,
                branch_to_new_plan,
            )

        # Wall meshes shifted to real-world coords for trajectory videos.
        wall_display_meshes = {}
        for _i, _geom in enumerate(
            integrated_planner.env.inventory.target_wall.geometries
        ):
            _wmesh = copy.deepcopy(_geom.get_mesh())
            _wmesh.translate(
                [target_structure_offset[0], target_structure_offset[1], 0.0]
            )
            _wmesh.compute_vertex_normals()
            wall_display_meshes[f"wall_{_i}"] = _wmesh

        place_height_box_mesh = None
        show_place_height_box = (
            abs(float(place_plane_height)) > 1e-9
            or abs(float(place_plane_height) - float(pick_plane_height)) > 1e-9
        )
        if show_place_height_box:
            place_height_box_mesh = _make_place_height_box_mesh(
                wall_display_meshes,
                pick_plane_height,
                place_plane_height,
            )
            if place_height_box_mesh is not None:
                print(
                    "[INFO] Showing place-height box in initial preview: "
                    f"top z={place_plane_height:.4f}"
                )

        context, _ = get_planner(
            pick_plane_height,
            place_plane_height,
            n_threads=resolve_thread_count(20, cfg),
        )

        excavator_model, excavator_meshes = get_excavator_model()
        stone_dsf_meshes, stone_configs, _, stone_meshes_highpoly = get_stone_model(
            cfg.environment.data.load_dir
        )
        stone_meshes = _select_stone_render_meshes(
            stone_dsf_meshes, stone_meshes_highpoly
        )
        _backfill_debug_stone_meshes(debug_data, stone_dsf_meshes)
        _set_debug_coordinate_metadata(debug_data, target_structure_offset)
        if args.resume is not None and not branch_to_new_plan:
            _save_debug_data(debug_data, debug_pkl_path)

        # Persist run-invariant artefacts up-front. A SceneID resume branches
        # into a new session directory, so it needs its own copies.
        if args.resume is None or branch_to_new_plan:
            with open(params_save_path, "wb") as f:
                pickle.dump(planning_params, f)
            OmegaConf.save(cfg, config_save_path)
            with open(pose_data_save_path, "wb") as f:
                pickle.dump(poses, f)
        elif args.resume is not None:
            _update_resume_planning_params(
                params_save_path,
                place_plane_height,
                inhand_replan_mode,
                target_structure_offset,
                regrasp_xy_pos,
                regrasp_xy_candidates,
                ground_first=bool(args.ground_first),
                lower_floor_fill_reject_stacked_enabled=bool(
                    planning_params[LOWER_FLOOR_FILL_REJECT_STACKED_KEY]
                ),
                mcts_max_depth=int(planning_params["mcts_max_depth"]),
                run_config_changes=planning_params.get("run_config_changes", []),
                poses=poses,
                pose_data_path=pose_data_runtime_path,
                pose_data_source_path=pose_data_load_path,
                refresh_poses=pose_data_refresh_from_config,
            )
            OmegaConf.save(cfg, config_save_path)

        n_step = 0
        action_sequence = []
        motion_sequence = []
        motion_result_sequence = []
        resume_state = None
        action_poses = {}
        pose_identified_ids = []
        placed_ids = set()
        overwrite_resume_tail = False

        if args.resume is not None:
            resume_state, n_step = _load_resume_state(
                resume_state_pkl, integrated_planner.env
            )
            resume_start_step = (
                args.resume_start_step
                if args.resume_start_step is not None
                else args.branch_start_step
            )
            if resume_start_step is not None:
                n_step = int(resume_start_step) - 1
                _truncate_resume_state(resume_state, n_step)

            if os.path.exists(source_action_save_path):
                with open(source_action_save_path, "rb") as f:
                    action_sequence = pickle.load(f)
            if os.path.exists(source_motion_save_path):
                with open(source_motion_save_path, "rb") as f:
                    motion_sequence = pickle.load(f)
            if os.path.exists(source_motion_result_save_path):
                with open(source_motion_result_save_path, "rb") as f:
                    motion_result_sequence = pickle.load(f)
            if len(motion_result_sequence) < len(motion_sequence):
                motion_result_sequence.extend(
                    [None] * (len(motion_sequence) - len(motion_result_sequence))
                )
            overwrite_resume_tail = (
                branch_from_resume_state
                or args.resume_start_step is not None
                or args.branch_start_step is not None
            )
            if overwrite_resume_tail:
                if (
                    len(action_sequence) > n_step
                    or len(motion_sequence) > n_step
                    or len(motion_result_sequence) > n_step
                ):
                    _backup_resume_trim_inputs(
                        save_dir,
                        [
                            action_save_path,
                            motion_save_path,
                            motion_result_save_path,
                            params_save_path,
                            debug_pkl_path,
                        ],
                    )
                action_sequence, motion_sequence, motion_result_sequence = (
                    _trim_resume_sequences(
                        action_sequence,
                        motion_sequence,
                        motion_result_sequence,
                        n_step,
                    )
                )
                with open(action_save_path, "wb") as f:
                    pickle.dump(action_sequence, f)
                with open(motion_save_path, "wb") as f:
                    pickle.dump(motion_sequence, f)
                with open(motion_result_save_path, "wb") as f:
                    pickle.dump(motion_result_sequence, f)
                planning_params["resume_step"] = int(n_step)
                planning_params["execution_start_step"] = int(n_step) + 1
                with open(params_save_path, "wb") as f:
                    pickle.dump(planning_params, f)
                print(
                    "[INFO] Execution will start from resumed branch step "
                    f"{planning_params['execution_start_step']}."
                )

            _fill_missing_resume_stone_poses(resume_state, action_sequence)
            if branch_from_resume_state and pose_data_refresh_from_config:
                _remap_resume_state_to_current_inventory(
                    resume_state,
                    integrated_planner.env.inventory.stone_set,
                )
            if overwrite_resume_tail:
                action_sequence, motion_sequence, motion_result_sequence = (
                    _sync_resume_sequences_to_state_prefix(
                        action_sequence,
                        motion_sequence,
                        motion_result_sequence,
                        resume_state,
                        n_step,
                    )
                )
                with open(action_save_path, "wb") as f:
                    pickle.dump(action_sequence, f)
                with open(motion_save_path, "wb") as f:
                    pickle.dump(motion_sequence, f)
                with open(motion_result_save_path, "wb") as f:
                    pickle.dump(motion_result_sequence, f)
            if overwrite_resume_tail:
                planning_params["resume_scene_poses"] = _resume_scene_pose_dict(
                    resume_state
                )
                with open(params_save_path, "wb") as f:
                    pickle.dump(planning_params, f)
            elif branch_to_new_plan:
                planning_params["resume_scene_poses"] = _resume_scene_pose_dict(
                    resume_state
                )
                with open(params_save_path, "wb") as f:
                    pickle.dump(planning_params, f)
            integrated_planner.env.update_from_state(resume_state)
            if overwrite_resume_tail:
                _trim_debug_data_steps(debug_data, n_step)
                debug_data["resume_state"] = copy.deepcopy(resume_state)
                debug_data["resume_step"] = n_step
                _save_debug_data(debug_data, debug_pkl_path)
            elif branch_to_new_plan:
                debug_data["resume_state"] = copy.deepcopy(resume_state)
                debug_data["resume_step"] = n_step
                _save_debug_data(debug_data, debug_pkl_path)
            action_poses = _action_pose_by_stone_id(action_sequence)
            pose_identified_ids = sorted(_pose_identified_stone_ids(resume_state))
            placed_ids = {
                int(resume_state.stone_set[int(idx)]) for idx in resume_state.stone_seq
            }

        pick_ids = {}
        scene_meshes = {}
        scene_configs = {}
        for id, mesh in stone_meshes.items():
            if id in placed_ids:
                sim_pose = np.asarray(
                    resume_state.stone_poses.get(
                        id,
                        action_poses.get(id),
                    ),
                    dtype=float,
                )
                if sim_pose.shape[0] != 7:
                    raise ValueError(
                        f"Could not recover a valid resume pose for stone {id}"
                    )
                place_cfg = _place_config_from_state_pose(
                    stone_configs[id],
                    sim_pose,
                    target_structure_offset,
                )
                context.add_place_body(place_cfg)
                mesh = copy.deepcopy(mesh)
                mesh.transform(place_cfg.pose.as_matrix())
                scene_meshes[id] = mesh
                scene_configs[id] = place_cfg
                continue

            if id not in poses:
                continue

            pos, quat = poses[id]
            stone_configs[id].pose.setPosition(pos)
            stone_configs[id].pose.setOrientation(quat)

            pose_T = stone_configs[id].pose.as_matrix()

            mesh = copy.deepcopy(mesh)
            mesh.transform(pose_T)
            scene_meshes[id] = mesh
            scene_configs[id] = stone_configs[id]
            pick_ids[id] = context.add_pick_body(stone_configs[id])

        _print_timing("setup", setup_start)

        if args.resume is not None:
            print(
                f"[INFO] Resumed: {len(resume_state.stone_seq)} stones placed, continuing from step {n_step + 1}"
            )
            if pose_identified_ids:
                if bool(cfg.environment.sim.get("simulate_pose_identified", False)):
                    print(
                        "[INFO] Scene-pose-identified stones will be simulated "
                        "with error_reduction_ratio 0.0 among themselves "
                        f"(default vs new stones): {pose_identified_ids}"
                    )
                else:
                    print(
                        "[INFO] Scene-pose-identified stones will be frozen in "
                        f"planning simulation: {pose_identified_ids}"
                    )

        _render_initial_scene(
            save_dir,
            wall_display_meshes,
            scene_meshes,
            regrasp_xy_candidates,
            place_height_box_mesh,
            camera_center,
            camera_position,
        )
        selected_simulate_ids = _show_initial_scene_interactive(
            wall_display_meshes,
            scene_meshes,
            regrasp_xy_candidates,
            place_height_box_mesh,
            camera_center,
            camera_position,
            pose_identified_ids=pose_identified_ids,
            simulate_checked=bool(
                cfg.environment.sim.get("simulate_pose_identified", False)
            ),
        )
        if pose_identified_ids and selected_simulate_ids is not None:
            selected = sorted(int(i) for i in selected_simulate_ids)
            all_ids = sorted(int(i) for i in pose_identified_ids)
            enabled = bool(selected)
            # None = simulate every identified stone; a list restricts the set.
            ids_value = None if (not enabled or selected == all_ids) else selected
            OmegaConf.update(
                cfg,
                "environment.sim.simulate_pose_identified",
                enabled,
                force_add=True,
            )
            OmegaConf.update(
                cfg,
                "environment.sim.simulate_pose_identified_ids",
                ids_value,
                force_add=True,
            )
            for simulator in (
                integrated_planner.env.simulator,
                integrated_planner.mcts.env.simulator,
            ):
                simulator.set_simulate_pose_identified(enabled, ids_value)
            planning_params["simulate_pose_identified"] = enabled
            planning_params["simulate_pose_identified_ids"] = (
                selected if enabled else []
            )
            with open(params_save_path, "wb") as f:
                pickle.dump(planning_params, f)
            OmegaConf.save(cfg, config_save_path)
            if enabled:
                print(
                    "[INFO] Preview selection: simulating identified stones "
                    f"{selected} with error_reduction_ratio 0.0 (others frozen)."
                )
            else:
                print(
                    "[INFO] Preview selection: all identified stones stay "
                    "frozen in planning simulation."
                )

        while n_step < integrated_planner.env.cfg.n_stone:
            n_step += 1
            step_start = time.perf_counter()
            find_feasible_action = False
            plan_attempt = 0
            execution_rejected_actions = []
            execution_rejected_stone_ids = set()
            motion_fail_counts_by_stone = {}

            while not find_feasible_action:
                plan_attempt += 1
                print("[INFO] Start stacking step ", n_step)
                # MCTS planning
                mcts_start = time.perf_counter()
                state = integrated_planner.env.get_state()
                actions, nodes, debug_nodes = integrated_planner.plan_one_step(
                    state,
                    use_policy=False,
                    use_feasibility_score=True,
                    execution_rejected_actions=execution_rejected_actions,
                    execution_rejected_stone_ids=sorted(execution_rejected_stone_ids),
                )
                _print_timing(f"step {n_step} MCTS attempt {plan_attempt}", mcts_start)
                if actions is None:
                    failure_reason = _no_candidate_failure_reason(
                        state,
                        integrated_planner.env,
                        execution_rejected_stone_ids,
                    )
                    print(
                        "[INFO] No MCTS action candidates returned: "
                        f"{failure_reason}"
                    )
                    _append_debug_step(
                        debug_data,
                        n_step,
                        False,
                        state,
                        debug_nodes or [],
                        integrated_planner.env,
                        attempt=plan_attempt,
                        failure_reason=failure_reason,
                        rejected_stone_ids=sorted(execution_rejected_stone_ids),
                        mcts=integrated_planner.mcts,
                    )
                    _save_debug_data(debug_data, debug_pkl_path)
                    return save_dir

                for action_idx, (action, node) in enumerate(
                    zip(actions, nodes), start=1
                ):
                    target_id = action["stone_id"]

                    candidate_pose = copy.deepcopy(node.state.stone_poses[target_id])
                    reject_stacked, ground_fill = _candidate_stacks_before_ground_fill(
                        state,
                        target_id,
                        candidate_pose,
                        stone_meshes,
                        cfg,
                    )
                    if reject_stacked:
                        print(
                            "[INFO] Rejecting candidate before motion planning: "
                            "ground fill is incomplete "
                            f"({ground_fill['occupancy']:.3f} < "
                            f"{ground_fill['required']:.3f}) and stone "
                            f"{target_id} starts above the ground layer "
                            f"(bottom={ground_fill['candidate_bottom']:.3f}, "
                            f"ground_top={ground_fill['ground_top']:.3f}, "
                            f"margin={ground_fill['margin']:.3f})."
                        )
                        node._motion_failed = True
                        if node.info is None:
                            node.info = {}
                        node.info["execution_rejected"] = True
                        node.info["execution_reject_reason"] = (
                            "stacked_before_ground_fill"
                        )
                        node.info["ground_fill_occupancy"] = float(
                            ground_fill["occupancy"]
                        )
                        node.info["ground_fill_required"] = float(
                            ground_fill["required"]
                        )
                        if getattr(node, "action", None) is not None:
                            execution_rejected_actions.append(node.action.copy())
                        continue

                    # Motion planning
                    place_pose = candidate_pose
                    place_pose[:2] += target_structure_offset[:2]
                    print("[INFO] Selected stone id: ", action["stone_id"])
                    print("[INFO] Target stone position: ", place_pose[:3])

                    pick_config = copy.deepcopy(stone_configs[target_id])

                    place_config = copy.deepcopy(pick_config)
                    place_config.pose.setPosition(place_pose[:3])
                    place_config.pose.setOrientation(place_pose[3:])
                    consistency = _long_sim_motion_scene_check(
                        state,
                        node.state,
                        target_id,
                        place_config,
                        scene_configs,
                        target_structure_offset,
                    )
                    if node.info is None:
                        node.info = {}
                    node.info.update(consistency)
                    settled_local_pose, resettle_info = (
                        _resettle_place_config_fixed_scene(
                            cfg,
                            state,
                            target_id,
                            place_config,
                            scene_configs,
                            target_structure_offset,
                        )
                    )
                    node.info.update(resettle_info)
                    if settled_local_pose is None:
                        reject_reason = str(
                            resettle_info.get(
                                "fixed_scene_resettle_reject_reason",
                                "invalid",
                            )
                        )
                        print(
                            "[INFO] Rejecting candidate before motion planning: "
                            f"fixed-scene re-settle rejected ({reject_reason})."
                        )
                        node._motion_failed = True
                        node.info["execution_rejected"] = True
                        node.info["execution_reject_reason"] = (
                            f"fixed_scene_resettle_{reject_reason}"
                        )
                        if getattr(node, "action", None) is not None:
                            execution_rejected_actions.append(node.action.copy())
                        continue
                    reject_active_floor, active_floor_status = (
                        _candidate_rejects_active_floor(
                            integrated_planner.env,
                            state,
                            target_id,
                            settled_local_pose,
                        )
                    )
                    node.info.update(active_floor_status)
                    if reject_active_floor:
                        reject_reason = str(
                            active_floor_status.get(
                                "active_floor_reject_reason",
                                "invalid",
                            )
                        )
                        print(
                            "[INFO] Rejecting candidate before motion planning: "
                            f"active-floor check rejected ({reject_reason}, "
                            "contact="
                            f"{active_floor_status.get('fixed_scene_active_layer_contact_cells')}, "
                            "min_contact="
                            f"{active_floor_status.get('fixed_scene_active_layer_min_contact_cells')})."
                        )
                        node._motion_failed = True
                        node.info["execution_rejected"] = True
                        node.info["execution_reject_reason"] = (
                            f"active_floor_{reject_reason}"
                        )
                        if getattr(node, "action", None) is not None:
                            execution_rejected_actions.append(node.action.copy())
                        continue
                    candidate_pose = settled_local_pose
                    for stone_id, pose in getattr(state, "stone_poses", {}).items():
                        node.state.stone_poses[int(stone_id)] = np.asarray(
                            pose,
                            dtype=float,
                        ).copy()
                    node.state.stone_poses[target_id] = settled_local_pose.copy()
                    node.info["fixed_scene_commit_preserves_parent_scene"] = True
                    if getattr(node, "action", None) is not None:
                        node.action.pose = settled_local_pose.copy()
                    place_pose = settled_local_pose.copy()
                    place_pose[:2] += target_structure_offset[:2]
                    if abs(float(place_z_offset)) > 1e-12:
                        place_pose[2] += float(place_z_offset)
                        place_config.pose.setPosition(place_pose[:3])
                        node.info["place_z_offset"] = float(place_z_offset)
                        node.info["place_motion_target_position"] = place_pose[
                            :3
                        ].copy()
                        print(
                            "[INFO] Re-settled target stone position with "
                            f"place_z_offset={place_z_offset:.4f}: ",
                            place_pose[:3],
                        )
                    else:
                        print(
                            "[INFO] Re-settled target stone position: ", place_pose[:3]
                        )

                    motion_start = time.perf_counter()
                    context.remove_body(pick_ids[target_id])
                    result = None
                    # Single regrasp_xy position (multi-position retry removed):
                    # the intermediate scene C is placed at the one regrasp_xy.
                    selected_regrasp_xy = np.asarray(regrasp_xy_pos, dtype=float).copy()
                    result = regrasp_planning(
                        context,
                        pick_config,
                        place_config,
                        q_home,
                        regrasp_xy_pos,
                        n_move,
                        n_grasp,
                        max_num_regrasp_solutions,
                    )
                    failure_stage = motion_failure_stage(result)
                    failure_detail = motion_failure_detail(result)
                    place_pick_transform = (
                        place_config.pose.as_matrix()
                        @ np.linalg.inv(pick_config.pose.as_matrix())
                    )
                    node._failed_grasps = _serializable_failed_grasps(
                        result, place_pick_transform
                    )
                    if failure_stage == "interrupted":
                        raise KeyboardInterrupt
                    motion_elapsed = time.perf_counter() - motion_start
                    print(
                        f"[INFO] Motion planning result: "
                        f"{motion_result_summary(result)}"
                    )
                    _print_timing(
                        f"step {n_step} motion attempt {action_idx}", motion_start
                    )
                    with open(motion_log_path, "a") as _mlog:
                        place_world_y = _node_world_y(
                            node,
                            target_structure_offset,
                        )
                        _mlog.write(
                            f"step={n_step} plan_attempt={plan_attempt}"
                            f" action_idx={action_idx} stone_id={target_id}"
                            f" place_world_y={place_world_y}"
                            f" place_z_offset={float(place_z_offset)}"
                            f" regrasp_xy={selected_regrasp_xy.tolist()}"
                            f" feasible={bool(result.is_feasible)}"
                            f" elapsed={motion_elapsed:.2f}s"
                            f" failure_stage={failure_stage!r}"
                            f" failure_detail={failure_detail!r}"
                            f"\n"
                        )
                    if not result.is_feasible:
                        print(
                            "[INFO] No feasible motion plan is found for this action, try another action..."
                        )
                        node._motion_failed = True
                        if node.info is None:
                            node.info = {}
                        node.info["motion_planning_failed"] = True
                        node.info["motion_attempt"] = int(action_idx)
                        node.info["motion_failure_stage"] = failure_stage
                        node.info["motion_failure_detail"] = failure_detail
                        node.info["motion_regrasp_xy"] = selected_regrasp_xy.copy()
                        if getattr(node, "action", None) is not None:
                            execution_rejected_actions.append(node.action.copy())
                        motion_fail_counts_by_stone[target_id] = (
                            motion_fail_counts_by_stone.get(target_id, 0) + 1
                        )
                        if (
                            motion_fail_counts_by_stone[target_id]
                            >= STONE_REJECTION_THRESHOLD
                        ):
                            execution_rejected_stone_ids.add(int(target_id))
                        pick_ids[target_id] = context.add_pick_body(
                            stone_configs[target_id]
                        )
                        continue
                    else:
                        find_feasible_action = True
                        pick_ids.pop(target_id, None)
                        break
                if not find_feasible_action:
                    print(
                        "All returned MCTS actions failed motion planning; "
                        f"retrying MCTS with {len(execution_rejected_actions)} "
                        "execution-rejected actions and "
                        f"{len(execution_rejected_stone_ids)} rejected stones."
                    )
                    _append_debug_step(
                        debug_data,
                        n_step,
                        False,
                        state,
                        debug_nodes or nodes,
                        integrated_planner.env,
                        resume_state=state,
                        attempt=plan_attempt,
                        failure_reason="motion_planning_failed",
                        rejected_stone_ids=sorted(execution_rejected_stone_ids),
                        mcts=integrated_planner.mcts,
                    )
                    _save_debug_data(debug_data, debug_pkl_path)
            update_start = time.perf_counter()
            integrated_planner.env.update_from_state(node.state)
            committed_place_config = _place_config_from_state_pose(
                stone_configs[target_id],
                node.state.stone_poses[target_id],
                target_structure_offset,
            )
            context.add_place_body(committed_place_config)
            scene_configs[target_id] = copy.deepcopy(committed_place_config)
            action_sequence.append(
                {
                    "stone_id": action["stone_id"],
                    "pose": node.state.stone_poses[target_id],
                    "place_z_offset": float(place_z_offset),
                }
            )
            path1, path2, path3, path4 = split_regrasp_place_paths(
                result,
                n_move,
                n_grasp,
                q_home=q_home,
            )
            motion_sequence.append([path1, path2, path3, path4])
            motion_result_sequence.append(_serializable_motion_result(result))
            action_sequence[-1]["regrasp_xy_pos"] = regrasp_xy_pos.copy()
            action_sequence[-1]["selected_regrasp_xy_pos"] = selected_regrasp_xy.copy()
            action_sequence[-1]["motion_mode"] = (
                "direct" if len(result.q_path_sequence) == 2 else "regrasp"
            )
            selected_motion_mode = action_sequence[-1]["motion_mode"]
            _print_timing(f"step {n_step} state update", update_start)

            _append_debug_step(
                debug_data,
                n_step,
                True,
                state,
                debug_nodes or nodes,
                integrated_planner.env,
                resume_state=node.state,
                selected_node=node,
                selected_action_idx=action_idx,
                selected_regrasp_xy=selected_regrasp_xy,
                selected_motion_mode=selected_motion_mode,
                mcts=integrated_planner.mcts,
            )
            _save_debug_data(debug_data, debug_pkl_path)

            visualization_start = time.perf_counter()
            save_path = os.path.join(video_dir, f"step_{n_step}.mp4")
            q_path, target_path = generate_path_with_opening_angle(
                result, n_opening_angle
            )
            scene_meshes.pop(target_id)
            trajectory_visualization_with_target(
                q_path,
                target_path,
                excavator_model,
                excavator_meshes,
                scene_meshes,
                stone_meshes[target_id],
                save_path,
                camera_center,
                camera_position,
                wall_meshes=wall_display_meshes,
            )

            target_mesh = copy.deepcopy(stone_meshes[target_id])
            target_mesh.transform(committed_place_config.pose.as_matrix())
            scene_meshes[target_id] = target_mesh
            _print_timing(f"step {n_step} visualization", visualization_start)

            checkpoint_start = time.perf_counter()
            # Checkpoint after every step so the run can be interrupted (Ctrl-C,
            # crash) without losing the steps planned so far.
            with open(action_save_path, "wb") as f:
                pickle.dump(action_sequence, f)
            with open(motion_save_path, "wb") as f:
                pickle.dump(motion_sequence, f)
            with open(motion_result_save_path, "wb") as f:
                pickle.dump(motion_result_sequence, f)
            print(f"Saved progress through step {n_step} to {save_dir}")
            _print_timing(f"step {n_step} checkpoint", checkpoint_start)
            _print_timing(f"step {n_step} total", step_start)

            if node.done:
                break
    except KeyboardInterrupt:
        print("[INFO] generate_sequence interrupted by Ctrl+C")
        if argv is None:
            sys.exit(130)
        raise
    finally:
        _print_timing("total generate_sequence", total_start)
    return save_dir


if __name__ == "__main__":
    with _NativeStderrFilter():
        main()
