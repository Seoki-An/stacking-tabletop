import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
from diffsimpy import diffsim
from scipy.spatial.transform import Rotation

from agent.config_views import support_config
from agent.env.components.action import Action
from agent.env.components.state import State

from .node import MCTS_Node


def best_sequence_from_node(node: MCTS_Node, cfg=None) -> List[dict]:
    sequence = []
    current = node
    root = node
    while current is not None and current.action is not None:
        solved_pose = solved_action_pose(current.action)
        settled_pose = np.asarray(current.action.pose, dtype=float)
        sequence.append(
            {
                "stone_idx": int(current.action.stone_idx),
                "stone_id": int(current.action.stone_id),
                "pose": settled_pose.copy(),
                "solved_pose": solved_pose.copy(),
                "settled_pose": settled_pose.copy(),
                "init_pose": np.asarray(current.action.init_pose, dtype=float).copy(),
                "reward": None if current.reward is None else float(current.reward),
                "q_value": float(current.q_value),
                "q_value_init": float(current.q_value_init),
                "value_init": float(current.value_init),
                "visits": int(current.visits),
                "failed": bool(current.failed),
                "info": dict(current.info or {}),
            }
        )
        current = best_sequence_child(current, cfg, root)
    return sequence


def best_sequence_child(
    node: MCTS_Node,
    cfg=None,
    root: Optional[MCTS_Node] = None,
) -> Optional[MCTS_Node]:
    children = node.simulated_children(include_failed=False)
    if cfg is not None and root is not None:
        selection = cfg.get("final_selection", {})
        if bool(selection.get("validate_descendant_path", False)):
            validation_depth = max(
                int(selection.get("validate_descendant_depth", 1)),
                0,
            )
            depth_from_root = int(node.depth) - int(root.depth)
            if depth_from_root >= validation_depth:
                return None
            children = [
                child
                for child in children
                if bool(
                    (child.info or {}).get(
                        "descendant_path_final_validated",
                        False,
                    )
                )
            ]
    if not children:
        return None
    return max(
        children,
        key=lambda child: best_sequence_child_key(node, child, cfg, root),
    )


def best_sequence_child_key(
    parent: MCTS_Node,
    child: MCTS_Node,
    cfg=None,
    root: Optional[MCTS_Node] = None,
) -> tuple:
    visited = int(child.visits > 0)
    return visited, int(child.visits), child_score_with_value(child)


def child_score_with_value(child: MCTS_Node) -> float:
    if child.visits > 0 and np.isfinite(child.q_value):
        return child.q_value / max(child.visits, 1.0) + child.q_value_init
    reward = 0.0 if child.reward is None else child.reward
    return reward + child.q_value_init + child.value_init


def validated_child_score(child: MCTS_Node) -> float:
    return child_score_with_value(child)


def is_duplicate_action(
    action: Action,
    others: List[Action],
    xy_thresh: float,
    yaw_thresh: float,
) -> bool:
    for other in others:
        if action.stone_id != other.stone_id:
            continue
        if np.linalg.norm(action.pose[:2] - other.pose[:2]) > xy_thresh:
            continue
        yaw_delta = (
            Rotation.from_quat(action.pose[3:])
            * Rotation.from_quat(other.pose[3:]).inv()
        ).as_euler("xyz")[2]
        if abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))) <= yaw_thresh:
            return True
    return False


def top_actions(
    actions: List[Action],
    scores: np.ndarray,
    keep: int,
    max_per_stone: Optional[int] = None,
) -> Tuple[List[Action], np.ndarray]:
    if len(actions) == 0:
        return [], np.array([])
    scores = np.asarray(scores, dtype=float)
    order = [i for i in np.argsort(scores)[::-1] if np.isfinite(scores[i])]
    if max_per_stone and max_per_stone > 0:
        stone_counts: Dict[int, int] = {}
        diverse: List[int] = []
        for i in order:
            sid = actions[i].stone_id
            if stone_counts.get(sid, 0) < max_per_stone:
                stone_counts[sid] = stone_counts.get(sid, 0) + 1
                diverse.append(i)
            if len(diverse) >= keep:
                break
        order = diverse
    else:
        order = order[:keep]
    order_arr = np.array(order, dtype=int)
    return [actions[i] for i in order_arr], scores[order_arr]


def settled_action_pose(env, action: Action, state: State) -> np.ndarray:
    stone = env.inventory.stones[action.stone_idx]
    pose = state.stone_poses.get(stone.id, action.pose)
    return np.asarray(pose, dtype=float)


def solved_action_pose(action: Action) -> np.ndarray:
    """Return the immutable posegen result, with legacy-action fallback."""
    pose = getattr(action, "solved_pose", None)
    if pose is None:
        pose = action.pose
    return np.asarray(pose, dtype=float)


def refined_action_from_state(action: Action, state: State) -> Action:
    """Return the simulator-owned action containing the frozen-settle pose."""
    history = getattr(state, "action_history", None) or []
    if history:
        refined = history[-1]
        if int(refined.stone_id) >= 0 and int(action.stone_id) >= 0:
            same_stone = int(refined.stone_id) == int(action.stone_id)
        else:
            same_stone = int(refined.stone_idx) == int(action.stone_idx)
        if same_stone:
            return refined.copy()
    return action.copy()


def final_support_ok(env, state: State, info_r: Dict) -> bool:
    support = support_config(env)
    if not support.enabled:
        return True
    if len(state.stone_seq) <= 1:
        return True

    count = int(info_r.get("support_count", 0))
    has_ground = bool(info_r.get("support_has_ground", False))
    if has_ground:
        return count >= 1

    min_supports = support.desired_sources
    if count >= max(min_supports, 1):
        return True

    if (
        count >= 1
        and support.stable_single_support_fallback_enabled
        and not bool(info_r.get("place_robustness_nonfinite", False))
        and not bool(info_r.get("place_robustness_clipped", False))
    ):
        max_displacement = support.stable_single_support_max_displacement
        displacement = float(info_r.get("place_robustness_displacement", np.inf))
        return displacement <= max_displacement

    return False


def place_scene_gap_failure_reason(
    env,
    state: State,
    stone_idx: int,
    pose: np.ndarray,
    info: Optional[Dict] = None,
) -> Optional[str]:
    min_gap = -support_config(env).contact_gap_tolerance

    gap = place_scene_min_gap(env, state, stone_idx, pose)
    if info is not None and gap is not None:
        info["place_scene_min_gap"] = float(gap)
        info["place_scene_min_gap_threshold"] = min_gap
        info["place_scene_gap_source"] = "diffsim_dsf_mtd"
    if gap is None:
        return None
    if gap < min_gap:
        return "place_scene_gap"
    return None


def place_plane_gap_failure_reason(
    env,
    stone_idx: int,
    pose: np.ndarray,
    info: Optional[Dict] = None,
) -> Optional[str]:
    min_gap = -support_config(env).contact_gap_tolerance

    gap = place_plane_min_gap(env, stone_idx, pose)
    if info is not None and gap is not None:
        info["place_plane_min_gap"] = float(gap)
        info["place_plane_min_gap_threshold"] = min_gap
        info["place_plane_gap_source"] = "diffsim_dsf_plane_support"
    if gap is None:
        return None
    if gap < min_gap:
        return "place_plane_gap"
    return None


def place_plane_min_gap(env, stone_idx: int, pose: np.ndarray) -> Optional[float]:
    stone = env.inventory.stones[int(stone_idx)]
    ground_z = support_config(env).ground_z
    pose = np.asarray(pose, dtype=float)
    gaps = []
    for geometry in stone.geometries:
        try:
            _, point = geometry.dsf.support(np.array([0.0, 0.0, -1.0]), pose)
            point = np.asarray(point, dtype=float).reshape(3, -1)[:, 0]
        except Exception:
            continue
        if np.all(np.isfinite(point)):
            gaps.append(float(point[2] - ground_z))
    return min(gaps) if gaps else None


def place_scene_min_gap(
    env,
    state: State,
    stone_idx: int,
    pose: np.ndarray,
) -> Optional[float]:
    """Return the deepest DSF collision gap to the fixed placed-stone scene."""
    obstacle_indices = [
        int(idx)
        for idx in getattr(state, "stone_seq", []) or []
        if int(idx) != int(stone_idx)
    ]
    if not obstacle_indices:
        return None

    target = env.inventory.stones[int(stone_idx)]
    target_pose = np.asarray(pose, dtype=float)
    target_lower, target_upper = stone_world_aabb(target, target_pose)
    target_config = body_config_at_pose(target.config, target_pose)
    gaps = []
    for idx in obstacle_indices:
        obstacle = env.inventory.stones[idx]
        obstacle_pose = np.asarray(
            state.stone_poses.get(obstacle.id, obstacle.pose),
            dtype=float,
        )
        obstacle_lower, obstacle_upper = stone_world_aabb(obstacle, obstacle_pose)
        if np.any(obstacle_upper < target_lower) or np.any(
            obstacle_lower > target_upper
        ):
            continue
        try:
            obstacle_config = body_config_at_pose(obstacle.config, obstacle_pose)
            gap = float(diffsim.dsf_dsf_min_gap(target_config, obstacle_config))
        except Exception:
            continue
        if np.isfinite(gap):
            gaps.append(gap)
    return min(gaps) if gaps else None


def body_config_at_pose(config, pose: np.ndarray):
    posed = copy.deepcopy(config)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    matrix[:3, 3] = pose[:3]
    posed.pose = diffsim.Pose().from_matrix(matrix)
    return posed


def transform_vertices(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rot = Rotation.from_quat(np.asarray(pose[3:7], dtype=float)).as_matrix()
    return np.asarray(vertices, dtype=float) @ rot.T + np.asarray(pose[:3], dtype=float)


def stone_world_aabb(stone, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(stone.local_aabb(), dtype=float)
    corners = np.array(
        [
            [local[x], local[y], local[z]]
            for x in (0, 1)
            for y in (2, 3)
            for z in (4, 5)
        ],
        dtype=float,
    )
    vertices = transform_vertices(corners, pose)
    return vertices.min(axis=0), vertices.max(axis=0)
