import argparse
import csv
import copy
from datetime import datetime
import json
import pickle
import time
import gc
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.env import StoneStackingEnv
from agent.env.components.state import State
from agent.mcts import MCTS_Node, MonteCarloTreeSearch
from agent.mcts.utils import solved_action_pose
from scripts.debug.mcts_map_images import (
    raw_scene_height_map_debug,
    save_step_map_images,
    state_score_map_debug,
)
from utils import SuppressOutput

# Four views for candidate PNGs (2×2 grid, each at half resolution → same total size).
_CANDIDATE_CAMERAS = [
    {"eye": [3.5,  3.5, 3.0], "look_at": [0.0, 0.0, 0.8], "up": [0, 0, 1]},  # front-left
    {"eye": [-3.5, 3.5, 3.0], "look_at": [0.0, 0.0, 0.8], "up": [0, 0, 1]},  # front-right
    {"eye": [-3.5,-3.5, 3.0], "look_at": [0.0, 0.0, 0.8], "up": [0, 0, 1]},  # back-right
    {"eye": [3.5, -3.5, 3.0], "look_at": [0.0, 0.0, 0.8], "up": [0, 0, 1]},  # back-left
]

# Bright amber, opaque — used to highlight the selected (rank-1) candidate.
_SELECTED_COLOR = np.array([1.0, 0.80, 0.05, 1.0])


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="config name in agent/configs (without .yml)",
    )
    p.add_argument(
        "-w",
        "--weights",
        type=str,
        default="",
        help="path to trained heightmap CNN checkpoint (.pt/.pth)",
    )
    p.add_argument(
        "--score-model",
        type=str,
        default=None,
        choices=(
            "cnn",
            "heuristic",
            "score",
        ),
        help="planar scorer to use during MCTS (default: value from config)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--video",
        type=str,
        default="mcts.mp4",
        help="output video path; pass empty string to skip visualization",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument(
        "--mesh",
        action="store_true",
        help="open an interactive Open3D mesh visualization of the final MCTS state",
    )
    p.add_argument(
        "--mesh-source",
        type=str,
        default="dsf",
        choices=("auto", "dsf", "ply"),
        help="mesh source for candidate/final visualization",
    )
    p.add_argument(
        "--serial",
        action="store_true",
        help="force one local MCTS instance (already the default when num_workers=0)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="ray workers for parallel MCTS (default: cfg.resource.num_workers)",
    )
    p.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="posegen workers per environment (default: divide resource.num_cpus across Ray workers)",
    )
    preserve_tree = p.add_mutually_exclusive_group()
    preserve_tree.add_argument(
        "--preserve-tree",
        dest="preserve_tree",
        action="store_true",
        help="reuse and revalidate the selected MCTS subtree between steps (default)",
    )
    preserve_tree.add_argument(
        "--no-preserve-tree",
        dest="preserve_tree",
        action="store_false",
        help="start each debug step from a fresh MCTS root",
    )
    p.set_defaults(preserve_tree=True)
    p.add_argument(
        "--max-child-drift",
        type=float,
        default=0.1,
        help="metres of preserved-child revalidation drift before dropping its subtree",
    )
    p.add_argument(
        "--exploration-constant",
        type=float,
        default=None,
        help="override algorithm.mcts.exploration_constant for diagnostics",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="maximum number of placements for this run (0 = environment.n_stone)",
    )
    p.add_argument(
        "--debug-dir",
        type=str,
        default=".debug/mcts",
        help="root directory for per-run logs and candidate visualizations",
    )
    p.add_argument(
        "--max-candidate-visuals",
        type=int,
        default=64,
        help="maximum number of candidate actions to draw per step",
    )
    p.add_argument(
        "--resume-debug-pkl",
        type=str,
        default="",
        help="resume from the latest scene stored in a previous debug_state.pkl",
    )
    p.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="1-based debug step whose scene should be used as the resume state; default uses the latest scene",
    )
    return p


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_n_threads(
    cfg: OmegaConf, args: argparse.Namespace, num_workers: int
) -> int:
    if args.n_threads is not None:
        return max(int(args.n_threads), 1)
    num_cpus = int(float(cfg.resource.num_cpus))
    if num_workers > 0:
        return max(num_cpus // num_workers, 1)
    return max(num_cpus, 1)


class _RayMCTSWorker:
    def __init__(self, mcts_cfg, env_args):
        self.mcts = MonteCarloTreeSearch(mcts_cfg, StoneStackingEnv, env_args)

    def search(
        self,
        root: MCTS_Node,
        preserve_tree: bool,
        revalidate_root: bool,
        max_child_drift: float,
    ):
        preserve_stats = None
        if preserve_tree and revalidate_root:
            preserve_stats = self.mcts.revalidate_preserved_root(
                root,
                use_qfunction=False,
                max_child_drift=max_child_drift,
            )
        nodes, scores = self.mcts.search(
            root=root,
            use_qfunction=False,
            preserve_tree=preserve_tree,
            epsilon=0.0,
            eval=True,
            multiple_nodes=True,
            log_info=True,
        )
        return (
            nodes,
            scores,
            list(getattr(self.mcts, "_last_final_validation_debug_nodes", [])),
            preserve_stats,
            getattr(self.mcts, "_last_search_width_stats", None),
            _action_sampling_debug(self.mcts.env, self.mcts),
        )

    def close(self) -> None:
        self.mcts.close()


def _search_one_step(
    node: MCTS_Node,
    mcts: MonteCarloTreeSearch,
    ray_workers: Optional[List],
    preserve_tree: bool,
    revalidate_root: bool,
    max_child_drift: float,
) -> Tuple[
    MCTS_Node,
    float,
    List[Tuple[MCTS_Node, float]],
    List[Tuple[MCTS_Node, float]],
    Optional[dict],
    Optional[dict],
    Optional[dict],
]:
    if ray_workers is None:
        preserve_stats = None
        if preserve_tree and revalidate_root:
            preserve_stats = mcts.revalidate_preserved_root(
                node,
                use_qfunction=False,
                max_child_drift=max_child_drift,
            )
        nodes, scores = mcts.search(
            root=node,
            use_qfunction=False,
            preserve_tree=preserve_tree,
            epsilon=0.0,
            eval=True,
            multiple_nodes=True,
            log_info=True,
        )
        candidates = list(zip(nodes, scores))
        debug_nodes = list(
            getattr(mcts, "_last_final_validation_debug_nodes", [])
        )
        rejected = _candidate_pairs(debug_nodes) or _root_candidate_pairs(node)
        search_width = getattr(mcts, "_last_search_width_stats", None)
        action_sampling = _action_sampling_debug(mcts.env, mcts)
        if candidates:
            best_node, best_score = candidates[0]
            return (
                best_node,
                float(best_score),
                candidates,
                rejected,
                preserve_stats,
                search_width,
                action_sampling,
            )
        return node, -np.inf, [], rejected, preserve_stats, search_width, action_sampling

    import ray

    results = ray.get(
        [
            worker.search.remote(
                copy.deepcopy(node),
                preserve_tree,
                revalidate_root,
                max_child_drift,
            )
            for worker in ray_workers
        ]
    )
    successful = []
    rejected = []
    fallback_action_sampling = None
    for nodes, scores, debug_nodes, preserve_stats, search_width, action_sampling in results:
        if fallback_action_sampling is None and action_sampling:
            fallback_action_sampling = action_sampling
        rejected.extend(_candidate_pairs(debug_nodes))
        candidates = list(zip(nodes, scores))
        if candidates:
            successful.append((candidates, preserve_stats, search_width, action_sampling))
    if not successful:
        rejected.sort(key=lambda item: item[1], reverse=True)
        return node, -np.inf, [], rejected, None, None, fallback_action_sampling

    candidates, preserve_stats, search_width, action_sampling = max(
        successful,
        key=lambda item: item[0][0][1],
    )
    best_node, best_score = candidates[0]
    return (
        best_node,
        float(best_score),
        candidates,
        rejected,
        preserve_stats,
        search_width,
        action_sampling,
    )


def _candidate_pairs(nodes: List[MCTS_Node]) -> List[Tuple[MCTS_Node, float]]:
    pairs = []
    seen = set()
    for node in nodes:
        if id(node) in seen or node.action is None or node.action.stone_idx < 0:
            continue
        seen.add(id(node))
        info = node.info or {}
        score = info.get("final_validation_rank_score", None)
        if score is None:
            score = node.q_value_init
            if node.is_simulated and node.visits > 0 and np.isfinite(node.q_value):
                score = node.q_value / max(node.visits, 1.0) + node.q_value_init
        pairs.append((node, float(score)))
    return pairs


def _root_candidate_pairs(root: MCTS_Node) -> List[Tuple[MCTS_Node, float]]:
    pairs = []
    for child in root.children:
        if child.action is None or child.action.stone_idx < 0:
            continue
        score = child.q_value_init
        if child.is_simulated and child.visits > 0 and np.isfinite(child.q_value):
            score = child.q_value / max(child.visits, 1.0) + child.q_value_init
        elif child.failed:
            score = -np.inf
        pairs.append((child, float(score)))
    pairs.sort(
        key=lambda item: (
            bool(item[0].failed),
            not np.isfinite(item[1]),
            -item[1] if np.isfinite(item[1]) else np.inf,
        )
    )
    return pairs


def _debug_candidate_pairs(
    candidates: List[Tuple[MCTS_Node, float]],
    rejected: List[Tuple[MCTS_Node, float]],
) -> List[Tuple[MCTS_Node, float]]:
    rows = []
    seen = set()
    for node, score in list(candidates) + list(rejected):
        if id(node) in seen:
            continue
        seen.add(id(node))
        rows.append((node, float(score)))
    return rows


def _step_reward_from_info(node: MCTS_Node, env_cfg: OmegaConf) -> float:
    if node.info is None:
        return float(node.reward or 0.0)

    weights = env_cfg.reward.weights
    return float(
        weights.stability * node.info.get("stability", 0.0)
        + weights.stone_IoU * node.info.get("stone_IoU_reward", 0.0)
        + weights.get("target_IoU_increment", 0.0)
        * node.info.get("target_IoU_increment", 0.0)
        + weights.get("place_stability", 0.0) * node.info.get("place_stability", 0.0)
        + weights.get("large_stone_lower", 0.0)
        * node.info.get("large_stone_lower", 0.0)
        + weights.get("inward_orientation", 0.0)
        * node.info.get("inward_orientation", 0.0)
        + weights.get("support", 0.0) * node.info.get("support", 0.0)
    )


def _painted_copy(mesh: o3d.geometry.TriangleMesh, color) -> o3d.geometry.TriangleMesh:
    mesh = copy.deepcopy(mesh)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def _make_debug_run_dir(root: str, config_name: str, seed: int) -> Path:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(root) / f"{tag}_{config_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "candidates").mkdir()
    return run_dir


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _candidate_rows(candidates: List[Tuple[MCTS_Node, float]]) -> List[dict]:
    rows = []
    for rank, (candidate, score) in enumerate(candidates, start=1):
        action = candidate.action
        if action is None:
            continue
        info, pose_solve_contacts = _candidate_diagnostics(action)
        info.update(_serializable_info(candidate.info))
        rows.append(
            {
                "rank": rank,
                "score": float(score),
                "stone_idx": int(action.stone_idx),
                "stone_id": int(action.stone_id),
                "pose": np.asarray(action.pose, dtype=float).tolist(),
                "solved_pose": solved_action_pose(action).tolist(),
                "settled_pose": np.asarray(action.pose, dtype=float).tolist(),
                "init_pose": np.asarray(action.init_pose, dtype=float).tolist(),
                "c_feq": float(action.c_feq),
                "c_gap": float(action.c_gap),
                "failed": bool(candidate.failed),
                "simulated": bool(candidate.is_simulated),
                "validated": bool(candidate.is_simulated and candidate.reward is not None),
                "reward": None if candidate.reward is None else float(candidate.reward),
                "info_place_stability": (
                    None
                    if candidate.info is None
                    else candidate.info.get("place_stability")
                ),
                "info_place_robustness_displacement": (
                    None
                    if candidate.info is None
                    else candidate.info.get("place_robustness_displacement")
                ),
                "info_place_robustness_nonfinite": (
                    None
                    if candidate.info is None
                    else candidate.info.get("place_robustness_nonfinite")
                ),
                "info_place_robustness_clipped": (
                    None
                    if candidate.info is None
                    else candidate.info.get("place_robustness_clipped")
                ),
                "info_support_count": (
                    None if candidate.info is None else candidate.info.get("support_count")
                ),
                "info_support_has_ground": (
                    None
                    if candidate.info is None
                    else candidate.info.get("support_has_ground")
                ),
                "visits": int(candidate.visits),
                "q_value": float(candidate.q_value),
                "q_value_init": float(candidate.q_value_init),
                "value_init": float(candidate.value_init),
                "info_target_IoU": (
                    None if candidate.info is None else candidate.info.get("target_IoU")
                ),
                "info_target_IoU_increment": (
                    None
                    if candidate.info is None
                    else candidate.info.get("target_IoU_increment")
                ),
                "info_stone_IoU": (
                    None if candidate.info is None else candidate.info.get("stone_IoU")
                ),
                "info_inward_orientation": (
                    None
                    if candidate.info is None
                    else candidate.info.get("inward_orientation")
                ),
                "velocity_integrals": (
                    {}
                    if candidate.state is None
                    else candidate.state.latest_velocity_integrals()
                ),
                "scene_motion": _candidate_scene_motion(candidate),
                "info": info,
                "contact_points": _candidate_contact_points(candidate),
                "pose_solve_contacts": pose_solve_contacts,
            }
        )
    return rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _score_color(score: float, lo: float, hi: float) -> np.ndarray:
    if not np.isfinite(score):
        return np.array([0.25, 0.25, 0.25, 0.35])
    t = 0.5 if hi <= lo else (score - lo) / (hi - lo)
    t = float(np.clip(t, 0.0, 1.0))
    # blue -> cyan -> yellow -> red
    anchors = np.array(
        [
            [0.20, 0.32, 0.85, 0.42],
            [0.00, 0.78, 0.82, 0.48],
            [0.98, 0.86, 0.20, 0.55],
            [0.90, 0.18, 0.16, 0.65],
        ]
    )
    x = t * (len(anchors) - 1)
    i = min(int(np.floor(x)), len(anchors) - 2)
    return anchors[i] * (i + 1 - x) + anchors[i + 1] * (x - i)


def _ply_mesh_path(stone) -> Path:
    path = Path(stone.model_path)
    return path.with_name(f"{path.stem}_mesh.ply")


def _load_ply_mesh(stone) -> o3d.geometry.TriangleMesh | None:
    path = _ply_mesh_path(stone)
    if not path.exists():
        return None
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        return None
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def _stone_mesh(stone, mesh_source: str = "dsf") -> o3d.geometry.TriangleMesh:
    if mesh_source in {"auto", "ply"}:
        mesh = _load_ply_mesh(stone)
        if mesh is not None:
            return mesh
        if mesh_source == "ply":
            print(f"PLY mesh missing for stone {stone.id}; falling back to DSF mesh.")
    return stone.get_lowpoly_mesh()


def _stone_mesh_array(stone, mesh_source: str = "dsf") -> tuple[np.ndarray, np.ndarray]:
    mesh = _stone_mesh(stone, mesh_source)
    return (
        np.asarray(mesh.vertices, dtype=float).copy(),
        np.asarray(mesh.triangles, dtype=int).copy(),
    )


def _mesh_at_pose(
    stone,
    pose: np.ndarray,
    color,
    mesh_source: str = "dsf",
) -> o3d.geometry.TriangleMesh:
    mesh = _stone_mesh(stone, mesh_source)
    mesh.transform(_pose_matrix(pose))
    return _painted_copy(mesh, color[:3])


def _pose_matrix(pose: np.ndarray) -> np.ndarray:
    from utils import pose_to_transformation_matrix

    return pose_to_transformation_matrix(np.asarray(pose, dtype=float))


def _add_mesh(
    renderer,
    name: str,
    mesh: o3d.geometry.TriangleMesh,
    color,
    transparent: bool = False,
) -> None:
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.base_color = np.asarray(color, dtype=float)
    mat.shader = "defaultLitTransparency" if transparent else "defaultLit"
    mat.has_alpha = transparent
    renderer.scene.add_geometry(name, mesh, mat)


def _render_multiview_grid(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    cameras: list,
) -> np.ndarray:
    """Render from each camera at half resolution and tile into a 2×2 grid.

    With 4 cameras each halved, the final image has the same pixel count as
    a single full-resolution render.
    """
    imgs = []
    for cam in cameras:
        renderer.scene.camera.look_at(cam["look_at"], cam["eye"], cam["up"])
        imgs.append(np.asarray(renderer.render_to_image())[::2, ::2])
    top = np.concatenate(imgs[:2], axis=1)
    bot = np.concatenate(imgs[2:], axis=1)
    return np.concatenate([top, bot], axis=0)


def _save_candidate_visualization(
    path: Path,
    env: StoneStackingEnv,
    state,
    candidates: List[Tuple[MCTS_Node, float]],
    max_candidates: int,
    mesh_source: str = "dsf",
) -> None:
    if not candidates:
        return

    scores = np.asarray([score for _, score in candidates], dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    lo = float(np.min(finite_scores)) if finite_scores.size else 0.0
    hi = float(np.max(finite_scores)) if finite_scores.size else 1.0

    with SuppressOutput():
        renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 900)
    renderer.scene.scene.set_indirect_light_intensity(25000)

    wall = env.inventory.target_wall
    plane_width = float(wall.cfg.width) + 10.0
    plane_length = float(wall.cfg.length) + 10.0
    plane = o3d.geometry.TriangleMesh.create_box(plane_width, plane_length, 0.001)
    plane.translate([-plane_width / 2, -plane_length / 2, -0.001])
    plane.compute_vertex_normals()
    _add_mesh(renderer, "ground", plane, [0.48, 0.58, 0.42, 1.0])

    for i, geometry in enumerate(wall.geometries):
        _add_mesh(
            renderer,
            f"target_{i}",
            geometry.get_mesh(),
            [0.78, 0.78, 0.82, 0.18],
            transparent=True,
        )

    for order, stone_idx in enumerate(state.stone_seq):
        stone = env.inventory.stones[stone_idx]
        pose = state.stone_poses.get(stone.id, stone.pose)
        if pose is None or not np.all(np.isfinite(pose)):
            continue
        mesh = _mesh_at_pose(stone, pose, [0.50, 0.50, 0.50, 1.0], mesh_source)
        _add_mesh(renderer, f"scene_{order}", mesh, [0.50, 0.50, 0.50, 1.0])

    for rank, (candidate, score) in enumerate(candidates[:max_candidates], start=1):
        action = candidate.action
        if action is None or action.stone_idx < 0:
            continue
        if not np.all(np.isfinite(action.pose)):
            continue
        stone = env.inventory.stones[action.stone_idx]
        if rank == 1:
            # Selected candidate: opaque amber so it stands out clearly
            mesh = _mesh_at_pose(stone, action.pose, _SELECTED_COLOR[:3], mesh_source)
            _add_mesh(renderer, f"candidate_{rank}", mesh, _SELECTED_COLOR)
        else:
            color = _score_color(float(score), lo, hi)
            mesh = _mesh_at_pose(stone, action.pose, color, mesh_source)
            _add_mesh(renderer, f"candidate_{rank}", mesh, color, transparent=True)

    grid = _render_multiview_grid(renderer, _CANDIDATE_CAMERAS)
    o3d.io.write_image(str(path), o3d.geometry.Image(grid.astype(np.uint8)))


def _visualize_final_mesh(env: StoneStackingEnv, mesh_source: str = "dsf") -> None:
    inventory = env.inventory
    state = env.get_state()
    target_wall = inventory.target_wall

    width = float(target_wall.cfg.width) + 10.0
    length = float(target_wall.cfg.length) + 10.0
    ground = o3d.geometry.TriangleMesh.create_box(width, length, 0.001)
    ground.translate([-width / 2.0, -length / 2.0, -0.001])
    ground = _painted_copy(ground, [0.45, 0.55, 0.42])

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    geoms = [origin, ground]
    for i, geometry in enumerate(target_wall.geometries):
        mesh = _painted_copy(geometry.get_mesh(), [0.72, 0.72, 0.76])
        geoms.append(mesh)

    colors = [
        [0.34, 0.56, 0.86],
        [0.25, 0.70, 0.38],
        [0.90, 0.55, 0.24],
        [0.64, 0.42, 0.82],
        [0.86, 0.34, 0.40],
        [0.30, 0.68, 0.72],
    ]
    for order, stone_idx in enumerate(state.stone_seq):
        stone = inventory.stones[stone_idx]
        pose = state.stone_poses.get(stone.id)
        if pose is None or not np.all(np.isfinite(pose)):
            print(f"Skipping stone {stone.id} mesh visualization; pose is invalid.")
            continue
        mesh = _stone_mesh(stone, mesh_source)
        mesh.transform(stone.get_pose_matrix())
        mesh = _painted_copy(mesh, colors[order % len(colors)])
        geoms.append(mesh)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="mcts final mesh: stones + target wall",
    )


def _init_debug_data(env: StoneStackingEnv, mesh_source: str = "dsf") -> dict:
    """Extract meshes and wall config once; steps are appended later."""
    stone_meshes: dict = {}
    stone_ply_meshes: dict = {}
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
        ply_mesh = _load_ply_mesh(stone)
        if ply_mesh is not None:
            stone_ply_meshes[int(stone.id)] = (
                np.asarray(ply_mesh.vertices, dtype=float).copy(),
                np.asarray(ply_mesh.triangles, dtype=int).copy(),
            )

    wall_meshes = []
    for geom in env.inventory.target_wall.geometries:
        mesh = geom.get_mesh()
        wall_meshes.append((
            np.asarray(mesh.vertices, dtype=float).copy(),
            np.asarray(mesh.triangles, dtype=int).copy(),
        ))

    wall = env.inventory.target_wall
    return {
        "target_wall_cfg": {
            "width":  float(wall.width),
            "length": float(wall.length),
            "height": float(wall.height),
        },
        "target_wall_meshes": wall_meshes,
        "stone_meshes": stone_meshes,
        "stone_ply_meshes": stone_ply_meshes,
        "mesh_source": mesh_source,
        "steps": [],
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


def _append_debug_step(
    debug_data: dict,
    step: int,
    succeeded: bool,
    state,
    candidates: List[Tuple[MCTS_Node, float]],
    env: StoneStackingEnv,
    resume_state: Optional[State] = None,
    preserve_stats: Optional[dict] = None,
    search_width: Optional[dict] = None,
    action_sampling: Optional[dict] = None,
    map_images: Optional[list[str]] = None,
    score_map: Optional[dict] = None,
) -> None:
    recoverable_state = resume_state if resume_state is not None else state
    stone_id_seq = [int(env.inventory.stones[idx].id) for idx in state.stone_seq]
    stone_poses = {
        int(k): np.asarray(v, dtype=float).copy()
        for k, v in state.stone_poses.items()
    }
    cand_dicts = []
    for rank, (node, score) in enumerate(candidates, start=1):
        if node.action is None or node.action.stone_idx < 0:
            continue
        a = node.action
        trajectory = _candidate_trajectory(node, a)
        final_pose = _candidate_final_pose(node, a)
        info, pose_solve_contacts = _candidate_diagnostics(a)
        info.update(_serializable_info(node.info))
        cand = {
            "rank":        rank,
            "score":       float(score),
            "stone_id":    int(a.stone_id),
            "stone_idx":   int(a.stone_idx),
            "pose":        np.asarray(a.pose,      dtype=float).copy(),
            "solved_pose": solved_action_pose(a).copy(),
            "settled_pose": np.asarray(a.pose, dtype=float).copy(),
            "init_pose":   np.asarray(a.init_pose, dtype=float).copy(),
            "failed":      bool(node.failed),
            "simulated":   bool(node.is_simulated),
            "validated":   bool(node.is_simulated and node.reward is not None),
            "reward":      None if node.reward is None else float(node.reward),
            "q_value":     float(node.q_value),
            "q_value_init":float(node.q_value_init),
            "value_init":  float(node.value_init),
            "visits":      int(node.visits),
            "trajectory":  trajectory,
            "velocity_integrals": (
                {} if node.state is None else node.state.latest_velocity_integrals()
            ),
            "scene_motion": _candidate_scene_motion(node),
            "info":        info,
            "contact_points": _candidate_contact_points(node),
            "pose_solve_contacts": pose_solve_contacts,
            "best_sequence": _candidate_best_sequence(node),
        }
        if final_pose is not None:
            cand["final_pose"] = final_pose
        cand_dicts.append(cand)
    step_data = {
        "step":      step + 1,
        "succeeded": succeeded,
        "scene": {
            "stone_seq":   stone_id_seq,
            "stone_poses": stone_poses,
        },
        "score_map": score_map or state_score_map_debug(env, state),
        "candidates": cand_dicts,
        # Raw simulator state for exact resume. The scene above is kept for the
        # candidate viewer and for compatibility with older debug pkls.
        "raw_state": copy.deepcopy(state),
        "resume_state": copy.deepcopy(recoverable_state),
        "preserve_revalidation": _json_safe(preserve_stats),
        "search_width": _json_safe(search_width),
        "action_sampling": action_sampling or _action_sampling_debug(env),
        "map_images": list(map_images or []),
    }
    debug_data["steps"].append(step_data)
    debug_data["resume_state"] = copy.deepcopy(recoverable_state)
    debug_data["resume_step"] = len(recoverable_state.stone_seq)


def _candidate_final_pose(node: MCTS_Node, action) -> Optional[np.ndarray]:
    """Pose the candidate stone reaches in its latest stored simulation."""
    if node.state is None or action is None:
        return None
    pose = getattr(node.state, "stone_poses", {}).get(int(action.stone_id))
    if pose is None:
        return None
    arr = np.asarray(pose, dtype=float)
    if arr.shape[0] >= 7 and np.all(np.isfinite(arr[:7])):
        return arr[:7].copy()
    return None


def _candidate_trajectory(node: MCTS_Node, action) -> list:
    if node.state is None or not getattr(node.state, "trajectories", None):
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


def _candidate_scene_motion(node: MCTS_Node) -> dict | None:
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


def _serializable_info(info: Optional[dict]) -> dict:
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


def _candidate_contact_points(node: MCTS_Node) -> list[dict]:
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
        involves_candidate = (
            item.get("stone_idx_1") == action.stone_idx
            or item.get("stone_idx_2") == action.stone_idx
        )
        if involves_candidate:
            contacts.append(item)
    return contacts


def _candidate_best_sequence(node: MCTS_Node) -> list[dict]:
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


def _save_debug_data(debug_data: dict, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(debug_data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _load_resume_state(
    env: StoneStackingEnv,
    pkl_path: Path,
    step: Optional[int],
) -> Tuple[State, int]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    steps = list(data.get("steps", []))
    if not steps:
        raise ValueError(f"no steps found in resume debug pkl: {pkl_path}")

    selected = None
    if step is None:
        selected = steps[-1]
    else:
        for item in steps:
            if int(item.get("step", -1)) == int(step):
                selected = item
                break
        if selected is None:
            raise ValueError(f"step {step} not found in resume debug pkl: {pkl_path}")

    raw_state = data.get("resume_state", None) if step is None else None
    if raw_state is None:
        raw_state = selected.get("resume_state", selected.get("raw_state", None))
    if raw_state is not None:
        raw_state = copy.deepcopy(raw_state)
        raw_state.stone_set = env.inventory.stone_set.copy()
        return raw_state, len(raw_state.stone_seq)

    scene = selected.get("scene", {})
    stone_ids = [int(stone_id) for stone_id in scene.get("stone_seq", [])]
    id_to_idx = {int(stone_id): idx for idx, stone_id in enumerate(env.inventory.stone_set)}
    missing = [stone_id for stone_id in stone_ids if stone_id not in id_to_idx]
    if missing:
        raise ValueError(
            f"resume pkl contains stone ids not in current inventory: {missing}"
        )

    stone_seq = [id_to_idx[stone_id] for stone_id in stone_ids]
    stone_poses = {
        int(stone_id): np.asarray(pose, dtype=float).copy()
        for stone_id, pose in scene.get("stone_poses", {}).items()
    }
    state = State(
        stone_set=env.inventory.stone_set.copy(),
        stone_seq=stone_seq,
        stone_poses=stone_poses,
        trajectories=[],
        action_history=[],
        terminated=False,
        failed=False,
    )
    return state, len(stone_seq)


def _write_summary(
    path: Path,
    args: argparse.Namespace,
    n_threads: int,
    video_path: str,
    rewards: List[float],
    discount: float,
    last_info: dict,
    step_logs: List[dict],
) -> None:
    discounted_return = sum(r * (discount**i) for i, r in enumerate(rewards))
    summary = {
        "config": args.config,
        "score_model": args.score_model,
        "seed": args.seed,
        "serial": bool(args.serial),
        "preserve_tree": bool(args.preserve_tree),
        "max_child_drift": float(args.max_child_drift),
        "n_threads": int(n_threads),
        "mesh_source": args.mesh_source,
        "video": video_path,
        "n_steps": len(rewards),
        "discounted_return": float(discounted_return),
        "target_IoU": float(last_info.get("target_IoU", 0.0)),
        "steps": step_logs,
    }
    path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")


def main() -> None:
    args = get_parser().parse_args()

    cfg = OmegaConf.load(f"agent/configs/{args.config}.yml")
    if args.exploration_constant is not None:
        cfg.algorithm.mcts.exploration_constant = float(args.exploration_constant)
    if args.score_model is not None:
        cfg.environment.action.planar.score_model = args.score_model
    args.score_model = str(cfg.environment.action.planar.score_model)
    if args.score_model == "cnn":
        cfg.environment.action.planar.cnn.weights = args.weights

    n_stone = int(cfg.environment.n_stone)
    if int(args.max_steps) > 0:
        n_stone = min(n_stone, int(args.max_steps))
    discount = float(cfg.algorithm.mcts.reward.discount)

    _seed_everything(args.seed)
    debug_dir = _make_debug_run_dir(args.debug_dir, args.config, args.seed)
    OmegaConf.save(cfg, debug_dir / "config.yml")
    print(f"debug output: {debug_dir}")
    video_path = ""
    if args.video:
        raw_video_path = Path(args.video)
        video_path = str(
            raw_video_path if raw_video_path.is_absolute() else debug_dir / raw_video_path
        )

    configured_workers = int(
        cfg.resource.num_workers if args.num_workers is None else args.num_workers
    )
    requested_workers = 0 if args.serial else max(configured_workers, 0)
    use_ray = requested_workers > 0
    args.serial = not use_ray
    n_threads = _resolve_n_threads(cfg, args, requested_workers)
    env_args = {"cfg": cfg.environment, "n_threads": n_threads}
    print(f"Python posegen workers per env: {n_threads}")

    env = StoneStackingEnv(env_args)
    init_state, init_obs = env.reset()
    start_step = 0
    if args.resume_debug_pkl:
        resume_path = Path(args.resume_debug_pkl)
        init_state, start_step = _load_resume_state(
            env,
            resume_path,
            args.resume_step,
        )
        env.update_from_state(init_state)
        init_obs = env.obs_builder.build(env.inventory, init_state)
        print(
            f"resumed from {resume_path} at step {start_step} "
            f"({len(init_state.stone_seq)} stones placed)"
        )

    debug_data = _init_debug_data(env, mesh_source=args.mesh_source)
    debug_pkl  = debug_dir / "debug_state.pkl"

    mcts = MonteCarloTreeSearch(cfg.algorithm.mcts, StoneStackingEnv, env_args)
    mcts.env.inventory.target_wall = env.inventory.target_wall.copy()
    ray_module = None
    ray_workers = None
    if use_ray:
        os.environ.setdefault("RAY_DEDUP_LOGS", "0")
        import ray

        ray_module = ray
        if not ray.is_initialized():
            ray.init()
        num_workers = requested_workers
        num_cpus = float(cfg.resource.num_cpus)
        num_gpus = float(cfg.resource.num_gpus)
        MCTSRay = ray.remote(_RayMCTSWorker)
        ray_workers = [
            MCTSRay.options(
                num_cpus=num_cpus / num_workers,
                num_gpus=num_gpus / num_workers,
            ).remote(cfg.algorithm.mcts, env_args)
            for _ in range(num_workers)
        ]
        print(f"parallel MCTS: {num_workers} workers")
    else:
        print("MCTS mode: serial")

    root = MCTS_Node(cfg=cfg.algorithm.mcts)
    root.update_state(init_state, init_obs, 0.0, False, False)
    node = root

    rewards = []
    step_logs = []
    last_info = {}
    try:
        for step in range(start_step, n_stone):
            elapsed_time = time.time()
            search_state = node.state
            env.update_from_state(search_state)
            raw_height_map = raw_scene_height_map_debug(env, search_state)
            (
                node,
                score,
                candidates,
                rejected_candidates,
                preserve_stats,
                search_width,
                action_sampling,
            ) = _search_one_step(
                node,
                mcts,
                ray_workers,
                preserve_tree=bool(args.preserve_tree),
                revalidate_root=bool(args.preserve_tree and step > start_step),
                max_child_drift=float(args.max_child_drift),
            )
            if preserve_stats:
                reasons = preserve_stats.get("reasons", {}) or {}
                reason_text = ",".join(
                    f"{key}:{value}" for key, value in sorted(reasons.items())
                )
                print(
                    "  preserve: "
                    f"revalidated={preserve_stats.get('revalidated', 0)} "
                    f"kept={preserve_stats.get('kept', 0)} "
                    f"failed={preserve_stats.get('failed', 0)} "
                    f"pre_failed={preserve_stats.get('pre_failed', 0)}"
                    + (f" reasons={reason_text}" if reason_text else "")
                )
            env.update_from_state(search_state)
            root_score_map = state_score_map_debug(env, search_state)
            map_images = save_step_map_images(
                debug_dir,
                step + 1,
                root_score_map,
                raw_height_map,
            )
            if score == -np.inf:
                elapsed_time = time.time() - elapsed_time
                rejected_rows = _candidate_rows(rejected_candidates)
                _write_csv(
                    debug_dir / "candidates" / f"step_{step + 1:02d}_rejected.csv",
                    rejected_rows,
                )
                if int(args.max_candidate_visuals) > 0:
                    _save_candidate_visualization(
                        debug_dir / "candidates" / f"step_{step + 1:02d}_rejected.png",
                        env,
                        search_state,
                        rejected_candidates,
                        max_candidates=int(args.max_candidate_visuals),
                        mesh_source=args.mesh_source,
                    )
                step_logs.append(
                    {
                        "step": step + 1,
                        "failed_to_find_candidate": True,
                        "elapsed_s": float(elapsed_time),
                        "n_rejected_candidates": len(rejected_rows),
                        "preserve_revalidation": _json_safe(preserve_stats),
                        "search_width": _json_safe(search_width),
                        "map_images": map_images,
                        "rejected_csv": str(
                            debug_dir
                            / "candidates"
                            / f"step_{step + 1:02d}_rejected.csv"
                        ),
                        "rejected_png": str(
                            debug_dir
                            / "candidates"
                            / f"step_{step + 1:02d}_rejected.png"
                        ),
                    }
                )
                _write_summary(
                    debug_dir / "summary.json",
                    args, n_threads, video_path,
                    rewards, discount, last_info, step_logs,
                )
                print(
                    f"  step {step + 1:2d}: no feasible candidate found; "
                    f"logged {len(rejected_rows)} rejected candidates"
                )
                _append_debug_step(
                    debug_data,
                    step,
                    False,
                    search_state,
                    rejected_candidates,
                    env,
                    resume_state=search_state,
                    preserve_stats=preserve_stats,
                    search_width=search_width,
                    action_sampling=action_sampling,
                    map_images=map_images,
                    score_map=root_score_map,
                )
                _save_debug_data(debug_data, debug_pkl)
                break
            _append_debug_step(
                debug_data,
                step,
                True,
                search_state,
                _debug_candidate_pairs(candidates, rejected_candidates),
                env,
                resume_state=node.state,
                preserve_stats=preserve_stats,
                search_width=search_width,
                action_sampling=action_sampling,
                map_images=map_images,
                score_map=root_score_map,
            )
            _save_debug_data(debug_data, debug_pkl)
            candidate_rows = _candidate_rows(candidates)
            _write_csv(
                debug_dir / "candidates" / f"step_{step + 1:02d}.csv",
                candidate_rows,
            )
            if int(args.max_candidate_visuals) > 0:
                _save_candidate_visualization(
                    debug_dir / "candidates" / f"step_{step + 1:02d}.png",
                    env,
                    search_state,
                    candidates,
                    max_candidates=int(args.max_candidate_visuals),
                    mesh_source=args.mesh_source,
                )
            step_reward = _step_reward_from_info(node, cfg.environment)
            rewards.append(step_reward)
            step_info = node.info or {}
            if step_info:
                last_info = step_info
            elapsed_time = time.time() - elapsed_time
            print(
                f"  step {step + 1:2d}: reward = {step_reward:2.3f}  "
                f"target_IoU = {float(step_info.get('target_IoU', 0.0)):.3f}  "
                f"time = {elapsed_time:.3f}s"
            )
            step_logs.append(
                {
                    "step": step + 1,
                    "score": float(score),
                    "reward": float(step_reward),
                    "elapsed_s": float(elapsed_time),
                    "n_candidates": len(candidate_rows),
                    "preserve_revalidation": _json_safe(preserve_stats),
                    "search_width": _json_safe(search_width),
                    "map_images": map_images,
                    "selected": candidate_rows[0] if candidate_rows else None,
                    "info": _json_safe(step_info),
                }
            )
            _write_summary(
                debug_dir / "summary.json",
                args, n_threads, video_path,
                rewards, discount, last_info, step_logs,
            )
            if node.done:
                break
            node.parent = None
    finally:
        if ray_module is not None and ray_workers is not None:
            ray_module.get([worker.close.remote() for worker in ray_workers])
            for worker in ray_workers:
                ray_module.kill(worker)
            ray_module.shutdown()

    _write_summary(
        debug_dir / "summary.json",
        args, n_threads, video_path,
        rewards, discount, last_info, step_logs,
    )
    discounted_return = sum(r * (discount**i) for i, r in enumerate(rewards))
    target_iou = float(last_info.get("target_IoU", 0.0))
    print(
        f"\nreturn={discounted_return:8.3f}  "
        f"target_IoU={target_iou:6.3f}  steps={len(rewards)}"
    )
    print(f"saved debug logs to {debug_dir}")

    if node.state is not None and (video_path or args.mesh):
        env.update_from_state(node.state)

    if video_path and node.state is not None:
        env.visualization(video_path, fps=args.fps, time_scale=args.time_scale)
        print(f"saved video to {video_path}")

    if args.mesh and node.state is not None:
        _visualize_final_mesh(env, mesh_source=args.mesh_source)

    mcts.close()
    env.close()


if __name__ == "__main__":
    main()
