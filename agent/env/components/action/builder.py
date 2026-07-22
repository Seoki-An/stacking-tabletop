import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..state import State
from ..inventory import InventoryManager, pick_pose_position_quat
from agent.config_views import support_config
from .planar import PlanarPoseSampler
from .pose_solver import PoseSolver
from .mesh_fit import ply_vertices, refine_pose_by_ply_faces
from .floor_fill import (
    active_floor_context,
    active_floor_initial_height_ceiling,
    active_layer_fill_metrics,
    lower_floor_fill_reject_stacked,
)
from .support import (
    planar_support_ok,
    support_constraint_ok,
    posegen_contact_support_ok,
    _support_sources_ok,
)
from utils.etc import resolve_thread_count
from utils._phase_timer import phase, timed


POSEGEN_GAP_THRESHOLD = 0.001


@dataclass
class Action:
    stone_idx: int
    pose: np.ndarray
    init_pose: np.ndarray
    solved_pose: Optional[np.ndarray] = None
    stone_id: int = -1
    c_feq: float = 0.0
    c_gap: float = 0.0
    place_robustness_displacement: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.solved_pose is None:
            self.solved_pose = np.asarray(self.pose, dtype=float).copy()

    @classmethod
    def from_dict(cls, action_dict: dict) -> "Action":
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in action_dict.items() if key in names})

    @classmethod
    def to_dict(cls, action: "Action") -> dict:
        return {
            "stone_idx": action.stone_idx,
            "stone_id": action.stone_id,
            "pose": action.pose,
            "solved_pose": np.asarray(action.solved_pose, dtype=float),
            "init_pose": action.init_pose,
            "c_feq": action.c_feq,
            "c_gap": action.c_gap,
            "place_robustness_displacement": action.place_robustness_displacement,
        }

    def copy(self) -> "Action":
        solved_pose = getattr(self, "solved_pose", None)
        return Action(
            stone_idx=self.stone_idx,
            pose=self.pose.copy(),
            init_pose=self.init_pose.copy(),
            solved_pose=(
                np.asarray(solved_pose, dtype=float).copy()
                if solved_pose is not None
                else self.pose.copy()
            ),
            stone_id=self.stone_id,
            c_feq=self.c_feq,
            c_gap=self.c_gap,
            place_robustness_displacement=self.place_robustness_displacement,
            diagnostics=dict(self.diagnostics),
        )


class ActionList:
    def __init__(self, actions: List[Action]):
        self.actions: List[Action] = actions

    def mask(self, action_mask: np.ndarray) -> List[Action]:
        return [a for a, m in zip(self.actions, action_mask) if m > -np.inf]

    def get_dict_array(self) -> Dict[str, np.ndarray]:
        if not self.actions:
            return {
                "stone_idx": np.array([], dtype=int),
                "stone_id": np.array([], dtype=int),
                "pose": np.zeros((0, 7)),
                "solved_pose": np.zeros((0, 7)),
                "init_pose": np.zeros((0, 7)),
                "c_feq": np.array([]),
                "c_gap": np.array([]),
                "place_robustness_displacement": np.array([]),
            }
        return {
            "stone_idx": np.array([a.stone_idx for a in self.actions]),
            "stone_id": np.array([a.stone_id for a in self.actions]),
            "pose": np.vstack([a.pose for a in self.actions]),
            "solved_pose": np.vstack([a.solved_pose for a in self.actions]),
            "init_pose": np.array([a.init_pose for a in self.actions]),
            "c_feq": np.array([a.c_feq for a in self.actions]),
            "c_gap": np.array([a.c_gap for a in self.actions]),
            "place_robustness_displacement": np.array(
                [a.place_robustness_displacement for a in self.actions]
            ),
        }

    def __iter__(self):
        return iter(self.actions)

    def __len__(self):
        return len(self.actions)


@timed("initialize_height")
def initialize_height(
    inventory: InventoryManager,
    state: State,
    pose: np.ndarray,
    idx: int,
    support_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Lift `pose` above any placed stone whose XY footprint can overlap it."""
    assert pose.shape in {(6,), (7,)}
    target = inventory.stones[idx]
    action_cfg = inventory.cfg.get("action", {})
    clearance = float(action_cfg.get("initial_height_clearance", 0.05))
    xy_padding = float(action_cfg.get("initial_height_xy_padding", 0.05))
    mesh_source = str(action_cfg.get("initial_height_mesh_source", "hull"))
    ground_z = support_config(inventory).ground_z

    target_min, target_max = _oriented_aabb(target, pose, mesh_source)
    target_xy_aabb = _translated_xy_aabb(
        target_min[:2],
        target_max[:2],
        np.asarray(pose[:2], dtype=float),
        xy_padding,
    )
    base_height = ground_z - float(target_min[2]) + clearance

    def clearance_over(indices) -> float:
        height = base_height
        for stone_idx in indices:
            st = inventory.stones[stone_idx]
            st_pose = np.asarray(state.stone_poses.get(st.id, st.pose), dtype=float)
            placed_vertices = _world_vertices(st, st_pose, mesh_source)
            placed_min, placed_max = _xyz_aabb(placed_vertices)
            placed_xy_aabb = (
                placed_min[:2] - xy_padding,
                placed_max[:2] + xy_padding,
            )
            if not _aabb_overlap(target_xy_aabb, placed_xy_aabb):
                continue
            height = max(
                height,
                float(placed_max[2]) - float(target_min[2]) + clearance,
            )
        return height

    # The full clearance over every placed stone is a hard non-penetration floor:
    # the initial pose must never start inside a placed stone. `support_indices`
    # only expresses the intended resting supports and can raise the height, but it
    # can never lower it below the floor that clears all overlapping stones.
    height = clearance_over(list(state.stone_seq))
    if support_indices is not None and len(support_indices) > 0:
        height = max(height, clearance_over(list(support_indices)))
    return np.array(pose[:2].tolist() + [height] + pose[-4:].tolist())


def infer_intended_support_indices(
    inventory: InventoryManager,
    state: State,
    pose: np.ndarray,
    idx: int,
) -> List[int]:
    """Infer support bodies that the planar pose is trying to use.

    This keeps support-pair/top-surface intent tied to the pose without changing
    the planar sampler API yet.
    """
    if len(state.stone_seq) < 2:
        return []

    floor_fill_cfg = inventory.cfg.action.planar.get("floor_fill", {})
    support = support_config(inventory)
    xy_factor = support.pre_pose_xy_factor()
    z_tolerance = support.pair_z_tolerance
    max_supports = int(floor_fill_cfg.get("intended_support_max_sources", 3))

    target = inventory.stones[idx]
    target_extent = _stone_xy_extent(target)
    xy = np.asarray(pose[:2], dtype=float)
    candidates = []
    for placed_idx in state.stone_seq:
        placed = inventory.stones[placed_idx]
        placed_pose = np.asarray(
            state.stone_poses.get(placed.id, placed.pose), dtype=float
        )
        xy_limit = xy_factor * 0.5 * (target_extent + _stone_xy_extent(placed))
        delta = xy - placed_pose[:2]
        if not _inside_xy_limit(delta, xy_limit):
            continue
        placed_vertices = _world_vertices(
            placed,
            placed_pose,
            str(inventory.cfg.action.get("initial_height_mesh_source", "hull")),
        )
        _, top = _z_bounds_from_vertices(placed_vertices)
        distance = float(np.linalg.norm(delta / np.maximum(xy_limit, 1e-6)))
        candidates.append((float(top), distance, int(placed_idx)))

    if len(candidates) < 2:
        return []

    # Prefer a same-level support set. If several levels exist nearby, use the
    # lower level first so upper-floor candidates do not accidentally lock onto
    # one high stone and skip unfinished lower fill.
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_group: List[tuple[float, float, int]] = []
    for i, item in enumerate(candidates):
        top = item[0]
        group = [c for c in candidates[i:] if abs(c[0] - top) <= z_tolerance]
        if len(group) >= 2:
            best_group = group
            break
    if not best_group:
        return []

    best_group.sort(key=lambda item: item[1])
    return [idx for _, _, idx in best_group[: max(max_supports, 2)]]


def _stone_xy_extent(stone) -> np.ndarray:
    return np.asarray(stone.local_aabb_extent()[:2], dtype=float)


def _inside_xy_limit(delta_xy: np.ndarray, limit_xy: np.ndarray) -> bool:
    limit = np.maximum(np.asarray(limit_xy, dtype=float), 1e-6)
    return bool(np.all(np.abs(np.asarray(delta_xy, dtype=float)[:2]) <= limit))


def _z_bounds_from_vertices(vertices: np.ndarray) -> tuple[float, float]:
    z = np.asarray(vertices[:, 2], dtype=float)
    return float(np.min(z)), float(np.max(z))


def _oriented_aabb(
    stone,
    pose: np.ndarray,
    mesh_source: str = "hull",
) -> tuple[np.ndarray, np.ndarray]:
    return _xyz_aabb(_oriented_vertices(stone, pose, mesh_source))


def _xyz_aabb(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=float)
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


def _translated_xy_aabb(
    local_min_xy: np.ndarray,
    local_max_xy: np.ndarray,
    xy: np.ndarray,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=float)
    return xy + local_min_xy - padding, xy + local_max_xy + padding


def _oriented_vertices(
    stone,
    pose: np.ndarray,
    mesh_source: str = "hull",
) -> np.ndarray:
    vertices = None
    if mesh_source == "ply":
        vertices = ply_vertices(stone)
    if vertices is None:
        vertices, _ = stone.get_global_hull_mesh_array()
    rot = Rotation.from_quat(_pose_quat(pose)).as_matrix()
    return vertices @ rot.T


def _world_vertices(
    stone,
    pose: np.ndarray,
    mesh_source: str = "hull",
) -> np.ndarray:
    vertices = _oriented_vertices(stone, pose, mesh_source)
    return vertices + _pose_xyz(pose)


def _pose_quat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape == (6,):
        return pose[2:]
    if pose.shape == (7,):
        return pose[3:]
    raise ValueError(f"expected pose shape (6,) or (7,), got {pose.shape}")


def _pose_xyz(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape == (6,):
        return np.array([pose[0], pose[1], 0.0], dtype=float)
    if pose.shape == (7,):
        return pose[:3]
    raise ValueError(f"expected pose shape (6,) or (7,), got {pose.shape}")


def _aabb_overlap(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
) -> bool:
    a_min, a_max = a
    b_min, b_max = b
    return bool(np.all(a_min <= b_max) and np.all(b_min <= a_max))


class ActionBuilder:
    """Builds stacking actions for the environment by combining sampling and pose solving."""

    def __init__(self, cfg, n_threads: Optional[int] = None):
        self.cfg = cfg
        self.n_threads = resolve_thread_count(n_threads, cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.solver = PoseSolver(cfg)
        self._thread_local = threading.local()
        self._executor: Optional[ThreadPoolExecutor] = None
        self.planar_sampler = PlanarPoseSampler(cfg, self.device)
        self._rejection_lock = threading.Lock()
        self._rejection_call_depth = 0
        self._collect_rejection_counts = False
        self.last_rejection_counts: Dict[str, int] = {}
        # Opt-in diagnostic: every candidate's posegen c_gap at solve time,
        # BEFORE any mask/threshold filtering, so the rejected tail is visible.
        self._collect_gap_samples = False
        self._gap_samples: List[float] = []
        self.last_planar_candidates: List[dict] = []
        self.recent_planar_candidate_calls: List[dict] = []
        self._planar_candidate_call_index = 0
        self._planar_candidate_history_limit = 8

    def _banned_stone_ids(self) -> set:
        action_cfg = self.cfg.get("action", {})
        return {int(stone_id) for stone_id in action_cfg.get("banned_stone_ids", [])}

    def _max_pose_per_stone(self) -> int:
        return int(
            self.cfg.action.get(
                "max_pose_per_stone", self.cfg.action.get("n_pose_per_stone", 1)
            )
        )

    def _num_action_samples(self, inventory: InventoryManager) -> int:
        default = len(inventory.stone_set) * self._max_pose_per_stone()
        return int(self.cfg.action.get("n_action_samples", default))

    def _available_stone_indices(
        self,
        inventory: InventoryManager,
        state: State,
    ) -> List[int]:
        banned_stone_ids = self._banned_stone_ids()
        return [
            idx
            for idx in range(len(inventory.stone_set))
            if idx not in state.stone_seq
            and int(inventory.stone_set[idx]) not in banned_stone_ids
        ]

    def get_action(
        self,
        inventory: InventoryManager,
        state: State,
        init_pose: np.ndarray,
        action_idx: int,
    ) -> Action:
        self.solver.reset_scene(inventory, state)

        stone = inventory.stones[action_idx]
        height_map_score = self.planar_sampler.score_at_xy(action_idx, init_pose[:2])
        initialized_pose = initialize_height(inventory, state, init_pose, action_idx)
        initialized_pose = refine_pose_by_ply_faces(
            inventory, state, action_idx, initialized_pose
        )
        initialized_pose = initialize_height(
            inventory, state, initialized_pose, action_idx
        )
        stone.pose = initialized_pose

        opt_pose, c_feq, c_gap, result = self.solver.solve(inventory, stone)
        diagnostics = self._posegen_solve_values(
            inventory, action_idx, result, self.solver
        )
        self._add_height_map_score(diagnostics, height_map_score)
        return Action(
            stone_idx=action_idx,
            pose=opt_pose,
            init_pose=initialized_pose,
            stone_id=int(inventory.stone_set[action_idx]),
            c_feq=c_feq,
            c_gap=c_gap,
            diagnostics=diagnostics,
        )

    def get_action_samples(
        self,
        inventory: InventoryManager,
        state: State,
        scene_height_map: Optional[np.ndarray] = None,
        n_action_samples: Optional[int] = None,
        rejected_actions: Optional[List[Action]] = None,
        reject_xy_radius: float = 0.0,
        sampling_priority: Optional[str] = None,
    ) -> Tuple[ActionList, np.ndarray]:
        root_debug_call = self._begin_rejection_counts()
        planar_debug_call = self._begin_planar_candidate_debug(
            n_action_samples=n_action_samples,
            sampling_priority=sampling_priority,
        )
        self.solver.reset_scene(inventory, state)
        uses_sample_override = n_action_samples is not None
        n_action_samples = (
            self._num_action_samples(inventory)
            if n_action_samples is None
            else int(n_action_samples)
        )
        planar_debug_call["n_action_samples"] = int(n_action_samples)
        planar_debug_call["uses_sample_override"] = bool(uses_sample_override)
        if n_action_samples <= 0:
            self._finish_planar_candidate_debug(planar_debug_call)
            self._finish_rejection_counts(root_debug_call)
            return ActionList([]), np.array([])
        max_pose_per_stone = self._max_pose_per_stone()
        shared_planar_poses = None
        if self.planar_sampler.score_model_kind != "score":
            with phase("action_planar_sample"):
                shared_planar_poses = self.planar_sampler.sample(
                    inventory,
                    state,
                    scene_height_map=scene_height_map,
                    min_count=max_pose_per_stone,
                    random_start=uses_sample_override,
                )

        actions: List[Action] = []
        masks: List[float] = []
        jobs: List[Tuple[int, np.ndarray]] = []
        available_indices = self._available_stone_indices(inventory, state)
        planar_debug_call["available_stone_ids"] = [
            int(inventory.stone_set[idx]) for idx in available_indices
        ]
        self._record_rejection_count("available_stones", len(available_indices))
        if not available_indices:
            self._record_rejection("no_available_stones")
        priority_mode = sampling_priority == "lower_floor_first"
        allocation_indices = self._available_indices_for_allocation(
            available_indices,
            n_action_samples,
            uses_sample_override=uses_sample_override,
            priority_mode=priority_mode,
            call_index=int(planar_debug_call.get("call_index", 0)),
        )
        if allocation_indices != available_indices:
            planar_debug_call["allocation_stone_ids"] = [
                int(inventory.stone_set[idx]) for idx in allocation_indices
            ]
        pose_count_by_stone = self._allocate_pose_counts(
            allocation_indices,
            n_action_samples,
            max_pose_per_stone,
        )
        planar_debug_call["pose_count_by_stone"] = {
            int(inventory.stone_set[idx]): int(count)
            for idx, count in pose_count_by_stone
        }
        for idx, pose_count in pose_count_by_stone:
            if shared_planar_poses is None:
                with phase("action_planar_sample"):
                    planar_poses = self.planar_sampler.sample(
                        inventory,
                        state,
                        scene_height_map=scene_height_map,
                        min_count=max(max_pose_per_stone, pose_count),
                        stone_idx=idx,
                        random_start=uses_sample_override,
                    )
            else:
                planar_poses = shared_planar_poses
            sampled_planar_poses = list(planar_poses)
            if len(planar_poses) == 0:
                self._record_rejection("empty_planar_sample")
            with phase("action_rejected_xy_filter"):
                before_rejected_xy = len(planar_poses)
                planar_poses = self._remove_rejected_planar_poses(
                    inventory,
                    idx,
                    planar_poses,
                    rejected_actions or [],
                    reject_xy_radius,
                )
            if len(planar_poses) < before_rejected_xy:
                self._record_rejection_count(
                    "rejected_xy",
                    before_rejected_xy - len(planar_poses),
                )
            after_rejected_xy_poses = list(planar_poses)
            with phase("action_pre_pose_filter"):
                planar_poses = self._filter_planar_poses_before_posegen(
                    inventory,
                    state,
                    idx,
                    planar_poses,
                )
            after_pre_pose_poses = list(planar_poses)
            if len(planar_poses) == 0:
                self._record_rejection("empty_after_pre_pose_filters")
                self._record_planar_candidate_debug(
                    planar_debug_call,
                    inventory,
                    idx,
                    sampled_planar_poses,
                    after_rejected_xy_poses,
                    after_pre_pose_poses,
                    [],
                )
                continue
            chosen_count = min(pose_count, len(planar_poses))
            self._record_rejection_count("jobs", chosen_count)
            chosen = (
                list(planar_poses[:chosen_count])
                if priority_mode
                else random.sample(planar_poses, chosen_count)
            )
            self._record_planar_candidate_debug(
                planar_debug_call,
                inventory,
                idx,
                sampled_planar_poses,
                after_rejected_xy_poses,
                after_pre_pose_poses,
                chosen,
            )
            for planar_pose in chosen:
                jobs.append((idx, planar_pose))

        if self.n_threads > 1 and len(jobs) > 1:
            with phase("action_build_samples"):
                results = list(
                    self._parallel_executor().map(
                        lambda job: self._build_sample_parallel(
                            inventory, state, job[0], job[1]
                        ),
                        jobs,
                    )
                )
            actions = [action for action, _ in results]
            masks = [mask for _, mask in results]
        else:
            self.solver.reset_scene(inventory, state)
            with phase("action_build_samples"):
                for idx, planar_pose in jobs:
                    action, mask = self._build_sample(
                        self.solver,
                        inventory,
                        state,
                        idx,
                        planar_pose,
                    )
                    actions.append(action)
                    masks.append(mask)

        while len(actions) < n_action_samples:
            actions.append(self._dummy_action())
            masks.append(-np.inf)
            self._record_rejection("dummy_padding")

        self._finish_planar_candidate_debug(planar_debug_call)
        self._finish_rejection_counts(root_debug_call)

        return ActionList(actions), np.array(masks)

    @staticmethod
    def _available_indices_for_allocation(
        available_indices: List[int],
        n_action_samples: int,
        uses_sample_override: bool,
        priority_mode: bool,
        call_index: int,
    ) -> List[int]:
        ordered = list(available_indices)
        if not uses_sample_override or len(ordered) <= 1:
            return ordered

        if priority_mode:
            if n_action_samples >= len(ordered):
                return ordered
            offset = max(int(call_index) - 1, 0) % len(ordered)
            return ordered[offset:] + ordered[:offset]

        random.shuffle(ordered)
        return ordered

    def _begin_planar_candidate_debug(
        self,
        n_action_samples: Optional[int],
        sampling_priority: Optional[str],
    ) -> dict:
        self._planar_candidate_call_index += 1
        return {
            "call_index": int(self._planar_candidate_call_index),
            "sampling_priority": sampling_priority,
            "n_action_samples": None
            if n_action_samples is None
            else int(n_action_samples),
            "uses_sample_override": n_action_samples is not None,
            "score_model_kind": str(self.planar_sampler.score_model_kind),
            "stones": [],
        }

    def _finish_planar_candidate_debug(self, call_debug: dict) -> None:
        self.last_planar_candidates = list(call_debug.get("stones", []))
        self.recent_planar_candidate_calls.append(call_debug)
        limit = max(1, int(self._planar_candidate_history_limit))
        if len(self.recent_planar_candidate_calls) > limit:
            self.recent_planar_candidate_calls = self.recent_planar_candidate_calls[
                -limit:
            ]

    def _record_planar_candidate_debug(
        self,
        call_debug: dict,
        inventory: InventoryManager,
        stone_idx: int,
        sampled_poses: List[np.ndarray],
        after_rejected_xy_poses: List[np.ndarray],
        after_pre_pose_poses: List[np.ndarray],
        chosen_poses: List[np.ndarray],
    ) -> None:
        record = {
            "stone_idx": int(stone_idx),
            "stone_id": int(inventory.stone_set[stone_idx]),
            "sampled_xy": self._planar_xy_array(sampled_poses),
            "after_rejected_xy": self._planar_xy_array(after_rejected_xy_poses),
            "after_pre_pose_filter_xy": self._planar_xy_array(after_pre_pose_poses),
            "chosen_xy": self._planar_xy_array(chosen_poses),
            "counts": {
                "sampled": int(len(sampled_poses)),
                "after_rejected_xy": int(len(after_rejected_xy_poses)),
                "after_pre_pose_filter": int(len(after_pre_pose_poses)),
                "chosen": int(len(chosen_poses)),
            },
        }
        score_map = self.planar_sampler.score_map_for_stone(stone_idx)
        if score_map is not None:
            record["score_map"] = self._copy_score_map_debug(score_map)
            call_debug["stones"].append(record)

    @staticmethod
    def _planar_xy_array(poses: List[np.ndarray]) -> np.ndarray:
        if not poses:
            return np.empty((0, 2), dtype=float)
        return np.asarray(
            [np.asarray(pose, dtype=float).reshape(-1)[:2] for pose in poses],
            dtype=float,
        )

    @staticmethod
    def _copy_score_map_debug(score_map: dict) -> dict:
        copied = {}
        for key, value in score_map.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, dict):
                copied[key] = dict(value)
            else:
                copied[key] = value
        return copied

    def _allocate_pose_counts(
        self,
        available_indices: List[int],
        n_action_samples: int,
        max_pose_per_stone: int,
    ) -> List[Tuple[int, int]]:
        counts = {idx: 0 for idx in available_indices}
        remaining = n_action_samples
        for _ in range(max_pose_per_stone):
            for idx in available_indices:
                if remaining <= 0:
                    return [(idx, count) for idx, count in counts.items() if count > 0]
                counts[idx] += 1
                remaining -= 1
        return [(idx, count) for idx, count in counts.items() if count > 0]

    def _filter_planar_poses_before_posegen(
        self,
        inventory: InventoryManager,
        state: State,
        stone_idx: int,
        planar_poses: List[np.ndarray],
    ) -> List[np.ndarray]:
        if self.planar_sampler.score_model_kind == "score":
            return list(planar_poses)

        initial_count = len(planar_poses)
        support = support_config(inventory)
        if not support.pre_pose_filter:
            poses = list(planar_poses)
        else:
            with phase("action_planar_support_filter"):
                poses = [
                    pose
                    for pose in planar_poses
                    if planar_support_ok(inventory, state, stone_idx, pose)
                ]
            if len(poses) < initial_count:
                self._record_rejection_count(
                    "pre_pose_support",
                    initial_count - len(poses),
                )
            if not poses and support.pre_pose_fallback_to_unfiltered:
                self._record_rejection("pre_pose_support_fallback")
                poses = list(planar_poses)

        before_large_filter = len(poses)
        with phase("action_large_stone_height_filter"):
            poses = self._filter_large_stone_height(inventory, state, stone_idx, poses)
        if len(poses) < before_large_filter:
            self._record_rejection_count(
                "large_stone_height",
                before_large_filter - len(poses),
            )
        before_floor_filter = len(poses)
        with phase("action_lower_floor_height_filter"):
            poses = self._filter_lower_floor_initial_height(
                inventory, state, stone_idx, poses
            )
        if len(poses) < before_floor_filter:
            self._record_rejection_count(
                "lower_floor_height",
                before_floor_filter - len(poses),
            )
        return poses

    def _filter_lower_floor_initial_height(
        self,
        inventory: InventoryManager,
        state: State,
        stone_idx: int,
        planar_poses: List[np.ndarray],
    ) -> List[np.ndarray]:
        cfg = self.cfg.action.planar.get("floor_fill", {})
        if not bool(cfg.get("lower_floor_initial_height_filter", True)):
            return list(planar_poses)
        if len(state.stone_seq) == 0:
            return list(planar_poses)

        ceiling = active_floor_initial_height_ceiling(inventory, state)
        if ceiling is None:
            return list(planar_poses)

        filtered = [
            pose
            for pose in planar_poses
            if initialize_height(inventory, state, pose, stone_idx)[2] <= ceiling
        ]
        if filtered or not bool(cfg.get("lower_floor_fallback_to_unfiltered", True)):
            return filtered
        return list(planar_poses)

    def _filter_large_stone_height(
        self,
        inventory: InventoryManager,
        state: State,
        stone_idx: int,
        planar_poses: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Filter poses that would place a large/heavy stone too high in the wall."""
        cfg = self.cfg.action.get("large_stone_height_limit", {})
        if not bool(cfg.get("enabled", False)):
            return list(planar_poses)

        volume_percentile = float(cfg.get("volume_percentile", 0.6))
        max_height_fraction = float(cfg.get("max_height_fraction", 0.5))

        volumes = np.array(
            [inventory.stones[i].volume for i in range(len(inventory.stone_set))],
            dtype=float,
        )
        threshold = float(np.percentile(volumes, volume_percentile * 100.0))
        if inventory.stones[stone_idx].volume < threshold:
            return list(planar_poses)

        max_z = max_height_fraction * float(inventory.target_wall.height)
        filtered = [
            pose
            for pose in planar_poses
            if initialize_height(inventory, state, pose, stone_idx)[2] <= max_z
        ]
        return filtered if filtered else list(planar_poses)

    @staticmethod
    def _remove_rejected_planar_poses(
        inventory: InventoryManager,
        stone_idx: int,
        planar_poses: List[np.ndarray],
        rejected_actions: List[Action],
        xy_radius: float,
    ) -> List[np.ndarray]:
        if not rejected_actions or xy_radius <= 0.0:
            return list(planar_poses)

        stone_id = int(inventory.stone_set[stone_idx])
        rejected_xy = [
            rejected.init_pose[:2]
            for rejected in rejected_actions
            if int(rejected.stone_id) == stone_id
        ]
        if not rejected_xy:
            return list(planar_poses)

        kept = []
        for pose in planar_poses:
            if any(np.linalg.norm(pose[:2] - xy) <= xy_radius for xy in rejected_xy):
                continue
            kept.append(pose)
        return kept

    @staticmethod
    def _dummy_action() -> Action:
        identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        return Action(
            stone_idx=-1,
            stone_id=-1,
            pose=identity.copy(),
            init_pose=identity.copy(),
        )

    def _build_sample(
        self,
        solver: PoseSolver,
        inventory: InventoryManager,
        state: State,
        idx: int,
        planar_pose: np.ndarray,
    ) -> Tuple[Action, float]:
        return self._build_sample_with_solver(
            solver, inventory, state, idx, planar_pose
        )

    def _build_sample_parallel(
        self,
        inventory: InventoryManager,
        state: State,
        idx: int,
        planar_pose: np.ndarray,
    ) -> Tuple[Action, float]:
        solver = self._thread_solver()
        solver.reset_scene(inventory, state)
        return self._build_sample_with_solver(
            solver, inventory, state, idx, planar_pose
        )

    def _thread_solver(self) -> PoseSolver:
        solver = getattr(self._thread_local, "solver", None)
        if solver is None:
            solver = PoseSolver(self.cfg)
            self._thread_local.solver = solver
        return solver

    def _parallel_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.n_threads)
        return self._executor

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def is_action_supported(
        self,
        inventory: InventoryManager,
        state: State,
        action: Action,
    ) -> bool:
        if action.stone_idx < 0:
            return False
        if self._posegen_gap_failed(getattr(action, "c_gap", 0.0)):
            return False
        return self._support_constraint_ok(
            inventory,
            state,
            action.stone_idx,
            action.pose,
        )

    def _build_sample_with_solver(
        self,
        solver: PoseSolver,
        inventory: InventoryManager,
        state: State,
        idx: int,
        planar_pose: np.ndarray,
    ) -> Tuple[Action, float]:
        height_map_score = self.planar_sampler.score_at_xy(idx, planar_pose[:2])
        scan_quat: Optional[np.ndarray] = None
        if self.cfg.action.pose_from_scan:
            _, scan_quat = pick_pose_position_quat(
                inventory.pick_poses[inventory.stone_set[idx]]
            )
            planar_pose = planar_pose.copy()
            planar_pose[2:] = (
                Rotation.from_quat(planar_pose[2:]) * Rotation.from_quat(scan_quat)
            ).as_quat()

        intended_supports = infer_intended_support_indices(
            inventory,
            state,
            planar_pose,
            idx,
        )
        init_supports = intended_supports if len(intended_supports) >= 2 else None
        init_pose = initialize_height(
            inventory,
            state,
            planar_pose,
            idx,
            support_indices=init_supports,
        )
        init_pose = refine_pose_by_ply_faces(inventory, state, idx, init_pose)
        init_pose = initialize_height(
            inventory,
            state,
            init_pose,
            idx,
            support_indices=init_supports,
        )

        stone = inventory.stones[idx].copy()
        stone.pose = init_pose
        pose, c_feq, c_gap, result = solver.solve(inventory, stone)

        mask = 0.0
        mask_reasons: List[str] = []

        def reject(reason: str) -> None:
            nonlocal mask
            mask = -np.inf
            if reason not in mask_reasons:
                mask_reasons.append(reason)

        diagnostics = self._posegen_gap_values(c_gap)
        diagnostics.update(
            self._posegen_solve_values(inventory, idx, result, solver)
        )
        self._add_height_map_score(diagnostics, height_map_score)
        self._record_gap_sample(c_gap)
        self._record_rejection("built")
        if not diagnostics["posegen_gap_ok"]:
            reject("posegen_gap")
            self._record_rejection("posegen_gap")
        active_layer = active_floor_context(inventory, state)
        lower_fill_candidate = False
        if active_layer is not None:
            fill_metrics = active_layer_fill_metrics(
                inventory,
                active_layer,
                idx,
                pose,
            )
            fill_score = float(fill_metrics["fill_score"])
            above_active_layer = bool(fill_metrics["above_active_layer"])
            lower_fill_candidate = fill_score > 0.0 and not above_active_layer
            diagnostics["active_layer_fill_score"] = float(fill_score)
            diagnostics["active_layer_above"] = bool(above_active_layer)
            diagnostics["active_layer_contact_cells"] = int(
                fill_metrics["contact_cells"]
            )
            diagnostics["active_layer_unfilled_cells"] = int(
                fill_metrics["unfilled_cells"]
            )
            diagnostics["active_layer_overlap_cells"] = int(
                fill_metrics["overlap_cells"]
            )
            floor_fill_cfg = self.cfg.action.planar.get("floor_fill", {})
            if lower_floor_fill_reject_stacked(floor_fill_cfg):
                if above_active_layer:
                    reject("active_layer_above")
                    self._record_rejection("active_layer_above")
                else:
                    min_contact = max(
                        int(
                            floor_fill_cfg.get(
                                "u_shape_min_frontier_contact_cells",
                                1,
                            )
                        ),
                        0,
                    )
                    diagnostics["active_layer_min_contact_cells"] = int(min_contact)
                    if (
                        min_contact > 0
                        and fill_score > 0.0
                        and int(fill_metrics["contact_cells"]) < min_contact
                    ):
                        reject("active_layer_disconnected")
                        self._record_rejection("active_layer_disconnected")
        diagnostics["lower_fill_candidate"] = bool(lower_fill_candidate)
        if scan_quat is not None:
            scan_values = self._scan_orientation_values(pose, scan_quat)
            diagnostics.update(scan_values)
            if not scan_values["scan_orientation_ok"]:
                if lower_fill_candidate:
                    self._record_rejection("scan_orientation_soft_lower_fill")
                else:
                    reject("scan_orientation")
                    self._record_rejection("scan_orientation")
        support_ok = self._support_constraint_ok(
            inventory,
            state,
            idx,
            pose,
            result,
            solver,
        )
        if not support_ok and len(intended_supports) >= 2:
            support_ok = self._intended_support_constraint_ok(
                inventory,
                state,
                idx,
                pose,
                intended_supports,
            )
        support_cfg = self.cfg.action.get("support_constraint", {})
        if not support_ok and bool(support_cfg.get("hard_filter", False)):
            reject("support_constraint")
            self._record_rejection("support_constraint")
        elif not support_ok:
            self._record_rejection("support_constraint_soft")
        orientation_cfg = self.cfg.action.planar.get("floor_fill", {}).get(
            "orientation", {}
        )
        if bool(orientation_cfg.get("hard_filter", False)) and not (
            self.planar_sampler.floor_fill_upper_face_inward_ok(inventory, idx, pose)
        ):
            reject("floor_fill_orientation")
            self._record_rejection("floor_fill_orientation")
        if mask_reasons:
            diagnostics["action_mask_reason"] = str(mask_reasons[0])
            diagnostics["action_mask_reasons"] = list(mask_reasons)

        self._record_rejection("accepted" if np.isfinite(mask) else "rejected")

        return (
            Action(
                stone_idx=idx,
                stone_id=int(inventory.stone_set[idx]),
                pose=pose,
                init_pose=init_pose,
                c_feq=c_feq,
                c_gap=c_gap,
                diagnostics=diagnostics,
            ),
            mask,
        )

    @staticmethod
    def _add_height_map_score(
        diagnostics: dict,
        score: tuple[float, float] | None,
    ) -> None:
        if score is None:
            return
        diagnostics["height_map_score"] = float(score[0])
        diagnostics["height_map_score_normalized"] = float(score[1])

    def _posegen_gap_threshold(self) -> float | None:
        value = self.cfg.action.get("posegen_gap_threshold", POSEGEN_GAP_THRESHOLD)
        return None if value is None else float(value)

    def _posegen_gap_values(self, c_gap: float) -> dict:
        threshold = self._posegen_gap_threshold()
        c_gap = float(c_gap)
        ok = np.isfinite(c_gap) and (threshold is None or c_gap <= threshold)
        return {
            "posegen_c_gap": c_gap,
            "posegen_gap_threshold": threshold,
            "posegen_gap_ok": bool(ok),
        }

    def _posegen_gap_failed(self, c_gap: float) -> bool:
        return not self._posegen_gap_values(c_gap)["posegen_gap_ok"]

    def _posegen_solve_values(
        self,
        inventory: InventoryManager,
        target_idx: int,
        result,
        solver: PoseSolver,
    ) -> dict:
        contacts = self._posegen_solve_contacts(
            inventory, target_idx, result, solver
        )
        force_norms = [
            float(item["force_norm"])
            for item in contacts
            if "force_norm" in item and np.isfinite(float(item["force_norm"]))
        ]
        return {
            "pose_solve_contacts": contacts,
            "pose_solve_contact_count": int(len(contacts)),
            "pose_solve_contact_force_max": (
                float(max(force_norms)) if force_norms else 0.0
            ),
        }

    def _posegen_solve_contacts(
        self,
        inventory: InventoryManager,
        target_idx: int,
        result,
        solver: PoseSolver,
    ) -> list[dict]:
        if result is None or not hasattr(result, "contact_point"):
            return []

        contact_points = getattr(result, "contact_point", {}) or {}
        contact_forces = getattr(result, "contact_force", {}) or {}
        contact_normals = getattr(result, "contact_normal", {}) or {}
        body_to_stone = getattr(solver, "body_id_to_stone_idx", {}) or {}

        contacts = []
        for raw_body_id, raw_points in contact_points.items():
            try:
                body_id = int(raw_body_id)
            except (TypeError, ValueError):
                continue

            points = self._posegen_vector_rows(raw_points, width=3)
            forces = self._posegen_vector_rows(
                self._posegen_map_get(contact_forces, raw_body_id, []), width=3
            )
            normals = self._posegen_vector_rows(
                self._posegen_map_get(contact_normals, raw_body_id, []), width=3
            )
            stone_idx = (
                target_idx
                if body_id == -1
                else self._posegen_map_get(body_to_stone, body_id, None)
            )

            for contact_idx, point in enumerate(points):
                if point.shape[0] < 3 or not np.all(np.isfinite(point[:3])):
                    continue
                item = {
                    "body_id": body_id,
                    "contact_idx": int(contact_idx),
                    "role": "target" if body_id == -1 else "scene",
                    "point": point[:3].astype(float).tolist(),
                }
                if stone_idx is not None:
                    item["stone_idx"] = int(stone_idx)
                    if 0 <= int(stone_idx) < len(inventory.stone_set):
                        item["stone_id"] = int(inventory.stone_set[int(stone_idx)])

                force = self._posegen_row_at(forces, contact_idx)
                if force is not None:
                    item["force"] = force[:3].astype(float).tolist()
                    item["force_norm"] = float(np.linalg.norm(force[:3]))

                normal = self._posegen_row_at(normals, contact_idx)
                if normal is not None:
                    item["normal"] = normal[:3].astype(float).tolist()

                contacts.append(item)
        return contacts

    @staticmethod
    def _posegen_vector_rows(values, width: int) -> list[np.ndarray]:
        if values is None:
            return []
        try:
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 1:
                return (
                    [arr[:width].astype(float).copy()]
                    if arr.shape[0] >= width and np.all(np.isfinite(arr[:width]))
                    else []
                )
            if arr.ndim == 2:
                return [
                    row[:width].astype(float).copy()
                    for row in arr
                    if row.shape[0] >= width and np.all(np.isfinite(row[:width]))
                ]
        except (TypeError, ValueError):
            pass

        try:
            iterator = iter(values)
        except TypeError:
            return []

        rows = []
        for value in iterator:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.shape[0] >= width and np.all(np.isfinite(arr[:width])):
                rows.append(arr[:width].copy())
        return rows

    @staticmethod
    def _posegen_row_at(rows: list[np.ndarray], index: int) -> np.ndarray | None:
        if index >= len(rows):
            return None
        row = np.asarray(rows[index], dtype=float).reshape(-1)
        if row.shape[0] < 3 or not np.all(np.isfinite(row[:3])):
            return None
        return row[:3].copy()

    @staticmethod
    def _posegen_map_get(mapping, key, default):
        if mapping is None:
            return default
        try:
            return mapping.get(key, default)
        except AttributeError:
            pass
        try:
            return mapping[key]
        except (KeyError, TypeError):
            return default

    def _begin_rejection_counts(self) -> bool:
        with self._rejection_lock:
            is_root_call = self._rejection_call_depth == 0
            if is_root_call:
                cfg = self.cfg.action.get("debug_rejection_counts", {})
                self._collect_rejection_counts = bool(cfg.get("enabled", False))
                self.last_rejection_counts = {}
            self._rejection_call_depth += 1
            return is_root_call

    def _finish_rejection_counts(self, is_root_call: bool) -> None:
        with self._rejection_lock:
            self._rejection_call_depth = max(self._rejection_call_depth - 1, 0)
            collect = self._collect_rejection_counts
            if is_root_call:
                self._collect_rejection_counts = False
        if is_root_call and collect:
            self._print_rejection_counts_if_enabled()

    def set_collect_gap_samples(self, enabled: bool) -> None:
        """Enable/disable capturing every candidate's posegen c_gap. Clears
        any accumulated samples. Off by default (no overhead in production)."""
        with self._rejection_lock:
            self._collect_gap_samples = bool(enabled)
            self._gap_samples = []

    def take_gap_samples(self) -> List[float]:
        """Return and clear the c_gap samples collected since the last call."""
        with self._rejection_lock:
            samples = list(self._gap_samples)
            self._gap_samples = []
        return samples

    def _record_gap_sample(self, c_gap: float) -> None:
        if not self._collect_gap_samples:
            return
        with self._rejection_lock:
            self._gap_samples.append(float(c_gap))

    def _record_rejection(self, reason: str) -> None:
        self._record_rejection_count(reason, 1)

    def _record_rejection_count(self, reason: str, count: int) -> None:
        if count <= 0 or not self._collect_rejection_counts:
            return
        with self._rejection_lock:
            self.last_rejection_counts[reason] = self.last_rejection_counts.get(
                reason, 0
            ) + int(count)

    def _print_rejection_counts_if_enabled(self) -> None:
        cfg = self.cfg.action.get("debug_rejection_counts", {})
        should_print = bool(cfg.get("print", cfg.get("enabled", False)))
        if not should_print:
            return
        counts = dict(self.last_rejection_counts)
        if not counts:
            return
        prefix = str(cfg.get("prefix", "[ActionBuilder] rejection counts"))
        ordered = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
        print(f"{prefix}: {ordered}")

    @staticmethod
    def _scan_orientation_values(pose: np.ndarray, scan_quat: np.ndarray) -> dict:
        rot_diff = Rotation.from_quat(pose[3:]) * Rotation.from_quat(scan_quat).inv()
        rotvec_xy_norm = float(np.linalg.norm(rot_diff.as_rotvec()[:2]))
        rotvec_xy_limit = float(np.pi / 4.0)
        matrix_zz = float(rot_diff.as_matrix()[2, 2])
        too_tilted = rotvec_xy_norm > rotvec_xy_limit
        flipped = matrix_zz < 0.0
        return {
            "scan_orientation_rotvec_xy_norm": rotvec_xy_norm,
            "scan_orientation_rotvec_xy_limit": rotvec_xy_limit,
            "scan_orientation_matrix_zz": matrix_zz,
            "scan_orientation_too_tilted": bool(too_tilted),
            "scan_orientation_flipped": bool(flipped),
            "scan_orientation_ok": bool(not (too_tilted or flipped)),
        }

    def _support_constraint_ok(
        self,
        inventory: InventoryManager,
        state: State,
        stone_idx: int,
        pose: np.ndarray,
        result=None,
        solver: Optional[PoseSolver] = None,
    ) -> bool:
        if result is not None:
            return posegen_contact_support_ok(
                inventory,
                state,
                stone_idx,
                pose,
                result,
                body_id_to_stone_idx=getattr(solver, "body_id_to_stone_idx", {}),
            )
        return support_constraint_ok(inventory, state, stone_idx, pose)

    def _intended_support_constraint_ok(
        self,
        inventory: InventoryManager,
        state: State,
        stone_idx: int,
        pose: np.ndarray,
        support_indices: List[int],
    ) -> bool:
        support = support_config(inventory)
        target = inventory.stones[stone_idx]
        target_extent = _stone_xy_extent(target)
        target_xy = np.asarray(pose[:2], dtype=float)
        xy_factor = support.xy_factor

        plausible = []
        for placed_idx in support_indices:
            placed = inventory.stones[int(placed_idx)]
            placed_pose = np.asarray(
                state.stone_poses.get(placed.id, placed.pose),
                dtype=float,
            )
            xy_limit = xy_factor * 0.5 * (target_extent + _stone_xy_extent(placed))
            if _inside_xy_limit(target_xy - placed_pose[:2], xy_limit):
                plausible.append(int(placed_idx))

        if len(plausible) < support.min_sources:
            return False
        return _support_sources_ok(
            inventory,
            state,
            stone_idx,
            pose,
            plausible,
            support,
        )
