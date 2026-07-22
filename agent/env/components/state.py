from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np
import torch

from .inventory import InventoryManager
from .stone import StoneObject

if TYPE_CHECKING:
    from .action import Action


class StoneTrajectory:
    def __init__(self, id: int):
        self.id = id
        self.poses = []
        self.energy = 0.0
        self.vel_integral = 0.0
        self.settle_pose_count = 0
        self.settle_position_delta = 0.0
        self.settle_path_length = 0.0

    def add_pose(self, pose):
        self.poses.append(pose)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx]


@dataclass
class State:
    stone_set: List[int]
    stone_seq: List[int]
    stone_poses: Dict[int, np.ndarray]
    trajectories: List[Dict[int, StoneTrajectory]]
    action_history: List[Action]
    terminated: bool = False
    failed: bool = False
    simulation_settled: bool = True
    contact_points: List[dict] = field(default_factory=list)
    pose_identified_stone_ids: Set[int] = field(default_factory=set)

    def copy(self) -> "State":
        return State(
            stone_set=self.stone_set.copy(),
            stone_seq=self.stone_seq.copy(),
            stone_poses=self.stone_poses.copy(),
            trajectories=self.trajectories.copy(),
            action_history=[action.copy() for action in self.action_history],
            terminated=self.terminated,
            failed=self.failed,
            simulation_settled=bool(getattr(self, "simulation_settled", True)),
            contact_points=copy.deepcopy(self.contact_points),
            pose_identified_stone_ids={
                int(stone_id)
                for stone_id in getattr(self, "pose_identified_stone_ids", set())
            },
        )

    def latest_velocity_integrals(self) -> Dict[int, float]:
        """Velocity integral for every stone in the latest simulation."""
        if not self.trajectories:
            return {}
        return {
            int(stone_id): float(trajectory.vel_integral)
            for stone_id, trajectory in self.trajectories[-1].items()
            if hasattr(trajectory, "vel_integral")
        }


@dataclass
class Observation:
    pending_points: np.ndarray
    pending_faces: np.ndarray
    stacked_points: np.ndarray
    stacked_faces: np.ndarray
    sharpness: np.ndarray
    target_points: np.ndarray
    target_faces: np.ndarray
    scene_mask: np.ndarray
    action_mask: np.ndarray
    scene_height_map: Optional[np.ndarray]
    target_height_map: np.ndarray
    target_width: np.ndarray
    target_length: np.ndarray
    target_height: np.ndarray
    target_origin: np.ndarray
    height_map_xlim: np.ndarray
    height_map_ylim: np.ndarray
    stone_ids: Optional[np.ndarray] = None
    stone_poses: Optional[np.ndarray] = None
    dsf_points: Optional[np.ndarray] = None
    dsf_normals: Optional[np.ndarray] = None
    dsf_point_mask: Optional[np.ndarray] = None
    stone_physical_features: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, obs_dict: dict) -> "Observation":
        return cls(**obs_dict)

    @classmethod
    def to_dict(cls, obs: "Observation") -> dict:
        out = {f.name: getattr(obs, f.name) for f in fields(obs)}
        if out["scene_height_map"] is None and out["target_height_map"] is not None:
            out["scene_height_map"] = np.zeros_like(
                out["target_height_map"],
                dtype=np.float32,
            )
        return out

    @classmethod
    def to_heightmap_dict(cls, obs: "Observation") -> dict:
        scene_height_map = obs.scene_height_map
        target_height_map = obs.target_height_map
        if scene_height_map is None and target_height_map is not None:
            scene_height_map = np.zeros_like(target_height_map, dtype=np.float32)
        if target_height_map is None and scene_height_map is not None:
            target_height_map = np.zeros_like(scene_height_map, dtype=np.float32)
        if scene_height_map is None and target_height_map is None:
            scene_height_map = np.zeros((0, 0), dtype=np.float32)
            target_height_map = np.zeros((0, 0), dtype=np.float32)

        out = {
            "scene_height_map": np.asarray(scene_height_map, dtype=np.float16),
            "target_height_map": np.asarray(target_height_map, dtype=np.float16),
            "height_map_xlim": np.asarray(obs.height_map_xlim, dtype=np.float32),
            "height_map_ylim": np.asarray(obs.height_map_ylim, dtype=np.float32),
        }
        if obs.stone_ids is not None:
            out.update(
                {
                    "stone_ids": np.asarray(obs.stone_ids, dtype=np.int32),
                    "stone_poses": np.asarray(obs.stone_poses, dtype=np.float32),
                    "scene_mask": np.asarray(obs.scene_mask, dtype=bool),
                    "stone_physical_features": np.asarray(
                        obs.stone_physical_features, dtype=np.float32
                    ),
                    "target_width": np.asarray(obs.target_width, dtype=np.float32),
                    "target_length": np.asarray(obs.target_length, dtype=np.float32),
                    "target_height": np.asarray(obs.target_height, dtype=np.float32),
                    "target_origin": np.asarray(obs.target_origin, dtype=np.float32),
                }
            )
        return out

    def get_torch_tensors(self, device: str = "cuda:0") -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            t = torch.tensor(value, device=device)
            if t.dtype == torch.float64:
                t = t.float()
            out[f.name] = t
        return out


class ObservationBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def _num_action_samples(self, n_stone: int) -> int:
        max_pose_per_stone = int(
            self.cfg.action.get(
                "max_pose_per_stone", self.cfg.action.get("n_pose_per_stone", 1)
            )
        )
        return int(
            self.cfg.action.get("n_action_samples", n_stone * max_pose_per_stone)
        )

    def _banned_stone_indices(self, stone_set: List[int]) -> List[int]:
        action_cfg = self.cfg.get("action", {})
        banned_stone_ids = {
            int(stone_id) for stone_id in action_cfg.get("banned_stone_ids", [])
        }
        return [
            idx
            for idx, stone_id in enumerate(stone_set)
            if int(stone_id) in banned_stone_ids
        ]

    def _available_stone_indices(self, state: State) -> List[int]:
        banned = set(self._banned_stone_indices(state.stone_set))
        placed = set(state.stone_seq)
        return [
            idx
            for idx in range(len(state.stone_set))
            if idx not in banned and idx not in placed
        ]

    def build(self, inventory: InventoryManager, state: State) -> Observation:
        stones: List[StoneObject] = inventory.stones
        stone_seq: List[int] = state.stone_seq

        pending_points: np.ndarray = inventory.points.copy()
        pending_faces: np.ndarray = inventory.faces.copy()

        n_stone = len(state.stone_set)
        points_shape = [n_stone] + list(pending_points.shape[1:])
        faces_shape = [n_stone] + list(pending_faces.shape[1:])
        stacked_points: np.ndarray = np.inf * np.ones(points_shape)

        stacked_faces: np.ndarray = -np.ones(faces_shape, dtype=int)
        for idx in stone_seq:
            points: np.ndarray = pending_points[idx].copy()
            pose_mat: np.ndarray = stones[idx].get_pose_matrix()
            with np.errstate(invalid="ignore"):
                points = points @ pose_mat[:3, :3].T + pose_mat[:3, 3].reshape(1, -1)
            stacked_points[idx] = points
            stacked_faces[idx] = pending_faces[idx]

        scene_mask: np.ndarray = np.zeros(n_stone, dtype=bool)
        scene_mask[stone_seq] = True

        n_action_samples = self._num_action_samples(n_stone)
        max_pose_per_stone = int(
            self.cfg.action.get(
                "max_pose_per_stone", self.cfg.action.get("n_pose_per_stone", 1)
            )
        )
        max_real_samples = (
            len(self._available_stone_indices(state)) * max_pose_per_stone
        )
        n_real_samples = min(n_action_samples, max_real_samples)
        action_mask: np.ndarray = np.full(n_action_samples, -np.inf)
        action_mask[:n_real_samples] = 0.0

        if self._uses_height_map_for_xy():
            scene_height_map = inventory.get_height_map(state)
        else:
            scene_height_map = None

        # `get_height_map` / `get_target_height_map` set inventory.{xlim,ylim}
        # via `get_renderer`; capture them here as the source of truth for the
        # binning grid that downstream training uses.
        xlim = np.asarray(inventory.xlim, dtype=np.float32)
        ylim = np.asarray(inventory.ylim, dtype=np.float32)

        target_wall = inventory.target_wall
        stone_poses = np.zeros((n_stone, 7), dtype=np.float32)
        stone_poses[:, 6] = 1.0
        for idx in stone_seq:
            stone = stones[idx]
            pose = state.stone_poses.get(stone.id, stone.pose)
            stone_poses[idx] = np.asarray(pose, dtype=np.float32)

        height_map_shape = tuple(int(v) for v in self.cfg.height_map.resolution)
        if scene_height_map is None:
            scene_height_map = np.zeros(height_map_shape, dtype=np.float32)
        target_height_map = (
            inventory.target_height_map.copy()
            if inventory.target_height_map is not None
            else np.zeros_like(scene_height_map, dtype=np.float32)
        )
        return Observation(
            pending_points=pending_points,
            pending_faces=pending_faces,
            stacked_points=stacked_points,
            stacked_faces=stacked_faces,
            sharpness=inventory.sharpness.copy(),
            target_points=inventory.target_wall.geometries[0].points.copy(),
            target_faces=inventory.target_wall.geometries[0].faces.copy(),
            scene_mask=scene_mask,
            action_mask=action_mask,
            scene_height_map=scene_height_map,
            target_height_map=target_height_map.copy(),
            target_width=np.float32(target_wall.width),
            target_length=np.float32(target_wall.length),
            target_height=np.float32(target_wall.height),
            target_origin=np.asarray(target_wall.origin, dtype=np.float32),
            height_map_xlim=xlim,
            height_map_ylim=ylim,
            stone_ids=np.asarray(state.stone_set, dtype=np.int32),
            stone_poses=stone_poses,
            dsf_points=inventory.dsf_points.copy(),
            dsf_normals=inventory.dsf_normals.copy(),
            dsf_point_mask=inventory.dsf_point_mask.copy(),
            stone_physical_features=inventory.stone_physical_features.copy(),
        )

    def _uses_height_map_for_xy(self) -> bool:
        if bool(self.cfg.get("observation", {}).get("render_height_map", False)):
            return True
        planar_cfg = self.cfg.action.get("planar", {})
        score_model = str(planar_cfg.get("score_model", "heuristic"))
        return score_model in {"heuristic", "cnn"}
