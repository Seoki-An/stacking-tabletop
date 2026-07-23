from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from utils._phase_timer import timed

from ..contexts import (
    environment_ground_height,
    get_posegen,
    posegen,
)
from ..inventory import InventoryManager
from ..state import State
from ..stone import StoneObject


@dataclass
class PoseSolverSchedule:
    t_init: float = 0.0
    t_fin: float = 1.0
    dt: float = 1.0 / 3.0
    max_iter_warmup: int = 50
    max_iter_final: int = 75


class PoseSolver:
    """Cosine-scheduled pose optimization for placing one stone on top of the scene."""

    def __init__(
        self,
        cfg,
        schedule: Optional[PoseSolverSchedule] = None,
    ):
        self.cfg = cfg
        self.schedule = schedule or PoseSolverSchedule()
        self.ground_height = environment_ground_height(cfg)
        self.context: posegen.Context = get_posegen(
            cfg=cfg, ground_height=self.ground_height
        )
        self.body_id_to_stone_idx: dict[int, int] = {}
        self._scene_fingerprint: Optional[tuple] = None

    def reset_scene(self, inventory: InventoryManager, state: State) -> None:
        # Rebuilding the posegen scene wipes its scene-contact caches and ADMM
        # warm-start, so skip when the scene is unchanged (the parallel action
        # builder calls reset_scene once per candidate with an identical scene).
        fingerprint = self._compute_scene_fingerprint(inventory, state)
        if fingerprint == self._scene_fingerprint:
            return
        self._scene_fingerprint = None  # invalid until the rebuild completes
        self.context.clear()
        self.body_id_to_stone_idx = {}
        boundary_radius = self.cfg.action.get("posegen_boundary_radius", None)
        # When True, force-solve equilibrium only for the new target stone; every
        # already-placed stone becomes a rigid boundary (contacts kept, own
        # equilibrium not solved). Otherwise stones farther than
        # posegen_boundary_radius from the target-wall region are boundary.
        target_only = bool(self.cfg.action.get("posegen_target_only", False))
        for idx in state.stone_seq:
            is_boundary = target_only or (
                boundary_radius is not None
                and self._xy_distance_to_target(inventory, idx) > float(boundary_radius)
            )
            body_id = self.context.add_body(
                inventory.stones[idx].config, is_boundary=is_boundary
            )
            self.body_id_to_stone_idx[int(body_id)] = int(idx)
        self._scene_fingerprint = fingerprint

    def _compute_scene_fingerprint(
        self, inventory: InventoryManager, state: State
    ) -> tuple:
        """Identity of the posegen scene: the placed stones, their poses, and the
        boundary-marking options (everything reset_scene consumes)."""
        stones = tuple(
            (int(idx), inventory.stones[idx].pose.tobytes()) for idx in state.stone_seq
        )
        boundary_radius = self.cfg.action.get("posegen_boundary_radius", None)
        target_only = bool(self.cfg.action.get("posegen_target_only", False))
        return (
            stones,
            None if boundary_radius is None else float(boundary_radius),
            target_only,
        )

    @staticmethod
    def _xy_distance_to_target(inventory: InventoryManager, idx: int) -> float:
        """XY distance from a stone to the target-wall region (0 if inside it)."""
        wall = inventory.target_wall
        xy = np.asarray(inventory.stones[idx].pose[:2], dtype=float)
        origin = np.asarray(wall.origin[:2], dtype=float)
        half = 0.5 * np.array([wall.width, wall.length], dtype=float)
        outside = np.maximum(np.abs(xy - origin) - half, 0.0)
        return float(np.linalg.norm(outside))

    @timed("pose_solve")
    def solve(
        self,
        inventory: InventoryManager,
        stone: StoneObject,
    ) -> Tuple[np.ndarray, float, float, posegen.Solution]:
        # self._set_target_eps(inventory)

        c_gap = 0.0
        result: Optional[posegen.Solution] = None
        for t in self._time_steps():
            self._set_obj_weights(t, c_gap)
            self._set_trust_region(t)

            result = self.context.solve(stone.config)
            stone.pose = result.optimal_pose
            c_gap = result.c_gap

        c_feq = max(np.linalg.norm(w, ord=1) for w in result.net_wrench.values()) / max(
            st.mass for st in inventory.stones
        )
        return (
            result.optimal_pose.vectorized().copy(),
            float(c_feq),
            float(c_gap),
            result,
        )

    def _time_steps(self) -> np.ndarray:
        s = self.schedule
        return np.arange(s.t_init, s.t_fin + s.dt, s.dt)

    def _set_target_eps(self, inventory: InventoryManager) -> None:
        extents = [
            np.asarray(stone.local_aabb_extent()[:2], dtype=float)
            for stone in inventory.stones
        ]
        mean_half_diagonal = (
            float(np.mean([np.linalg.norm(0.5 * extent) for extent in extents]))
            if extents
            else 1.0
        )
        self.context.config().obj.eps_target = (
            1.0 - self.cfg.reward.IoU_thresh
        ) * mean_half_diagonal

    def _set_obj_weights(self, t: float, c_gap: float) -> None:
        obj = self.context.config().obj
        ramp = 0.5 * (1.0 - np.cos(np.pi * t))
        anti_ramp = 0.5 * (1.0 + np.cos(np.pi * t))
        sine = np.sin(np.pi * t)

        obj.set_k_wrench(ramp * np.diag([1, 1, 1, 10, 10, 10]))
        obj.k_comp = 10.0 * ramp
        obj.k_gap_c = 150.0 * ramp + 40.0 + t * 1e5 * c_gap
        obj.k_gap = 20.0 * sine
        obj.k_target = 5.0 * anti_ramp
        obj.k_xy = anti_ramp
        obj.k_potential = 0.10 + 0.05 * anti_ramp
        obj.k_reg = 0.0 * anti_ramp

    def _set_trust_region(self, t: float) -> None:
        tr = self.context.config().tr
        tr.tol_eps = 10 ** (-4 * (t + 1))
        tr.max_iter = (
            self.schedule.max_iter_final if t == 1.0 else self.schedule.max_iter_warmup
        )
