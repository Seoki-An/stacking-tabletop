"""Diffsim-based grasp-accessibility labels for the feasibility sampler.

This module asks diffsim's real place-scene grasp sampler whether a placed stone
can actually be grasped. For each candidate
it builds the target stone posed at the candidate pose and calls
``Context.sample_place_grasps`` against the place scene of already-placed stones
(no IK / no motion planning), labelling the candidate accessible iff at least one
grasp converges and passes the approach-direction filter.

This is offline data-generation only and is heavier than the proxies, so it is
gated by config and off by default.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from agent.env.components.contexts import environment_ground_height
from agent.env.components.state import State


class PlaceGraspSampler:
    """Labels placement candidates by sampling real place-scene grasps.

    One planner ``Context`` is built lazily and reused for the whole episode; the
    place bodies (already-placed stones) are swapped per parent via
    :meth:`set_scene`, while the registered planes persist.
    """

    def __init__(self, inventory, cfg, env_cfg, default_threads: int = 1) -> None:
        self.inventory = inventory
        self.num_samples = int(cfg.get("num_samples", 32))
        # 0 (or unset) -> match the actor's per-worker CPU budget. Running this
        # under Ray, parallelism already comes from the worker pool, so spawning
        # extra grasp threads per worker just oversubscribes the cores.
        self.n_threads = int(cfg.get("n_threads", 0)) or max(int(default_threads), 1)
        self.ground_height = float(environment_ground_height(env_cfg))
        self._context = None
        self._place_body_ids: list[int] = []
        self._scene_key = None
        self._score_cache: dict[
            tuple[int, tuple[float, ...]],
            tuple[bool, int, float],
        ] = {}

    def _ensure_context(self):
        if self._context is None:
            from planning.planning import get_planner

            self._context, _ = get_planner(
                pick_plane_height=self.ground_height,
                place_plane_height=self.ground_height,
                n_threads=self.n_threads,
            )
        return self._context

    def _posed_config(self, stone_idx: int, pose: np.ndarray):
        pose = np.asarray(pose, dtype=float)
        config = copy.deepcopy(self.inventory.stones[int(stone_idx)].config)
        config.pose.setPosition(pose[:3])
        config.pose.setOrientation(pose[3:7])
        return config

    def set_scene(self, state: State) -> None:
        """Register the already-placed stones as the place-scene obstacles."""
        scene_key = self._make_scene_key(state)
        if scene_key == self._scene_key:
            return

        context = self._ensure_context()
        for body_id in self._place_body_ids:
            context.remove_body(body_id)
        self._place_body_ids = []
        self._scene_key = scene_key
        self._score_cache = {}

        for idx in state.stone_seq:
            stone = self.inventory.stones[int(idx)]
            pose = state.stone_poses.get(stone.id)
            if pose is None:
                continue
            config = self._posed_config(int(idx), pose)
            self._place_body_ids.append(context.add_place_body(config))

    def _make_scene_key(self, state: State):
        key = []
        for idx in state.stone_seq:
            stone = self.inventory.stones[int(idx)]
            pose = state.stone_poses.get(stone.id)
            if pose is None:
                continue
            key.append((int(idx), tuple(np.round(np.asarray(pose), 6))))
        return tuple(key)

    def score(self, stone_idx: int, pose: np.ndarray) -> tuple[bool, int, float]:
        """Return ``(accessible, n_grasps, best_score)`` for one candidate.

        ``accessible`` is True iff diffsim returns at least one feasible grasp.
        On any diffsim failure the candidate is reported inaccessible.
        """
        if pose is None or np.asarray(pose).shape != (7,):
            return False, 0, 0.0
        cache_key = (
            int(stone_idx),
            tuple(np.round(np.asarray(pose, dtype=float), 6)),
        )
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        context = self._ensure_context()
        target = self._posed_config(int(stone_idx), pose)
        try:
            sols = context.sample_place_grasps(target, self.num_samples)
        except Exception as exc:  # pragma: no cover - diffsim runtime fallback
            print(f"[WARN] sample_place_grasps failed ({exc}); marking inaccessible")
            result = (False, 0, 0.0)
            self._score_cache[cache_key] = result
            return result

        n = len(sols)
        best_score = max((float(s.score) for s in sols), default=0.0)
        result = (n > 0, n, best_score)
        self._score_cache[cache_key] = result
        return result
