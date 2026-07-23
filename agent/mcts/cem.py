import copy
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from agent.config_views import mcts_root_proposal_config, mcts_validation_config
from agent.env.components.action import Action
from utils._phase_timer import phase

from .node import MCTS_Node
from .utils import is_duplicate_action, top_actions


DUPLICATE_YAW_THRESH = np.deg2rad(5.0)


class CEMRootMixin:
    def _record_root_planar_candidate_call(self) -> None:
        builder = getattr(self.env, "action_builder", None)
        calls = getattr(builder, "recent_planar_candidate_calls", None)
        if not calls:
            return
        call = calls[-1]
        call_index = int(call.get("call_index", -1))
        recorded = getattr(self, "_last_root_planar_candidate_calls", [])
        if recorded and int(recorded[-1].get("call_index", -2)) == call_index:
            return
        recorded.append(copy.deepcopy(call))
        self._last_root_planar_candidate_calls = recorded

    def _expand_root_with_cem(self, node: MCTS_Node, use_qfunction: bool) -> None:
        room = node.child_expansion_room()
        if room <= 0:
            return
        actions, scores = self._cem_root_proposal(node, use_qfunction, keep=room)
        actions, scores = self._remove_duplicate_actions(node, actions, scores)
        node.value_init = node.get_estimated_reward_to_go()
        self._append_children(node, actions, scores)

    def _cem_root_proposal(
        self,
        node: MCTS_Node,
        use_qfunction: bool,
        keep: Optional[int] = None,
    ) -> Tuple[List[Action], np.ndarray]:
        cfg = mcts_root_proposal_config(self.cfg)
        root_keep = int(
            cfg.root_keep(
                self.cfg.get("max_first_children_num", self.cfg.max_children_num)
            )
        )
        population = max(cfg.population(root_keep), self._root_action_sample_count())
        elite = cfg.elite(population)
        iterations = cfg.iterations
        keep = int(root_keep if keep is None else keep)
        xy_std = cfg.mutation_xy_std
        xy_mode = cfg.mutation_xy_mode
        yaw_std = cfg.mutation_yaw_std_rad
        fresh_fraction = cfg.fresh_fraction
        max_per_stone = cfg.max_per_stone
        sampling_priority = self._sampling_priority_for_root(node)
        with phase("mcts_cem_proposal"):
            return self._cem_root_proposal_once(
                node,
                use_qfunction,
                population,
                elite,
                iterations,
                keep,
                xy_std,
                xy_mode,
                yaw_std,
                max_per_stone=max_per_stone,
                fresh_fraction=fresh_fraction,
                sampling_priority=sampling_priority,
            )

    def _sample_root_action_pool(
        self,
        node: MCTS_Node,
        use_qfunction: bool,
        population: int,
        sampling_priority: Optional[str] = None,
    ) -> Tuple[List[Action], np.ndarray]:
        scene_height_map = node.obs.scene_height_map if node.obs is not None else None
        with phase("mcts_cem_sample"):
            action_samples, action_mask = self.env.get_action_samples(
                node.state,
                scene_height_map=scene_height_map,
                n_action_samples=population,
                sampling_priority=sampling_priority,
            )
        self._record_root_planar_candidate_call()
        actions = list(action_samples)
        scores = self._score_cem_actions(
            node,
            actions,
            np.isfinite(action_mask),
            use_qfunction,
        )
        external_seed_actions = self._external_seed_actions_for_root()
        if external_seed_actions:
            seed_scores = self._score_cem_actions(
                node,
                external_seed_actions,
                np.ones(len(external_seed_actions), dtype=bool),
                use_qfunction,
            )
            actions = actions + external_seed_actions
            scores = np.concatenate([scores, seed_scores])
        result = top_actions(actions, scores, keep=population)
        return result

    def _external_seed_actions_for_root(self) -> List[Action]:
        seeds = getattr(self, "_external_seed_actions", []) or []
        return [action.copy() for action in seeds if action is not None]

    def _cem_root_proposal_once(
        self,
        node: MCTS_Node,
        use_qfunction: bool,
        population: int,
        elite: int,
        iterations: int,
        keep: int,
        xy_std: float,
        xy_mode: str,
        yaw_std: float,
        max_per_stone: Optional[int] = None,
        fresh_fraction: float = 0.0,
        sampling_priority: Optional[str] = None,
    ) -> Tuple[List[Action], np.ndarray]:

        scene_height_map = node.obs.scene_height_map if node.obs is not None else None
        actions, scores = self._sample_root_action_pool(
            node,
            use_qfunction,
            population,
            sampling_priority=sampling_priority,
        )
        actions, scores = top_actions(
            actions,
            scores,
            keep=max(elite, keep),
        )

        for _ in range(max(iterations - 1, 0)):
            # Diversity-aware elite selection: mutations spread across different stones.
            parents, _ = top_actions(
                actions,
                scores,
                keep=elite,
                max_per_stone=max_per_stone,
            )
            if not parents:
                break
            fresh_count = int(round(population * np.clip(fresh_fraction, 0.0, 1.0)))
            mutate_count = max(population - fresh_count, 0)
            with phase("mcts_cem_mutate"):
                candidates = self._mutate_actions(
                    node, parents, mutate_count, xy_std, xy_mode, yaw_std
                )
            if fresh_count > 0:
                fresh_samples, fresh_mask = self.env.get_action_samples(
                    node.state,
                    scene_height_map=scene_height_map,
                    n_action_samples=fresh_count,
                    sampling_priority=sampling_priority,
                )
                self._record_root_planar_candidate_call()
                candidates.extend(
                    action
                    for action, valid in zip(fresh_samples, np.isfinite(fresh_mask))
                    if valid
                )
            candidate_scores = self._score_cem_actions(
                node,
                candidates,
                np.ones(len(candidates), dtype=bool),
                use_qfunction,
            )
            actions, scores = top_actions(
                actions + candidates,
                np.concatenate([scores, candidate_scores]),
                keep=max(elite, keep),
            )

        actions, scores = top_actions(
            actions,
            scores,
            keep=keep,
            max_per_stone=max_per_stone,
        )
        return actions, scores

    def _score_cem_actions(
        self,
        node: MCTS_Node,
        actions: List[Action],
        valid_mask: np.ndarray,
        use_qfunction: bool,
    ) -> np.ndarray:
        scores = self._score_actions(node, actions, valid_mask, use_qfunction)
        weight = mcts_root_proposal_config(self.cfg).height_map_score_weight
        for idx, action in enumerate(actions):
            diagnostics = getattr(action, "diagnostics", None)
            if diagnostics is None:
                diagnostics = {}
                action.diagnostics = diagnostics
            normalized = diagnostics.get("height_map_score_normalized", None)
            bonus = 0.0
            if np.isfinite(scores[idx]) and normalized is not None:
                normalized = float(normalized)
                if np.isfinite(normalized):
                    bonus = weight * float(np.clip(normalized, 0.0, 1.0))
                    scores[idx] += bonus
            diagnostics["cem_height_map_score_bonus"] = float(bonus)
        return scores

    def _remove_duplicate_actions(
        self,
        node: MCTS_Node,
        actions: List[Action],
        scores: np.ndarray,
    ) -> Tuple[List[Action], np.ndarray]:
        if not actions:
            return actions, scores

        kept_actions = []
        kept_scores = []
        duplicate_xy_threshold = mcts_root_proposal_config(
            self.cfg
        ).duplicate_xy_threshold
        existing = [child.action for child in node.children if child.action is not None]
        if node.depth == 0:
            existing.extend(
                action
                for action in getattr(self, "_root_retry_rejected_actions", [])
                if action is not None
            )
        for action, score in zip(actions, scores):
            if is_duplicate_action(
                action,
                existing + kept_actions,
                duplicate_xy_threshold,
                DUPLICATE_YAW_THRESH,
            ):
                continue
            kept_actions.append(action)
            kept_scores.append(score)
        return kept_actions, np.asarray(kept_scores, dtype=float)

    def _mutate_actions(
        self,
        node: MCTS_Node,
        parents: List[Action],
        population: int,
        xy_std: float,
        xy_mode: str,
        yaw_std: float,
    ) -> List[Action]:
        mutated = []
        for _ in range(population):
            parent = parents[np.random.randint(len(parents))]
            init_pose = parent.init_pose.copy()
            if xy_mode == "gaussian" and xy_std > 0.0:
                init_pose[:2] += np.random.normal(0.0, xy_std, size=2)
            elif xy_mode != "anchored":
                raise ValueError(
                    "root_proposal.mutation_xy_mode must be 'gaussian' or 'anchored'"
                )
            yaw = np.random.normal(0.0, yaw_std)
            init_pose[3:] = (
                Rotation.from_euler("z", yaw) * Rotation.from_quat(init_pose[3:])
            ).as_quat()
            try:
                mutated.append(
                    self.env.get_action(node.state, init_pose, parent.stone_idx)
                )
            except Exception:
                continue
        return mutated

    def _thread_env(self):
        env = getattr(self._thread_local, "env", None)
        if env is None:
            env_args = dict(self.env_args)
            env_args["cfg"] = copy.deepcopy(self.env_args["cfg"])
            env_args["n_threads"] = 1
            env_args["build_action_builder"] = False
            env = self.env_class(env_args)
            env.inventory.target_wall = self.env.inventory.target_wall.copy()
            self._thread_local.env = env
        return env

    def _simulation_executor_instance(self) -> ThreadPoolExecutor:
        executor = getattr(self, "_simulation_executor", None)
        if executor is None:
            n_threads = mcts_validation_config(self.cfg).worker_threads(
                self.env_args
            )
            executor = ThreadPoolExecutor(max_workers=max(int(n_threads), 1))
            self._simulation_executor = executor
        return executor

    def _shutdown_simulation_executor(self) -> None:
        executor = getattr(self, "_simulation_executor", None)
        self._simulation_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def _root_action_sample_count(self) -> int:
        try:
            return max(int(self.env.cfg.action.get("n_action_samples", 1)), 1)
        except Exception:
            return 1
