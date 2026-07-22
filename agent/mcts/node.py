import copy
import math
import random

import numpy as np

import gc

from typing import Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf


from agent.config_views import (
    mcts_action_generation_config,
    mcts_root_proposal_config,
)
from agent.env.components.state import State, Observation
from agent.env.components.action import Action, ActionList
from loader.heightmap_feasibility import stone_heightmaps_from_mesh


class MCTS_Node:

    def __init__(
        self,
        cfg: OmegaConf,
        action: Action = None,
        q_value_init: float = 0.0,
        parent: Union["MCTS_Node", None] = None,
        depth: int = 0,
    ):
        """
        action = (action): action generated the current state
        parent = MCTS_Node
        depth = (int)
        """
        self.cfg: OmegaConf = cfg
        self.action: Action = action
        self.q_value_init: float = q_value_init
        self.value_init: float = 0.0

        self.parent: Union["MCTS_Node", None] = parent
        self.depth: int = depth

        self.children: List["MCTS_Node"] = []
        self.visits: int = 0
        self.q_value: float = 0.0

        self.state: State = None
        self.obs: Observation = None
        self.reward: float = None
        self.info: dict = None

        self._failed: bool = False
        self._done: bool = False
        self._is_simulated: bool = False
        self._aggregated: bool = False
        self._action_generation_exhausted: bool = False
        self._debug_skipped_child_reasons: Dict[str, int] = {}

        self.confidence: float = self.cfg.get("confidence", 1.0)

    def update_state(
        self,
        state: State,
        obs: Observation,
        reward: float,
        done: bool,
        failed: bool,
        info: dict = None,
    ):
        """
        state = State
        obs = Observation
        reward = (float)
        done = (bool)
        failed = (bool)
        info = {Dict} (other information)
        """
        self.state = state
        self.obs = obs
        self.reward = reward
        self.info = info

        self._done = done
        self._failed = failed

    def select_child(
        self,
        c_param: float,
        epsilon: float = 0.0,
        include_failed: bool = False,
    ) -> "MCTS_Node":
        if len(self.children) == 0:
            return self

        expandable = self.expandable_children()
        if expandable and (not self.is_fully_expanded(include_failed)):
            if np.random.rand() < epsilon:
                return random.choice(expandable)
            return max(expandable, key=lambda child: child.q_value_init)

        children = self.simulated_children(include_failed)
        if not children:
            return self

        priors = np.asarray([child.q_value_init for child in children], dtype=float)
        priors = priors - np.max(priors)
        priors = np.exp(priors)
        priors = priors / max(float(np.sum(priors)), 1e-12)
        weights = [
            child.q_value / max(child.visits, 1.0)
            + c_param * p * math.sqrt(max(self.visits, 1)) / (child.visits + 1.0)
            for child, p in zip(children, priors)
        ]
        return children[int(np.argmax(weights))]

    def get_leaves(self) -> List["MCTS_Node"]:
        if self._done and self._is_simulated:
            return [self]
        else:
            leaves = []
            for child in self.children:
                if child.is_simulated:
                    leaves.extend(child.get_leaves())
            return leaves

    def best_child(self) -> Tuple["MCTS_Node", float]:
        if len(self.children) > 0:
            choices_weights = []
            for child in self.children:
                if child.is_simulated:
                    choices_weights.append(child.q_value / max(child.visits, 1.0))
                else:
                    choices_weights.append(-np.inf)
            best_idx = np.argmax(choices_weights)
            if choices_weights[best_idx] > -np.inf:
                return self.children[best_idx], choices_weights[best_idx]
            else:
                return self, -np.inf
        else:
            return self, self.q_value / max(self.visits, 1.0)

    def visit(self, reward_to_go: float):
        self.visits += 1
        self.q_value += reward_to_go

    def reset(self, preserve_tree: bool = False, depth: int = 0):
        self.depth = depth
        if preserve_tree:
            if depth == 0:
                self._action_generation_exhausted = False
                if not self._done and len(self.selectable_children()) == 0:
                    self.children = []
            for child in self.children:
                child.reset(preserve_tree, depth + 1)
        else:
            self.visits = 0
            self.q_value = 0
            self.reward = 0
            self.children = []
            self.parent = None
            self._action_generation_exhausted = False
            self._debug_skipped_child_reasons = {}

    def reset_search_statistics(self) -> None:
        """Discard old search evidence while keeping the preserved subtree."""
        self.visits = 0
        self.q_value = -np.inf if self.failed else 0.0
        for child in self.children:
            child.reset_search_statistics()

    def is_terminal(self) -> bool:
        no_path_forward = (
            len(self.selectable_children()) == 0 and not self.can_expand_children()
        )
        return (
            self._done
            or self._failed
            or no_path_forward
            or (self.depth > self.cfg.max_depth)
        ) and self._is_simulated

    def is_fully_expanded(self, include_failed: bool = False) -> bool:
        limit = self._child_limit()
        if len(self.expandable_children()) > 0:
            return False
        return (
            (
                (len(self.simulated_children(False)) >= limit)
                or not self.can_expand_children()
                or (self.depth > self.cfg.max_depth)
            )
            and self._is_simulated
            and (len(self.children) > 0)
        )

    def _child_limit(self) -> int:
        hard_limit = (
            self._root_child_limit()
            if self.depth == 0
            else int(self.cfg.max_children_num)
        )
        if self.depth == 0:
            return hard_limit
        widening_cfg = self.cfg.get("widening", {})
        if not bool(widening_cfg.get("enabled", True)):
            return hard_limit
        c = float(widening_cfg.get("c", 2.0))
        alpha = float(widening_cfg.get("alpha", 0.5))
        visits = max(int(self.visits), 1)
        return min(hard_limit, max(int(math.ceil(c * (visits**alpha))), 1))

    def is_expandable(self) -> bool:
        return (
            self.can_expand_children()
            and self.q_value_init + self.q_value > -np.inf
        )

    def can_expand_children(self, include_failed: bool = False) -> bool:
        if (
            self._done
            or self._failed
            or not self._is_simulated
            or self._action_generation_exhausted
        ):
            return False
        # max_depth is the deepest allowed node depth (root = 0), so a node at
        # max_depth cannot expand children.
        if self.depth >= self.cfg.max_depth:
            return False
        return (
            len(self.selectable_children()) < self._child_limit()
            and len(self.children) < self._attempt_limit()
        )

    def child_expansion_room(self, include_failed: bool = False) -> int:
        selectable_room = self._child_limit() - len(self.selectable_children())
        attempt_room = self._attempt_limit() - len(self.children)
        return max(min(selectable_room, attempt_room), 0)

    def _attempt_limit(self) -> int:
        hard_limit = (
            self._root_child_limit()
            if self.depth == 0
            else int(self.cfg.max_children_num)
        )

        multiplier = mcts_action_generation_config(self.cfg).attempt_multiplier
        return max(int(math.ceil(hard_limit * multiplier)), hard_limit)

    def _root_child_limit(self) -> int:
        fallback = self.cfg.get("max_first_children_num", self.cfg.max_children_num)
        return mcts_root_proposal_config(self.cfg).root_keep(fallback)

    def set_action_generation_exhausted(self, exhausted: bool):
        self._action_generation_exhausted = exhausted

    def record_skipped_children(self, reason: str, count: int = 1) -> None:
        count = int(count)
        if count <= 0:
            return
        reason = str(reason)
        self._debug_skipped_child_reasons[reason] = (
            self._debug_skipped_child_reasons.get(reason, 0) + count
        )

    def simulated_children(self, include_failed: bool = False) -> List["MCTS_Node"]:
        children = []
        for child in self.children:
            if child.is_simulated and (include_failed or not child.failed):
                children.append(child)
        return children

    def sampleable_simulated_children(
        self,
        include_failed: bool = False,
    ) -> List["MCTS_Node"]:
        children = []
        for child in self.simulated_children(include_failed=include_failed):
            if child.state is None or child.obs is None:
                continue
            children.append(child)
        return children

    def expandable_children(self) -> List["MCTS_Node"]:
        children = []
        for child in self.selectable_children():
            if not child.is_simulated:
                children.append(child)
        return children

    def selectable_children(self) -> List["MCTS_Node"]:
        children = []
        for child in self.children:
            if child.q_value_init + child.q_value > -np.inf:
                children.append(child)
        return children

    def child_debug_counts(self) -> Dict[str, int]:
        simulated_all = self.simulated_children(include_failed=True)
        simulated_ok = self.simulated_children(include_failed=False)
        return {
            "children": len(self.children),
            "selectable": len(self.selectable_children()),
            "expandable": len(self.expandable_children()),
            "simulated": len(simulated_all),
            "nonfailed": len(simulated_ok),
            "failed": len([child for child in simulated_all if child.failed]),
            "can_expand": int(self.can_expand_children()),
            "exhausted": int(self._action_generation_exhausted),
        }

    def tree_debug_counts(self) -> Dict[str, int]:
        counts = {
            "nodes": 1,
            "simulated": int(self._is_simulated),
            "unsimulated": int(not self._is_simulated),
            "frontier_leaves": 0,
            "terminal": int(self.is_terminal()),
            "max_depth": self.depth,
        }
        simulated_children = self.simulated_children(include_failed=True)
        if self._is_simulated and len(simulated_children) == 0:
            counts["frontier_leaves"] = 1
        for child in self.children:
            child_counts = child.tree_debug_counts()
            for key in ("nodes", "simulated", "unsimulated", "frontier_leaves", "terminal"):
                counts[key] += child_counts[key]
            counts["max_depth"] = max(counts["max_depth"], child_counts["max_depth"])
        return counts

    def search_width_debug(self) -> Dict[str, Union[int, List[Dict[str, object]]]]:
        by_depth: Dict[int, Dict[str, float]] = {}
        unselectable_by_depth: Dict[int, Dict[str, int]] = {}
        skipped_by_depth: Dict[int, Dict[str, int]] = {}

        def visit(node: "MCTS_Node") -> None:
            depth = int(node.depth)
            row = by_depth.setdefault(
                depth,
                {
                    "depth": float(node.depth),
                    "nodes": 0.0,
                    "simulated_nodes": 0.0,
                    "expanded_nodes": 0.0,
                    "child_edges": 0.0,
                    "selectable_child_edges": 0.0,
                    "expandable_child_edges": 0.0,
                    "simulated_child_edges": 0.0,
                    "nonfailed_simulated_child_edges": 0.0,
                    "failed_simulated_child_edges": 0.0,
                    "max_children": 0.0,
                    "max_selectable_children": 0.0,
                    "max_child_limit": 0.0,
                    "max_attempt_limit": 0.0,
                },
            )
            children = len(node.children)
            selectable = len(node.selectable_children())
            expandable = len(node.expandable_children())
            simulated_all = node.simulated_children(include_failed=True)
            simulated_ok = node.simulated_children(include_failed=False)

            row["nodes"] += 1.0
            row["simulated_nodes"] += float(node._is_simulated)
            row["expanded_nodes"] += float(children > 0)
            row["child_edges"] += float(children)
            row["selectable_child_edges"] += float(selectable)
            row["expandable_child_edges"] += float(expandable)
            row["simulated_child_edges"] += float(len(simulated_all))
            row["nonfailed_simulated_child_edges"] += float(len(simulated_ok))
            row["failed_simulated_child_edges"] += float(
                len(simulated_all) - len(simulated_ok)
            )
            row["max_children"] = max(row["max_children"], float(children))
            row["max_selectable_children"] = max(
                row["max_selectable_children"], float(selectable)
            )
            row["max_child_limit"] = max(
                row["max_child_limit"], float(node._child_limit())
            )
            row["max_attempt_limit"] = max(
                row["max_attempt_limit"], float(node._attempt_limit())
            )
            if node._debug_skipped_child_reasons:
                skipped_counts = skipped_by_depth.setdefault(depth, {})
                for reason, count in node._debug_skipped_child_reasons.items():
                    skipped_counts[reason] = skipped_counts.get(reason, 0) + int(count)

            for child in node.children:
                reason = child.unselectable_reason()
                if reason is not None:
                    reason_counts = unselectable_by_depth.setdefault(depth, {})
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                visit(child)

        visit(self)

        rows: List[Dict[str, object]] = []
        for depth in sorted(by_depth):
            row = by_depth[depth]
            expanded = max(row["expanded_nodes"], 1.0)
            out = {key: int(value) for key, value in row.items()}
            out["depth"] = int(depth)
            out["avg_children_per_expanded"] = row["child_edges"] / expanded
            out["avg_selectable_per_expanded"] = (
                row["selectable_child_edges"] / expanded
            )
            out["unselectable_reasons"] = dict(
                sorted(
                    unselectable_by_depth.get(depth, {}).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            skipped = dict(
                sorted(
                    skipped_by_depth.get(depth, {}).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            )
            out["skipped_child_reasons"] = skipped
            out["skipped_child_edges"] = int(sum(skipped.values()))
            rows.append(out)

        max_children = max((row["max_children"] for row in rows), default=0.0)
        return {
            "max_depth": int(max(by_depth) if by_depth else self.depth),
            "total_nodes": int(sum(row["nodes"] for row in rows)),
            "total_child_edges": int(sum(row["child_edges"] for row in rows)),
            "total_skipped_child_edges": int(
                sum(row["skipped_child_edges"] for row in rows)
            ),
            "max_children": int(max_children),
            "by_depth": rows,
        }

    def unselectable_reason(self) -> Optional[str]:
        if self.q_value_init + self.q_value > -np.inf:
            return None

        info = self.info or {}
        reason = info.get("mcts_unselectable_reason", None)
        if reason:
            return str(reason)
        if bool(info.get("final_validation_failed", False)):
            return str(info.get("final_validation_failure", "final_validation"))
        if bool(info.get("posegen_pruned", False)):
            return "posegen_" + str(info.get("posegen_prune_reason", "unknown"))

        state = self.state
        if bool(getattr(state, "failed", False)):
            return "simulation_failure_unattributed"
        if self.failed:
            return "failed"
        if self.action is None:
            return "missing_action"
        if int(getattr(self.action, "stone_idx", -1)) < 0:
            return "invalid_stone_idx"

        diagnostics = getattr(self.action, "diagnostics", {}) or {}
        reason = diagnostics.get("mcts_rejection_reason", None)
        if reason:
            return str(reason)

        if not np.isfinite(self.q_value_init):
            return "nonfinite_prior"
        if not np.isfinite(self.q_value):
            return "nonfinite_value"
        return "unselectable"

    def number_of_leaves(self) -> int:
        n_children = 0
        for child in self.simulated_children():
            n_children += child.number_of_leaves()
        if (len(self.simulated_children()) == 0) and self._is_simulated:
            n_children += 1

        return n_children

    def max_child_depth(self) -> int:
        depth = self.depth
        for child in self.simulated_children():
            depth = max(depth, child.max_child_depth())
        return depth

    def _make_sample(
        self,
        action: Action,
        reward: float,
        next_node: "MCTS_Node",
        action_samples: Dict[str, np.ndarray],
        next_action_samples: Dict[str, np.ndarray],
        q_mcts: np.ndarray,
        pi_mcts: np.ndarray,
        n_step: int,
    ) -> Dict[str, Union[dict, float, bool, int]]:
        return {
            "state": Observation.to_heightmap_dict(self.obs),
            "action": self._make_action_sample(action),
            "reward": reward,
            "next_state": Observation.to_heightmap_dict(next_node.obs),
            "action_samples": action_samples,
            "next_action_samples": next_action_samples,
            "q_mcts": q_mcts,
            "pi_mcts": pi_mcts,
            "done": next_node.done,
            "failed": next_node.failed,
            "visits": self.visits,
            "n_step": n_step,
        }

    def make_multistep_sample(
        self,
        action: Action,
        reward_to_go: float,
        next_node: "MCTS_Node",
        n_step: int,
    ) -> Dict[str, Union[dict, float, bool, int]]:
        """Build a committed-trajectory return from this snapshotted root."""
        if not self._aggregated or not hasattr(self, "actions"):
            raise RuntimeError("one-step aggregation must snapshot the node first")
        return self._make_sample(
            action=action,
            reward=reward_to_go,
            next_node=next_node,
            action_samples=self.actions,
            next_action_samples=self.actions,
            q_mcts=self.q_mcts,
            pi_mcts=self.pi_mcts,
            n_step=n_step,
        )

    def _make_action_sample(self, action: Action) -> dict:
        out = Action.to_dict(action)
        bottom, top = self._action_stone_height_maps(action)
        out["stone_bottom_height_map"] = bottom
        out["stone_top_height_map"] = top
        return out

    def _action_stone_height_maps(self, action: Action) -> Tuple[np.ndarray, np.ndarray]:
        if self.obs is None:
            return np.zeros((0, 0), dtype=np.float16), np.zeros((0, 0), dtype=np.float16)
        height_map = (
            self.obs.scene_height_map
            if self.obs.scene_height_map is not None
            else self.obs.target_height_map
        )
        shape = tuple(np.asarray(height_map).shape)
        zeros = np.zeros(shape, dtype=np.float16)
        if action is None or self.obs is None or len(shape) != 2:
            return zeros, zeros
        stone_idx = int(action.stone_idx)
        if stone_idx < 0 or stone_idx >= len(self.obs.pending_points):
            return zeros, zeros

        bottom, top, _ = stone_heightmaps_from_mesh(
            self.obs.pending_points[stone_idx],
            self.obs.pending_faces[stone_idx],
            action.pose,
            self.obs.height_map_xlim,
            self.obs.height_map_ylim,
            shape,
        )
        return bottom.astype(np.float16), top.astype(np.float16)

    def _snapshot_children_info(self) -> None:
        # Cache action/q/pi distributions over the current child set on `self`.
        # Multistep aggregation runs after the sampler prunes ancestors down to
        # {best_child, terminals}, so it must read the snapshot taken here —
        # before pruning — rather than recomputing from `self.children`.
        width = self._snapshot_child_width()
        actions = [child.action for child in self.children]
        self.actions = self._pad_action_samples(
            ActionList(actions).get_dict_array(),
            width,
        )
        q_values = np.array(
            [child.q_value / max(child.visits, 1.0) for child in self.children],
            dtype=np.float32,
        )
        if self.visits > 0:
            pi_values = np.array(
                [child.visits / self.visits for child in self.children],
                dtype=np.float32,
            )
        else:
            pi_values = np.zeros(len(self.children), dtype=np.float32)
        self.q_mcts = self._pad_vector(q_values, width, fill_value=-np.inf)
        self.pi_mcts = self._pad_vector(pi_values, width, fill_value=0.0)

    def _snapshot_child_width(self) -> int:
        fallback = self.cfg.get("max_first_children_num", self.cfg.max_children_num)
        root_cfg = mcts_root_proposal_config(self.cfg)
        root_keep = root_cfg.root_keep(fallback)
        root_population = root_cfg.population(root_keep)
        return max(self._attempt_limit(), root_population, len(self.children), 1)

    @staticmethod
    def _pad_vector(values: np.ndarray, width: int, fill_value: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        out = np.full(width, fill_value, dtype=np.float32)
        n = min(len(values), width)
        if n > 0:
            out[:n] = values[:n]
        return out

    @staticmethod
    def _pad_action_samples(
        actions: Dict[str, np.ndarray],
        width: int,
    ) -> Dict[str, np.ndarray]:
        padded = {}
        for key, values in actions.items():
            values = np.asarray(values)
            shape = (width,) + tuple(values.shape[1:])
            if key in {"stone_idx", "stone_id"}:
                fill_value = -1
            else:
                fill_value = 0.0
            out = np.full(shape, fill_value, dtype=values.dtype)
            n = min(values.shape[0], width)
            if n > 0:
                out[:n] = values[:n]
            padded[key] = out
        return padded

    def aggregate_tree_states(self) -> List[Dict[str, Union[dict, float, list, bool]]]:
        samples: List[Dict] = []
        if self._done or self.depth > self.cfg.max_depth:
            return samples

        self._snapshot_children_info()
        for child in self.sampleable_simulated_children(include_failed=True):
            # In-sample TD: Q(s', xy) is bootstrapped only over xys that appear
            # in s's child distribution. Pre-snapshot child so its action_samples
            # are available; fall back to self's set when child is an unexpanded
            # leaf (the bootstrap is dominated by `done` in that case anyway).
            if child.children and not child.failed:
                child._snapshot_children_info()
                next_actions = child.actions
            else:
                next_actions = self.actions

            if not (self._aggregated and child._aggregated):
                self.set_aggregated(True)
                child.set_aggregated(True)
                samples.append(
                    self._make_sample(
                        action=child.action,
                        reward=child.reward,
                        next_node=child,
                        action_samples=self.actions,
                        next_action_samples=next_actions,
                        q_mcts=self.q_mcts,
                        pi_mcts=self.pi_mcts,
                        n_step=1,
                    )
                )
            if len(child.children) > 0 and not child.failed:
                samples.extend(child.aggregate_tree_states())

        return samples

    def get_estimated_reward_to_go(self) -> float:
        if self.state is None:
            return 0.0

        value_cfg = self.cfg.reward.get("value_estimate", {})
        remaining = len(self.state.stone_set) - len(self.state.stone_seq)
        # max_depth is the deepest node depth (root = 0); a node at max_depth
        # still estimates one placement beyond itself.
        depth_budget = max(int(self.cfg.max_depth) + 1 - int(self.depth), 0)
        remaining = min(remaining, depth_budget)
        horizon = value_cfg.get("horizon", None)
        if horizon is not None:
            remaining = min(remaining, max(int(horizon), 0))

        reward_to_go = 0.0
        for _ in range(remaining):
            reward_to_go = self.cfg.reward.mean + self.cfg.reward.discount * reward_to_go

        if not np.isfinite(reward_to_go):
            return 0.0
        return reward_to_go

    def prune_to_best_child(self, save_terminal_nodes: bool = False):
        # remove children except the best child
        choices_weights = []
        new_children = []
        if len(self.children) > 0:
            for child in self.children:
                if child.is_simulated:
                    choices_weights.append(child.q_value / max(child.visits, 1.0))
                else:
                    choices_weights.append(-np.inf)
                if save_terminal_nodes and child.done and not child.failed:
                    new_children.append(child)
            best_idx = np.argmax(choices_weights)
            if choices_weights[best_idx] > -np.inf:
                new_children.append(self.children[best_idx])
            self.children = new_children
            gc.collect()

    def divide(self, n: int) -> List["MCTS_Node"]:
        selectable_children = self.selectable_children()
        selectable_values = [child.q_value_init for child in selectable_children]
        k, m = divmod(len(selectable_children), n)

        sort_indices = np.argsort(selectable_values)[::-1]
        sorted_children = [selectable_children[i] for i in sort_indices]

        sort_indices = np.arange(n * (k + 1)).reshape(k + 1, n).transpose()
        for i in range(k + 1):
            if i % 2 == 1:
                sort_indices[:, i] = sort_indices[::-1, i]
        sort_indices = sort_indices.flatten()
        sort_indices = sort_indices[
            ~np.isin(sort_indices, np.arange(len(selectable_children), n * (k + 1)))
        ]
        sorted_children = [sorted_children[i] for i in sort_indices]

        l_node = []

        for i in range(n):
            node = copy.deepcopy(self)
            node.children = sorted_children[
                i * k + min(i, m) : (i + 1) * k + min(i + 1, m)
            ]
            for child in node.children:
                child.parent = node
            l_node.append(node)
        return l_node

    @property
    def done(self) -> bool:
        return self._done

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def is_simulated(self) -> bool:
        return self._is_simulated

    @property
    def aggregated(self) -> bool:
        return self._aggregated

    def set_is_simulated(self, is_simulated: bool):
        self._is_simulated = is_simulated

    def set_aggregated(self, aggregated: bool):
        self._aggregated = aggregated

    def set_failed(self, failed: bool):
        self._failed = failed
