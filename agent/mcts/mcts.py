import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from typing import Callable, Dict, List, Optional, Tuple
from collections import Counter
from omegaconf import OmegaConf
import gymnasium as gym

from agent.env.components.state import State, Observation
from agent.env.components.action import Action
from agent.env.components.action.support import ground_placement_connected_ok
from agent.config_views import (
    mcts_action_generation_config,
    mcts_validation_config,
    mcts_sampling_config,
    support_config,
)
from utils._phase_timer import phase, timed, reset, format_summary

from .cem import CEMRootMixin
from .node import MCTS_Node
from .root_lookahead import RootLookaheadMixin
from .utils import (
    best_sequence_from_node,
    place_plane_gap_failure_reason,
    final_support_ok,
    place_scene_gap_failure_reason,
    refined_action_from_state,
    settled_action_pose,
)

TREE_SIMULATION_DEPTH = 5  # default deepest node depth simulated; overridden by algorithm.mcts.tree_simulation_depth
CandidateFilter = Callable[[MCTS_Node, MCTS_Node], Optional[MCTS_Node]]


class MonteCarloTreeSearch(CEMRootMixin, RootLookaheadMixin):

    def __init__(
        self,
        cfg: OmegaConf,
        env_class: gym.Env,
        env_args: Dict[str, OmegaConf],
    ):
        self.cfg = cfg
        self.verbose = bool(cfg.get("verbose", False))
        self.env_class = env_class
        _mcts_env_args = dict(env_args)
        _mcts_env_args["fast_sim"] = True
        self.env_args = _mcts_env_args
        self.env = env_class(_mcts_env_args)
        self._thread_local = threading.local()
        self._simulation_executor: Optional[ThreadPoolExecutor] = None
        self._search_deadline: Optional[float] = None
        self._execution_rejected_actions: List[Action] = []
        self._execution_rejected_stone_ids: set[int] = set()
        self._external_seed_actions: List[Action] = []
        self._last_final_validation_debug_nodes: List[MCTS_Node] = []
        self._last_root_planar_candidate_calls: List[Dict] = []
        self._last_search_width_stats: Optional[Dict] = None
        self._root_long_validation_failures: List[MCTS_Node] = []
        self._root_long_validation_stats: Dict = {}
        self._root_retry_rejected_actions: List[Action] = []
        self._mean_fallback_retry_stats: Dict = {}

        self.qfunction = None
        self.exploration_constant = cfg.exploration_constant

    def _log(self, *args, **kwargs) -> None:
        if bool(getattr(self, "verbose", False)):
            print(*args, **kwargs)

    def set_qfunction(
        self,
        model_cls: torch.nn.Module,
        model_cfg: OmegaConf,
        state_dict: dict = None,
        device: str = "cuda:0",
    ):
        self.qfunction = model_cls(model_cfg)
        if state_dict:
            self.qfunction.load_state_dict(state_dict)
        self.qfunction.set_device(device)
        self.qfunction.eval()

    def search(
        self,
        root: MCTS_Node,
        use_qfunction: bool = False,
        preserve_tree: bool = True,
        epsilon: float = 1e-1,
        eval: bool = False,
        multiple_nodes: bool = False,
        log_info: bool = False,
        execution_rejected_actions: Optional[List[Action]] = None,
        execution_rejected_stone_ids: Optional[List[int]] = None,
        seed_actions: Optional[List[Action]] = None,
        candidate_filter: Optional[CandidateFilter] = None,
        diversify_root_sampling: bool = False,
    ) -> Tuple[MCTS_Node, float]:
        configured_depth = int(self.cfg.max_depth)
        effective_depth = self._effective_max_depth(root, configured_depth)
        self._last_effective_max_depth = effective_depth
        if (
            bool(getattr(self, "verbose", False))
            and log_info
            and effective_depth != configured_depth
        ):
            placed = len(getattr(getattr(root, "state", None), "stone_seq", []) or [])
            print(
                "[INFO] Initial depth-1 MCTS phase: "
                f"placed={placed}, limit={int(self.cfg.initial_depth1_steps)}"
            )
        root_cfg = getattr(root, "cfg", None)
        depth_changed = effective_depth != configured_depth
        root_depth = (
            int(root_cfg.max_depth)
            if depth_changed and root_cfg is not None and root_cfg is not self.cfg
            else None
        )
        if depth_changed:
            self.cfg.max_depth = effective_depth
            if root_depth is not None:
                root_cfg.max_depth = effective_depth
        try:
            return self._search(
                root,
                use_qfunction,
                preserve_tree,
                epsilon,
                eval,
                multiple_nodes,
                log_info,
                execution_rejected_actions,
                execution_rejected_stone_ids,
                seed_actions,
                candidate_filter,
                diversify_root_sampling,
            )
        finally:
            if depth_changed:
                self.cfg.max_depth = configured_depth
                if root_depth is not None:
                    root_cfg.max_depth = root_depth

    def _effective_max_depth(
        self,
        root: MCTS_Node,
        configured_depth: Optional[int] = None,
    ) -> int:
        configured_depth = (
            int(self.cfg.max_depth)
            if configured_depth is None
            else int(configured_depth)
        )
        initial_steps = max(int(self.cfg.get("initial_depth1_steps", 0)), 0)
        state = getattr(root, "state", None)
        placed = len(getattr(state, "stone_seq", []) or []) if state is not None else 0
        if initial_steps > 0 and placed < initial_steps:
            return min(configured_depth, 1)
        return configured_depth

    def _search(
        self,
        root: MCTS_Node,
        use_qfunction: bool = False,
        preserve_tree: bool = True,
        epsilon: float = 1e-1,
        eval: bool = False,
        multiple_nodes: bool = False,
        log_info: bool = False,
        execution_rejected_actions: Optional[List[Action]] = None,
        execution_rejected_stone_ids: Optional[List[int]] = None,
        seed_actions: Optional[List[Action]] = None,
        candidate_filter: Optional[CandidateFilter] = None,
        diversify_root_sampling: bool = False,
    ) -> Tuple[MCTS_Node, float]:
        root.set_is_simulated(True)
        if not preserve_tree:
            root.parent = None

        reset()
        self._execution_rejected_actions = list(execution_rejected_actions or [])
        self._execution_rejected_stone_ids = {
            int(stone_id) for stone_id in (execution_rejected_stone_ids or [])
        }
        self._external_seed_actions = [
            action.copy() for action in (seed_actions or []) if action is not None
        ]
        self._final_validation_stats = None
        self._last_final_validation_debug_nodes = []
        self._last_root_planar_candidate_calls = []
        self._short_simulation_stats = self._empty_short_simulation_stats()
        self._root_long_validation_failures = []
        self._root_long_validation_stats = {
            "attempted": 0,
            "kept": 0,
            "rejected": 0,
            "value_seeded": 0,
            "reasons": {},
        }
        self._root_retry_rejected_actions = []
        self._mean_fallback_retry_stats = {
            "attempted": False,
            "retired_rejections": 0,
            "new_attempts": 0,
            "new_selectable": 0,
            "validated_path_found": False,
        }
        self._diversify_root_sampling = bool(diversify_root_sampling)
        self._no_candidate_retry_without_deadline = False
        self._search_deadline = self._make_search_deadline()
        timed_out = False
        _search_start = time.perf_counter()

        def run_search_iterations(iteration_count: int) -> bool:
            block_timed_out = False
            for _ in range(max(int(iteration_count), 0)):
                if self._search_timed_out():
                    block_timed_out = True
                    break
                coverage_child = self._next_root_lookahead_child(root)
                if coverage_child is not None:
                    self._run_root_lookahead_iteration(
                        coverage_child,
                        use_qfunction,
                        epsilon,
                        eval,
                    )
                    continue
                node = self._select(node=root, epsilon=epsilon, include_failed=eval)

                while not node.is_terminal():
                    if self._search_timed_out():
                        block_timed_out = True
                        break
                    if node.is_expandable() and len(node.expandable_children()) == 0:
                        self.expand_node(node, use_qfunction)
                        if len(node.expandable_children()) == 0:
                            break
                    parent = node
                    node = parent.select_child(
                        self.exploration_constant,
                        epsilon=epsilon,
                        include_failed=eval,
                    )
                    if not node.is_simulated and node.parent is not None:
                        node = self._simulate_selected_with_siblings(
                            parent,
                            node,
                            use_qfunction,
                        )
                        if (
                            len(node.parent.selectable_children()) == 0
                            and not node.parent.is_expandable()
                        ):
                            node.parent.q_value = -np.inf

                if block_timed_out:
                    break
                if node.is_expandable():
                    self.expand_node(node, use_qfunction)

                self._backpropagate(node)  # backpropagate even if failed
            return block_timed_out

        def retry_mean_fallback(nodes: List[MCTS_Node]) -> bool:
            nonlocal timed_out
            if not self._only_mean_fallback_candidates(nodes):
                return False
            retry = self._prepare_mean_fallback_retry(root)
            if not retry["attempted"]:
                return False
            self._log(
                "[INFO] No validated root continuation found; "
                "retrying retired root-attempt slots with diversified sampling."
            )
            self._no_candidate_retry_without_deadline = True
            self._search_deadline = self._make_retry_deadline()
            self._diversify_root_sampling = True
            try:
                timed_out = run_search_iterations(int(self.cfg.n_iter)) or timed_out
            finally:
                self._diversify_root_sampling = False
            return True

        timed_out = run_search_iterations(int(self.cfg.n_iter))

        _tree_search_elapsed = time.perf_counter() - _search_start
        if root.number_of_leaves() == 1 and not root.done:
            self._log("Only one leaf is expanded, return the best child")
            self._log(f"[INFO] Root child counts: {root.child_debug_counts()}")

        if not multiple_nodes:

            best, score = self._best_feasible_child(
                root,
                use_qfunction,
                candidate_filter=candidate_filter,
            )
            if retry_mean_fallback([best]):
                best, score = self._best_feasible_child(
                    root,
                    use_qfunction,
                    candidate_filter=candidate_filter,
                )
                self._finish_mean_fallback_retry(root, [best])
            if (
                best is root
                and self._final_validation_rejected_everything()
                and root.can_expand_children()
            ):
                self._no_candidate_retry_without_deadline = True
                self._search_deadline = self._make_retry_deadline()
            while (
                best is root
                and self._final_validation_rejected_everything()
                and root.can_expand_children()
            ):
                self._log(
                    "[INFO] MCTS final selection kept no feasible candidate; "
                    "continuing search with diversified root sampling."
                )
                before = (len(root.children), len(root.selectable_children()))
                self._diversify_root_sampling = True
                try:
                    timed_out = run_search_iterations(int(self.cfg.n_iter)) or timed_out
                finally:
                    self._diversify_root_sampling = False
                best, score = self._best_feasible_child(
                    root,
                    use_qfunction,
                    candidate_filter=candidate_filter,
                )
                after = (len(root.children), len(root.selectable_children()))
                if after == before:
                    break
            if (
                candidate_filter is not None
                and best is root
                and score == -np.inf
                and self._final_validation_rejected_everything()
            ):
                self._mark_no_feasible_root(root)
            self._log_search_summary(
                log_info,
                root,
                timed_out,
                _search_start,
                _tree_search_elapsed,
            )
            best.reset(preserve_tree, 0)
            best.parent = None
            return best, score

        nodes, scores = self._feasible_root_candidates(
            root,
            use_qfunction,
            candidate_filter=candidate_filter,
        )
        debug_nodes = list(getattr(self, "_last_final_validation_debug_nodes", []))
        if retry_mean_fallback(nodes):
            nodes, scores = self._feasible_root_candidates(
                root,
                use_qfunction,
                candidate_filter=candidate_filter,
            )
            self._finish_mean_fallback_retry(root, nodes)
            debug_nodes = list(
                getattr(self, "_last_final_validation_debug_nodes", [])
            )
        if not nodes and self._final_validation_rejected_everything():
            self._log(
                "[INFO] MCTS final validation kept no candidates; "
                "continuing with a bounded diversified retry."
            )
            self._no_candidate_retry_without_deadline = True
            self._search_deadline = self._make_retry_deadline()
            self._diversify_root_sampling = True
            try:
                timed_out = run_search_iterations(int(self.cfg.n_iter)) or timed_out
            finally:
                self._diversify_root_sampling = False
            nodes, scores = self._feasible_root_candidates(
                root,
                use_qfunction,
                candidate_filter=candidate_filter,
            )
            debug_nodes = list(getattr(self, "_last_final_validation_debug_nodes", []))
        if not debug_nodes:
            debug_nodes = list(nodes)

        for node in debug_nodes:
            node._debug_best_sequence = best_sequence_from_node(node, self.cfg)
        self._log_search_summary(
            log_info,
            root,
            timed_out,
            _search_start,
            _tree_search_elapsed,
        )

        seen = set()
        for node in list(nodes) + debug_nodes:
            if id(node) in seen:
                continue
            seen.add(id(node))
            node.reset(preserve_tree, 0)
            node.parent = None

        self._last_final_validation_debug_nodes = debug_nodes
        return nodes, scores

    def _log_search_summary(
        self,
        log_info: bool,
        root: MCTS_Node,
        timed_out: bool,
        search_start: float,
        tree_search_elapsed: float,
    ) -> None:
        width_stats = root.search_width_debug()
        width_stats["short_simulation"] = dict(
            getattr(self, "_short_simulation_stats", {})
        )
        width_stats["root_long_validation"] = dict(
            getattr(self, "_root_long_validation_stats", {})
        )
        width_stats["root_lookahead_coverage"] = self._root_lookahead_coverage_stats(
            root
        )
        width_stats["mean_fallback_retry"] = dict(
            getattr(self, "_mean_fallback_retry_stats", {})
        )
        width_stats["initial_search_timed_out"] = bool(timed_out)
        width_stats["no_candidate_retry_without_deadline"] = bool(
            getattr(self, "_no_candidate_retry_without_deadline", False)
        )
        self._last_search_width_stats = width_stats
        if not bool(getattr(self, "verbose", False)) or not log_info:
            return
        tree_counts = root.tree_debug_counts()
        if tree_counts is not None:
            print(
                "[INFO] Tree nodes: {nodes}, simulated: {simulated}, "
                "frontier leaves: {frontier_leaves}, max depth: {max_depth}".format(
                    **tree_counts
                )
            )
        self._log_search_width_summary(width_stats)
        self._log_short_simulation_summary(width_stats.get("short_simulation"))
        self._log_root_long_validation_summary(
            width_stats.get("root_long_validation")
        )
        self._log_root_lookahead_coverage_summary(
            width_stats.get("root_lookahead_coverage")
        )
        self._log_mean_fallback_retry_summary(
            width_stats.get("mean_fallback_retry")
        )
        if timed_out:
            if bool(width_stats["no_candidate_retry_without_deadline"]):
                print(
                    "[INFO] Initial MCTS search reached max_search_time; "
                    "the no-candidate retry continued without a deadline."
                )
            else:
                print("[INFO] MCTS search stopped by max_search_time budget.")
        total_elapsed = time.perf_counter() - search_start
        print(
            format_summary(
                "[Timing] MCTS total wall="
                f"{total_elapsed:.2f}s  tree_search={tree_search_elapsed:.2f}s"
            )
        )
        self._log_final_validation_summary()

    @staticmethod
    def _log_root_long_validation_summary(stats: Optional[Dict]) -> None:
        if not stats or int(stats.get("attempted", 0)) == 0:
            return
        print(
            "[INFO] Root long validation: "
            f"attempted={int(stats['attempted'])}, "
            f"kept={int(stats['kept'])}, rejected={int(stats['rejected'])}, "
            f"value_seeded={int(stats.get('value_seeded', 0))}"
        )
        reasons = stats.get("reasons", {}) or {}
        if reasons:
            ordered = sorted(reasons.items(), key=lambda item: item[1], reverse=True)
            print(
                "[INFO] Root long validation rejection reasons: "
                + ", ".join(f"{key}={value}" for key, value in ordered)
            )

    def _log_search_width_summary(self, width_stats: Optional[Dict]) -> None:
        if not width_stats:
            return
        parts = []
        for row in width_stats.get("by_depth", []):
            depth = int(row.get("depth", 0))
            reasons = self._format_unselectable_reasons(
                row.get("unselectable_reasons", {})
            )
            skipped = self._format_unselectable_reasons(
                row.get("skipped_child_reasons", {})
            )
            detail_parts = []
            if reasons:
                detail_parts.append(f"rejects={reasons}")
            if skipped:
                detail_parts.append(f"skipped={skipped}")
            detail_text = ", " + ", ".join(detail_parts) if detail_parts else ""
            parts.append(
                "d{depth}: nodes={nodes}, expanded={expanded}, "
                "children={selectable}/{children}, simulated={simulated}, "
                "avg={avg:.1f}, max={max_children}, "
                "limit={child_limit}/{attempt_limit}{detail_text}".format(
                    depth=depth,
                    nodes=int(row.get("nodes", 0)),
                    expanded=int(row.get("expanded_nodes", 0)),
                    selectable=int(row.get("selectable_child_edges", 0)),
                    children=int(row.get("child_edges", 0)),
                    simulated=int(row.get("simulated_child_edges", 0)),
                    avg=float(row.get("avg_children_per_expanded", 0.0)),
                    max_children=int(row.get("max_children", 0)),
                    child_limit=int(row.get("max_child_limit", 0)),
                    attempt_limit=int(row.get("max_attempt_limit", 0)),
                    detail_text=detail_text,
                )
            )
        if parts:
            print("[INFO] MCTS search width: " + " | ".join(parts))

    @staticmethod
    def _empty_short_simulation_stats() -> Dict[str, int]:
        return {
            "batches": 0,
            "jobs": 0,
            "parallel_batches": 0,
            "parallel_jobs": 0,
            "max_batch_size": 0,
        }

    def _record_short_simulation_batch(self, jobs: int, parallel: bool) -> None:
        jobs = max(int(jobs), 0)
        if jobs == 0:
            return
        stats = getattr(self, "_short_simulation_stats", None)
        if stats is None:
            stats = self._empty_short_simulation_stats()
            self._short_simulation_stats = stats
        stats["batches"] += 1
        stats["jobs"] += jobs
        stats["max_batch_size"] = max(stats["max_batch_size"], jobs)
        if parallel:
            stats["parallel_batches"] += 1
            stats["parallel_jobs"] += jobs

    @staticmethod
    def _log_short_simulation_summary(stats: Optional[Dict]) -> None:
        if not stats or int(stats.get("jobs", 0)) == 0:
            return
        print(
            "[INFO] MCTS short simulation: "
            f"batches={int(stats.get('batches', 0))}, "
            f"jobs={int(stats.get('jobs', 0))}, "
            f"parallel_batches={int(stats.get('parallel_batches', 0))}, "
            f"parallel_jobs={int(stats.get('parallel_jobs', 0))}, "
            f"max_batch_size={int(stats.get('max_batch_size', 0))}"
        )

    @staticmethod
    def _format_unselectable_reasons(reasons: Dict, limit: int = 3) -> str:
        if not reasons:
            return ""
        ordered = sorted(reasons.items(), key=lambda item: (-int(item[1]), item[0]))
        return ",".join(f"{key}={int(value)}" for key, value in ordered[:limit])

    def _make_search_deadline(self) -> Optional[float]:
        max_search_time = self.cfg.get("max_search_time", None)
        if max_search_time is None:
            return None
        max_search_time = float(max_search_time)
        if max_search_time <= 0.0:
            return None
        return time.monotonic() + max_search_time

    @staticmethod
    def _make_retry_deadline() -> None:
        """Let the bounded no-candidate retry exhaust its action attempts."""
        return None

    def _search_timed_out(self) -> bool:
        return (
            self._search_deadline is not None
            and time.monotonic() >= self._search_deadline
        )

    def step(
        self, state: State, action: Action, simulate: bool = True
    ) -> Tuple[State, Observation, bool, float, Dict]:
        self.env.update_from_state(state)
        state, obs, done, reward, info_r = self.env.step(action, simulate)

        return state, obs, done, reward, info_r

    def _best_feasible_child(
        self,
        root: MCTS_Node,
        use_qfunction: bool,
        candidate_filter: Optional[CandidateFilter] = None,
    ) -> Tuple[MCTS_Node, float]:
        nodes, scores = self._feasible_root_candidates(
            root,
            use_qfunction,
            max_candidates=1,
            candidate_filter=candidate_filter,
        )
        if nodes:
            return nodes[0], scores[0]
        return root, -np.inf

    def _feasible_root_candidates(
        self,
        root: MCTS_Node,
        use_qfunction: bool,
        max_candidates: Optional[int] = None,
        candidate_filter: Optional[CandidateFilter] = None,
    ) -> Tuple[List[MCTS_Node], List[float]]:
        candidates = self._ranked_root_children(root)
        validation_limit = self._final_validation_candidate_limit()
        if candidate_filter is not None and max_candidates == 1:
            validation_limit = None
        selected: List[Tuple[MCTS_Node, float]] = []
        deferred_mean_fallbacks: List[Tuple[MCTS_Node, float]] = []
        debug_nodes: List[MCTS_Node] = list(
            getattr(self, "_root_long_validation_failures", [])
        )
        batch_size = len(candidates) if validation_limit is None else validation_limit
        batch_size = max(int(batch_size), 1)
        next_debug_start = len(candidates)
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            validated_children = self._ensure_children_feasible(
                root,
                [child for child, _ in batch],
                use_qfunction,
            )
            batch_selected = []
            for offset, ((_, provisional_score), child) in enumerate(
                zip(batch, validated_children),
                start=1,
            ):
                attempted = start + offset
                self._annotate_final_validation_attempt(
                    child,
                    attempted,
                    provisional_score,
                )
                if candidate_filter is not None and not child.failed:
                    filtered = candidate_filter(root, child)
                    if filtered is not None:
                        child = filtered
                self._annotate_final_validation_attempt(
                    child,
                    attempted,
                    provisional_score,
                )
                if child.failed:
                    self._mark_final_validation_failure(
                        child,
                        self._node_failure_reason(child),
                    )
                debug_nodes.append(child)
                if child.failed:
                    continue

                # Root validation can change the settled parent state.
                selected_score = self._root_child_selection_score(child)
                if not np.isfinite(selected_score):
                    reason = self._node_failure_reason(child)
                    if reason == "unknown":
                        reason = "nonfinite_selection_score"
                    self._mark_final_validation_failure(
                        child,
                        reason,
                    )
                    continue
                batch_selected.append((child, float(selected_score)))

            if not batch_selected:
                continue

            if not self._conditional_descendant_fallback_active(
                batch_selected[0][0]
            ):
                selected.extend(batch_selected)
                next_debug_start = start + batch_size
                break

            validated = []
            fallback = []
            for item in batch_selected:
                has_validated_path = bool(
                    (item[0].info or {}).get(
                        "validated_descendant_path_available",
                        False,
                    )
                )
                if has_validated_path:
                    validated.append(item)
                else:
                    fallback.append(item)
            deferred_mean_fallbacks.extend(fallback)
            if validated:
                selected.extend(validated)
                for child, _ in deferred_mean_fallbacks:
                    if child.info is None:
                        child.info = {}
                    child.info[
                        "final_selection_exclusion"
                    ] = "validated_descendant_preferred"
                next_debug_start = start + batch_size
                break

        if not selected and deferred_mean_fallbacks:
            selected = deferred_mean_fallbacks

        selected.sort(key=lambda item: item[1], reverse=True)
        feasible_nodes = [node for node, _ in selected]
        if selected and max_candidates is None:
            debug_nodes.extend(
                self._extra_final_validation_failures(
                    root,
                    candidates[next_debug_start:],
                    use_qfunction,
                    candidate_filter,
                    start_rank=next_debug_start + 1,
                )
            )
        self._record_final_validation_stats(debug_nodes, feasible_nodes)
        self._last_final_validation_debug_nodes = debug_nodes
        if max_candidates is not None:
            selected = selected[:max_candidates]
        for child in root.children:
            if child.info is not None:
                child.info["final_selection_best"] = False
        for rank, (child, score) in enumerate(selected, start=1):
            if child.info is None:
                child.info = {}
            child.info["final_selection_rank"] = int(rank)
            child.info["final_selection_score"] = float(score)
            child.info["final_selection_best"] = bool(rank == 1)
        selected_nodes = [node for node, _ in selected]
        selected_scores = [score for _, score in selected]
        return selected_nodes, selected_scores

    def _extra_final_validation_failures(
        self,
        root: MCTS_Node,
        candidates: List[Tuple[MCTS_Node, float]],
        use_qfunction: bool,
        candidate_filter: Optional[CandidateFilter],
        start_rank: int,
    ) -> List[MCTS_Node]:
        extra_limit = mcts_validation_config(self.cfg).debug_extra_candidates
        if extra_limit <= 0 or not candidates:
            return []

        extra_nodes: List[MCTS_Node] = []
        for offset, (child, provisional_score) in enumerate(candidates[:extra_limit]):
            child = self._ensure_child_feasible(root, child, use_qfunction)
            attempted = int(start_rank + offset)
            self._annotate_final_validation_attempt(
                child,
                attempted,
                provisional_score,
            )
            if candidate_filter is not None and not child.failed:
                filtered = candidate_filter(root, child)
                if filtered is not None:
                    child = filtered
                    self._annotate_final_validation_attempt(
                        child,
                        attempted,
                        provisional_score,
                    )
            if child.failed:
                self._mark_final_validation_failure(
                    child,
                    self._node_failure_reason(child),
                )
                if child.info is None:
                    child.info = {}
                child.info["debug_extra_failure"] = True
                extra_nodes.append(child)
                continue

            selected_score = self._root_child_selection_score(child)
            if not np.isfinite(selected_score):
                reason = self._node_failure_reason(child)
                if reason == "unknown":
                    reason = "nonfinite_selection_score"
                self._mark_final_validation_failure(child, reason)
                if child.info is None:
                    child.info = {}
                child.info["debug_extra_failure"] = True
                extra_nodes.append(child)

        return extra_nodes

    def _record_final_validation_stats(
        self,
        debug_nodes: List[MCTS_Node],
        selected_nodes: List[MCTS_Node],
    ) -> None:
        reasons: Dict[str, int] = {}
        for node in debug_nodes:
            info = getattr(node, "info", None) or {}
            if bool(info.get("final_validation_failed", False)):
                reason = str(info.get("final_validation_failure", "unknown"))
            elif info.get("final_selection_exclusion"):
                reason = str(info["final_selection_exclusion"])
            else:
                continue
            reasons[reason] = reasons.get(reason, 0) + 1
        self._final_validation_stats = {
            "attempted": int(len(debug_nodes)),
            "kept": int(len(selected_nodes)),
            "rejected": int(max(len(debug_nodes) - len(selected_nodes), 0)),
            "reasons": reasons,
        }

    def _log_final_validation_summary(self) -> None:
        stats = getattr(self, "_final_validation_stats", None)
        if not stats or int(stats.get("attempted", 0)) == 0:
            return
        print(
            "[INFO] Final validation: "
            f"attempted={stats['attempted']}, kept={stats['kept']}, "
            f"rejected={stats['rejected']}"
        )
        reasons = stats.get("reasons", {}) or {}
        if reasons:
            ordered = sorted(reasons.items(), key=lambda item: item[1], reverse=True)
            print(
                "[INFO] Final validation rejection reasons: "
                + ", ".join(f"{key}={value}" for key, value in ordered)
            )

    def _final_validation_rejected_everything(self) -> bool:
        stats = getattr(self, "_final_validation_stats", None)
        if not stats:
            return False
        return (
            int(stats.get("attempted", 0)) > 0
            and int(stats.get("kept", 0)) == 0
            and int(stats.get("rejected", 0)) > 0
        )

    def _annotate_final_validation_attempt(
        self,
        child: MCTS_Node,
        rank: int,
        prior_score: float,
    ) -> None:
        if child.info is None:
            child.info = {}
        child.info["final_validation_rank"] = int(rank)
        child.info["final_selection_mode"] = self._final_selection_mode()
        child.info["final_validation_rank_score"] = float(prior_score)
        child.info["final_validation_prior_score"] = float(prior_score)
        child.info.pop("final_selection_exclusion", None)

    def _mark_final_validation_failure(
        self,
        node: MCTS_Node,
        reason: str,
    ) -> None:
        if node.info is None:
            node.info = {}
        if not node.info.get("final_validation_failed", False):
            node.info["final_validation_failure"] = str(reason)
        node.info["final_validation_failed"] = True

    def _mark_no_feasible_root(self, root: MCTS_Node) -> None:
        root.set_failed(True)
        root.q_value = -np.inf
        info = dict(root.info or {})
        info.update(
            {
                "search_failed": True,
                "search_failure_reason": "no_feasible_root_candidate",
            }
        )
        root.info = info

    def _node_failure_reason(self, node: MCTS_Node) -> str:
        info = getattr(node, "info", None) or {}
        for key in (
            "final_validation_failure",
            "mcts_unselectable_reason",
            "grasp_failure_reason",
            "posegen_prune_reason",
        ):
            reason = info.get(key, None)
            if reason:
                return str(reason)
        state = getattr(node, "state", None)
        if bool(getattr(state, "failed", False)):
            return "simulation_failure_unattributed"
        return "unknown"

    def _final_validation_candidate_limit(self) -> Optional[int]:
        return mcts_validation_config(self.cfg).max_candidates

    def _ranked_root_children(self, root: MCTS_Node) -> List[Tuple[MCTS_Node, float]]:
        ranked = []
        for child in root.selectable_children():
            score = self._provisional_root_child_selection_score(child)
            if np.isfinite(score):
                ranked.append((child, float(score)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _final_selection_mode(self) -> str:
        selection = self.cfg.get("final_selection", {})
        mode = str(selection.get("mode", "mean")).lower().replace("-", "_")
        if mode in {"max", "max_descendant", "descendant_max"}:
            return "max_descendant"
        if mode in {"mix", "mixed", "max_mean_mix"}:
            return "mixed"
        return "mean"

    def _final_selection_max_weight(self) -> float:
        selection = self.cfg.get("final_selection", {})
        if self._final_selection_mode() == "max_descendant":
            default = 1.0
        elif self._final_selection_mode() == "mixed":
            default = 0.5
        else:
            default = 0.0
        return float(np.clip(selection.get("max_weight", default), 0.0, 1.0))

    def _final_selection_validate_descendant_path(self) -> bool:
        selection = self.cfg.get("final_selection", {})
        return bool(selection.get("validate_descendant_path", False))

    def _final_selection_require_validated_descendant(self) -> bool:
        selection = self.cfg.get("final_selection", {})
        return bool(selection.get("require_validated_descendant", False))

    def _conditional_descendant_fallback_active(self, child: MCTS_Node) -> bool:
        return bool(
            self._final_selection_max_weight() > 0.0
            and self._final_selection_validate_descendant_path()
            and not self._final_selection_require_validated_descendant()
            and self._final_selection_validate_descendant_depth(child) > 0
        )

    def _final_selection_validate_descendant_depth(
        self,
        child: Optional[MCTS_Node] = None,
    ) -> int:
        selection = self.cfg.get("final_selection", {})
        requested = max(int(selection.get("validate_descendant_depth", 1)), 0)
        if child is None:
            return requested
        available = max(int(self.cfg.max_depth) - int(child.depth), 0)
        return min(requested, available)

    def _provisional_root_child_selection_score(self, child: MCTS_Node) -> float:
        """Rank roots without triggering long simulations."""
        return self._combine_root_child_scores(
            child,
            self._max_descendant_path_return(child),
        )

    def _root_child_selection_score(self, child: MCTS_Node) -> float:
        if self._final_selection_max_weight() <= 0.0:
            self._record_root_selection_diagnostics(
                child,
                validation_depth=0,
                path_score=-np.inf,
                used_mean_fallback=False,
            )
            return self._mean_child_selection_score(child)
        validation_depth = self._final_selection_validate_descendant_depth(child)
        if self._final_selection_validate_descendant_path():
            path_score = self._validated_descendant_path_return(
                child,
                validation_depth,
            )
        else:
            path_score = self._max_descendant_path_return(child)
        if child.info is None:
            child.info = {}
        used_mean_fallback = bool(
            self._final_selection_validate_descendant_path()
            and validation_depth > 0
            and not np.isfinite(path_score)
        )
        child.info["validated_descendant_required"] = bool(
            self._final_selection_require_validated_descendant()
            and self._final_selection_validate_descendant_path()
            and validation_depth > 0
        )
        child.info["validated_descendant_path_available"] = bool(
            np.isfinite(path_score)
        )
        if np.isfinite(path_score):
            child.info["validated_descendant_path_return"] = float(path_score)
        else:
            child.info.pop("validated_descendant_path_return", None)
        self._record_root_selection_diagnostics(
            child,
            validation_depth=validation_depth,
            path_score=path_score,
            used_mean_fallback=used_mean_fallback,
        )
        if child.info["validated_descendant_required"] and not np.isfinite(path_score):
            self._mark_node_unselectable(child, "missing_validated_descendant")
            return -np.inf
        return self._combine_root_child_scores(
            child,
            path_score,
        )

    def _combine_root_child_scores(
        self,
        child: MCTS_Node,
        path_score: float,
    ) -> float:
        mean_score = self._mean_child_selection_score(child)
        max_weight = self._final_selection_max_weight()
        if max_weight <= 0.0:
            return mean_score
        if not np.isfinite(path_score):
            return mean_score
        max_score = path_score + float(child.q_value_init)
        if max_weight >= 1.0:
            return max_score
        return max_weight * max_score + (1.0 - max_weight) * mean_score

    @staticmethod
    def _mean_child_selection_score(child: MCTS_Node) -> float:
        if child.is_simulated:
            return child.q_value / max(child.visits, 1.0) + child.q_value_init
        return child.q_value_init

    def _max_descendant_path_return(self, node: MCTS_Node) -> float:
        """Best known discounted return from this node through its descendants."""
        if node.failed:
            return -np.inf

        reward = 0.0 if node.reward is None else float(node.reward)
        if not np.isfinite(reward):
            reward = 0.0
        if node.done:
            return reward

        child_returns = [
            self._max_descendant_path_return(child)
            for child in node.simulated_children(include_failed=False)
        ]
        child_returns = [value for value in child_returns if np.isfinite(value)]
        if child_returns:
            tail = max(child_returns)
        else:
            tail = float(node.value_init)
            if not np.isfinite(tail):
                tail = node.get_estimated_reward_to_go()
        if not np.isfinite(tail):
            tail = 0.0
        return reward + float(self.cfg.reward.discount) * tail

    def _validated_descendant_path_return(
        self,
        node: MCTS_Node,
        remaining_depth: int,
    ) -> float:
        """Best known path, requiring the next promoted descendants to validate."""
        if node.failed:
            return -np.inf

        reward = 0.0 if node.reward is None else float(node.reward)
        if not np.isfinite(reward):
            reward = 0.0
        if node.done:
            return reward
        if remaining_depth <= 0:
            return reward

        # A child rejected against an older parent state may become feasible
        # after authoritative validation changes that parent state.
        children = node.simulated_children(include_failed=True)
        if not children:
            return -np.inf

        ranked = sorted(
            children,
            key=lambda child: self._max_descendant_path_return(child),
            reverse=True,
        )
        for child in ranked:
            if not self._ensure_descendant_final_valid(node, child):
                continue
            child_return = self._validated_descendant_path_return(
                child,
                remaining_depth - 1,
            )
            if np.isfinite(child_return):
                return reward + float(self.cfg.reward.discount) * child_return
        return -np.inf

    def _ensure_descendant_final_valid(
        self,
        parent: MCTS_Node,
        child: MCTS_Node,
    ) -> bool:
        if parent.state is None or child.action is None:
            return False

        validation_key = self._descendant_validation_key(parent.state, child.action)
        info = child.info or {}
        if info.get("descendant_path_validation_key") == validation_key:
            return bool(info.get("descendant_path_final_validated", False))

        self._simulate_long_candidate(parent.state, child, use_qfunction=False)
        valid = not (
            child.failed
            or child.state is None
            or bool(getattr(child.state, "failed", False))
        )
        validation_key = self._descendant_validation_key(parent.state, child.action)
        self._record_descendant_validation(child, validation_key, valid)
        if valid and not np.isfinite(child.q_value):
            # A failed validation against an older parent may have set -inf.
            child.q_value = 0.0
            child.visits = 0
        return valid

    def _ensure_descendants_final_valid(
        self,
        parent: MCTS_Node,
        children: List[MCTS_Node],
        use_qfunction: bool,
    ) -> Dict[MCTS_Node, Tuple[bool, bool]]:
        outcomes: Dict[MCTS_Node, Tuple[bool, bool]] = {}
        jobs = []
        for child in children:
            if parent.state is None or child.action is None:
                outcomes[child] = (False, False)
                continue
            validation_key = self._descendant_validation_key(
                parent.state,
                child.action,
            )
            info = child.info or {}
            if info.get("descendant_path_validation_key") == validation_key:
                outcomes[child] = (
                    bool(info.get("descendant_path_final_validated", False)),
                    False,
                )
                continue
            jobs.append((child, validation_key))

        n_threads = self._simulation_threads(len(jobs))
        if n_threads <= 1:
            for child, _ in jobs:
                outcomes[child] = (
                    self._ensure_descendant_final_valid(parent, child),
                    True,
                )
            return outcomes

        with phase("mcts_final_validate"):
            results = list(
                self._simulation_executor_instance().map(
                    lambda item: self._run_long_candidate_simulation(
                        self._thread_env(),
                        parent.state,
                        item[0].action,
                    ),
                    jobs,
                )
            )
        for (child, _), result in zip(jobs, results):
            self._apply_long_candidate_result(child, result, use_qfunction)
            valid = not (
                child.failed
                or child.state is None
                or bool(getattr(child.state, "failed", False))
            )
            validation_key = self._descendant_validation_key(
                parent.state,
                child.action,
            )
            self._record_descendant_validation(child, validation_key, valid)
            outcomes[child] = (valid, True)
        return outcomes

    @staticmethod
    def _record_descendant_validation(
        child: MCTS_Node,
        validation_key: tuple,
        valid: bool,
    ) -> None:
        if child.info is None:
            child.info = {}
        child.info["descendant_path_validation_key"] = validation_key
        child.info["descendant_path_final_validated"] = bool(valid)

    @staticmethod
    def _descendant_validation_key(parent_state: State, action: Action) -> tuple:
        return (
            MonteCarloTreeSearch._validation_state_key(parent_state),
            MonteCarloTreeSearch._validation_action_key(action),
        )

    @staticmethod
    def _validation_state_key(state: State) -> tuple:
        poses = tuple(
            (
                int(stone_id),
                tuple(
                    float(value) for value in np.round(np.asarray(pose, dtype=float), 7)
                ),
            )
            for stone_id, pose in sorted(state.stone_poses.items())
        )
        return (
            tuple(int(idx) for idx in state.stone_seq),
            poses,
        )

    @staticmethod
    def _validation_action_key(action: Action) -> tuple:
        return (
            int(action.stone_idx),
            int(action.stone_id),
            tuple(
                float(value)
                for value in np.round(np.asarray(action.pose, dtype=float), 7)
            ),
            tuple(
                float(value)
                for value in np.round(np.asarray(action.init_pose, dtype=float), 7)
            ),
        )

    def _record_root_long_validation(
        self,
        root: MCTS_Node,
        child: MCTS_Node,
    ) -> None:
        if child.info is None:
            child.info = {}
        child.info["root_long_validation_parent_key"] = self._validation_state_key(
            root.state
        )
        child.info["root_long_validation_action_key"] = self._validation_action_key(
            child.action
        )
        child.info["root_long_validated"] = bool(
            child.is_simulated and not child.failed and child.state is not None
        )

    def _root_long_validation_is_current(
        self,
        root: MCTS_Node,
        child: MCTS_Node,
    ) -> bool:
        info = child.info or {}
        if not bool(info.get("root_long_validated", False)):
            return False
        if child.failed or child.state is None or child.action is None:
            return False
        return (
            info.get("root_long_validation_parent_key")
            == self._validation_state_key(root.state)
            and info.get("root_long_validation_action_key")
            == self._validation_action_key(child.action)
        )

    def _ensure_child_feasible(
        self,
        root: MCTS_Node,
        child: MCTS_Node,
        use_qfunction: bool,
    ) -> MCTS_Node:
        if self._root_long_validation_is_current(root, child):
            return child
        child = self._simulate_long_candidate(root.state, child, use_qfunction)
        self._record_root_long_validation(root, child)
        if child.failed:
            child.q_value = -np.inf
        return child

    def _ensure_children_feasible(
        self,
        root: MCTS_Node,
        children: List[MCTS_Node],
        use_qfunction: bool,
    ) -> List[MCTS_Node]:
        validated = list(children)
        jobs = [
            (index, child)
            for index, child in enumerate(children)
            if not self._root_long_validation_is_current(root, child)
        ]

        if not jobs:
            return validated

        if self._simulation_threads(len(jobs)) <= 1:
            for index, child in jobs:
                validated[index] = self._ensure_child_feasible(
                    root,
                    child,
                    use_qfunction,
                )
            return validated

        with phase("mcts_final_validate"):
            results = list(
                self._simulation_executor_instance().map(
                    lambda item: self._run_long_candidate_simulation(
                        self._thread_env(),
                        root.state,
                        item[1].action,
                    ),
                    jobs,
                )
            )
        for (index, child), result in zip(jobs, results):
            validated[index] = self._apply_long_candidate_result(
                child,
                result,
                use_qfunction,
            )
            self._record_root_long_validation(root, validated[index])
        return validated

    def _simulation_threads(self, n_jobs: int) -> int:
        if n_jobs <= 1:
            return 1
        try:
            configured = mcts_validation_config(self.cfg).worker_threads(self.env_args)
        except Exception:
            configured = 1
        return max(min(int(configured), int(n_jobs)), 1)

    def revalidate_preserved_root(
        self,
        root: MCTS_Node,
        use_qfunction: bool = False,
        max_child_drift: Optional[float] = None,
    ) -> Dict[str, int]:
        """Re-run authoritative final validation on a preserved root's children.

        When the MCTS tree is preserved across steps, the promoted depth-1
        children of the new root were previously depth-2 nodes simulated only
        with the lighter tree-search settings -- they never saw the
        final-validation sim that ordinary root children get. This re-simulates
        each against the (already validated) new root state with the validation
        config. Children that now fail validation are marked unselectable and
        their stale subtree is dropped; children whose settled position drifts
        beyond ``max_child_drift`` metres keep their (now inconsistent) subtree
        dropped and are reset to fresh expandable leaves. Call this on the new
        root before running ``search(..., preserve_tree=True)``.
        """
        stats = {
            "revalidated": 0,
            "failed": 0,
            "pruned": 0,
            "kept": 0,
            "removed": 0,
            "pre_failed": 0,
        }
        reason_counts = Counter()
        if not root.children:
            root.reset_search_statistics()
            stats["reasons"] = {}
            return stats
        self.env.update_from_state(root.state)

        ranked_children = [child for child, _ in self._ranked_root_children(root)]
        preserve_limit = max(int(self.cfg.max_children_num), 1)
        retained_set = set(ranked_children[:preserve_limit])
        retained_children = [child for child in root.children if child in retained_set]
        for child in root.children:
            if child in retained_set:
                continue
            child.children = []
            stats["removed"] += 1
            if child in ranked_children:
                reason_counts["preserve_limit"] += 1
            else:
                stats["pre_failed"] += 1
                reason_counts[self._node_failure_reason(child)] += 1

        simulated_children = [
            child
            for child in retained_children
            if child.action is not None
            and child.state is not None
            and child.is_simulated
        ]
        original_poses = {
            child: settled_action_pose(self.env, child.action, child.state)
            for child in simulated_children
        }
        outcomes = self._ensure_descendants_final_valid(
            root,
            simulated_children,
            use_qfunction,
        )
        validated_children: List[MCTS_Node] = []
        for child in retained_children:
            if child not in outcomes:
                validated_children.append(child)
                continue

            valid, simulated = outcomes[child]
            if simulated:
                stats["revalidated"] += 1
            if not valid:
                child.children = []  # subtree was built on an invalid parent
                reason = "pre_failed" if not simulated else "unknown"
                if child.info is not None:
                    reason = str(child.info.get("final_validation_failure", reason))
                if child.state is not None and child.state.failed:
                    reason = self._node_failure_reason(child)
                reason_counts[reason] += 1
                stats["failed" if simulated else "pre_failed"] += 1
                stats["removed"] += 1
                continue
            self._record_root_long_validation(root, child)
            if simulated and max_child_drift is not None and child.children:
                new_pose = settled_action_pose(self.env, child.action, child.state)
                drift = float(np.linalg.norm(new_pose[:3] - original_poses[child][:3]))
                if drift > float(max_child_drift):
                    child.children = []
                    if child.info is not None:
                        child.info.pop("root_lookahead_attempted", None)
                    child._action_generation_exhausted = False
                    reason_counts["drift_pruned"] += 1
                    stats["pruned"] += 1
                    validated_children.append(child)
                    continue
            stats["kept"] += 1
            validated_children.append(child)

        if stats["removed"] > 0 or len(validated_children) != len(root.children):
            root.children = validated_children
            root._action_generation_exhausted = False

        if not root.done and len(root.selectable_children()) == 0:
            # All promoted children failed authoritative validation. Reopen the
            # root so the next search call can run CEM from the current state
            # instead of treating the preserved subtree as exhausted.
            root.children = []
            root.visits = 0
            root.q_value = 0.0
            root._failed = False
            root._action_generation_exhausted = False
        elif not np.isfinite(root.q_value):
            # A stale -inf root value can prevent later expansion even when
            # some revalidated children remain.
            root.visits = 0
            root.q_value = 0.0
        root.reset_search_statistics()
        stats["value_seeded"] = self._seed_root_long_values(
            root,
            root.selectable_children(),
        )
        stats["reasons"] = dict(reason_counts)
        return stats

    @timed("mcts_expand_node")
    def expand_node(self, node: MCTS_Node, use_qfunction: bool = False):
        self.env.update_from_state(node.state)
        if node.depth == 0:
            first_new_child = len(node.children)
            self._expand_root_with_cem(node, use_qfunction)
            self._long_validate_new_root_children(
                node,
                node.children[first_new_child:],
                use_qfunction,
            )
            return

        scene_height_map = node.obs.scene_height_map if node.obs is not None else None
        action_cfg = mcts_action_generation_config(self.cfg)
        target_new_selectable = min(
            action_cfg.min_feasible_children,
            max(node.child_expansion_room(), 1),
        )
        start_selectable = len(node.selectable_children())
        sampling_priority = self._sampling_priority_for_expansion(node)

        for _ in range(action_cfg.expansion_retry_batches):
            if node.child_expansion_room() <= 0:
                break

            n_action_samples = self._num_action_samples_for_expansion(node)
            if n_action_samples <= 0:
                break

            action_samples, action_mask = self.env.get_action_samples(
                node.state,
                scene_height_map=scene_height_map,
                n_action_samples=n_action_samples,
                sampling_priority=sampling_priority,
            )
            if len(action_mask) == 0:
                node.set_action_generation_exhausted(True)
                break

            actions = list(action_samples)
            valid_mask = np.isfinite(action_mask)
            skipped_reasons = Counter()
            for action, valid in zip(actions, valid_mask):
                if bool(valid):
                    continue
                reason = self._skipped_action_reason(action)
                skipped_reasons[reason] += 1
            for reason, count in skipped_reasons.items():
                node.record_skipped_children(reason, count)
            if not np.any(valid_mask):
                if all(int(getattr(action, "stone_idx", -1)) < 0 for action in actions):
                    node.set_action_generation_exhausted(True)
                    break
                continue

            actions = [
                action for action, valid in zip(actions, valid_mask) if bool(valid)
            ]
            q_values = self._score_actions(
                node,
                actions,
                np.ones(len(actions), dtype=bool),
                use_qfunction,
            )
            node.value_init = node.get_estimated_reward_to_go()
            self._append_children(node, actions, q_values)

            new_selectable = len(node.selectable_children()) - start_selectable
            if new_selectable >= target_new_selectable:
                break

    def _long_validate_new_root_children(
        self,
        root: MCTS_Node,
        children: List[MCTS_Node],
        use_qfunction: bool,
    ) -> None:
        candidates = [
            child
            for child in children
            if child.action is not None and not child.is_simulated and not child.failed
        ]
        if not candidates:
            return

        validated = self._ensure_children_feasible(
            root,
            candidates,
            use_qfunction,
        )
        stats = self._root_long_validation_stats
        stats["attempted"] = int(stats.get("attempted", 0)) + len(validated)
        reasons = Counter(stats.get("reasons", {}))
        for child in validated:
            if child.failed:
                stats["rejected"] = int(stats.get("rejected", 0)) + 1
                reasons[self._node_failure_reason(child)] += 1
                if child not in self._root_long_validation_failures:
                    self._root_long_validation_failures.append(child)
            else:
                stats["kept"] = int(stats.get("kept", 0)) + 1
        stats["value_seeded"] = int(stats.get("value_seeded", 0)) + int(
            self._seed_root_long_values(root, validated)
        )
        stats["reasons"] = dict(reasons)

    @staticmethod
    def _skipped_action_reason(action: Action) -> str:
        if action is None or int(getattr(action, "stone_idx", -1)) < 0:
            return "dummy_padding"
        diagnostics = getattr(action, "diagnostics", {}) or {}
        reasons = diagnostics.get("action_mask_reasons", None)
        if reasons:
            return str(reasons[0])
        reason = diagnostics.get("action_mask_reason", None)
        if reason:
            return str(reason)
        if diagnostics.get("posegen_gap_ok", True) is False:
            return "posegen_gap"
        if diagnostics.get("final_pose_ok", True) is False:
            return "final_pose"
        if diagnostics.get("scan_orientation_ok", True) is False:
            return "scan_orientation"
        return "action_mask"

    def _append_children(
        self,
        node: MCTS_Node,
        actions: List[Action],
        q_values: np.ndarray,
    ) -> None:
        for action, q_value in zip(actions, q_values):
            posegen_failed = self._posegen_prune_failed(action)
            posegen_gap_failed = self._posegen_gap_prune_failed(action)
            unselectable_reason = self._action_unselectable_reason(
                action,
                q_value,
                posegen_failed=posegen_failed,
                posegen_gap_failed=posegen_gap_failed,
            )
            child = MCTS_Node(
                self.cfg,
                action=action,
                q_value_init=(
                    -np.inf if posegen_failed or posegen_gap_failed else q_value
                ),
                parent=node,
                depth=node.depth + 1,
            )
            if posegen_failed or posegen_gap_failed:
                child._failed = True
                child.q_value = -np.inf
                child.reward = -1.0
                child.info = {
                    "posegen_pruned": True,
                    "c_feq": float(action.c_feq),
                    "c_gap": float(action.c_gap),
                    "posegen_thresh": self._posegen_prune_threshold(),
                    "posegen_gap_threshold": self._posegen_gap_threshold(),
                    "posegen_prune_reason": (
                        "c_gap" if posegen_gap_failed else "c_feq"
                    ),
                    "mcts_unselectable_reason": unselectable_reason,
                }
                child.set_is_simulated(True)
            elif unselectable_reason is not None:
                child.info = {
                    "mcts_unselectable_reason": unselectable_reason,
                }
            node.children.append(child)
        if len(actions) == 0:
            node.set_action_generation_exhausted(True)

    def _action_unselectable_reason(
        self,
        action: Action,
        q_value: float,
        posegen_failed: bool,
        posegen_gap_failed: bool,
    ) -> Optional[str]:
        if posegen_gap_failed:
            return "posegen_gap"
        if posegen_failed:
            return "posegen_c_feq"
        if action is None:
            return "missing_action"
        if int(getattr(action, "stone_idx", -1)) < 0:
            return "invalid_stone_idx"
        diagnostics = getattr(action, "diagnostics", {}) or {}
        reason = diagnostics.get("mcts_rejection_reason", None)
        if reason:
            return str(reason)
        if not np.isfinite(q_value):
            return "nonfinite_prior"
        return None

    def _posegen_prune_failed(self, action: Action) -> bool:
        sampling = mcts_sampling_config(self.cfg)
        if not sampling.posegen_prune_enabled:
            return False
        threshold = self._posegen_prune_threshold()
        if threshold is None:
            return False
        c_feq = float(getattr(action, "c_feq", np.nan))
        return np.isfinite(c_feq) and c_feq >= threshold

    def _posegen_gap_prune_failed(self, action: Action) -> bool:
        threshold = self._posegen_gap_threshold()
        if threshold is None:
            return False
        c_gap = float(getattr(action, "c_gap", np.nan))
        return not np.isfinite(c_gap) or c_gap > threshold

    def _posegen_gap_threshold(self) -> Optional[float]:
        value = self.env.cfg.action.get("posegen_gap_threshold", 0.001)
        return None if value is None else float(value)

    def _posegen_prune_threshold(self) -> Optional[float]:
        return mcts_sampling_config(self.cfg).posegen_prune_threshold(self.env)

    def _num_action_samples_for_expansion(self, node: MCTS_Node) -> Optional[int]:
        room = node.child_expansion_room()
        if room <= 0:
            return 0
        batch_size = mcts_action_generation_config(self.cfg).batch_size
        return min(batch_size, room)

    def _sampling_priority_for_expansion(self, node: MCTS_Node) -> Optional[str]:
        sampling = mcts_sampling_config(self.cfg)
        priority = sampling.deeper_sampling_priority
        start_depth = sampling.priority_child_depth
        child_depth = node.depth + 1
        if priority and child_depth >= start_depth:
            return priority
        return None

    def _sampling_priority_for_root(self, node: MCTS_Node) -> Optional[str]:
        if bool(getattr(self, "_diversify_root_sampling", False)):
            return None
        return mcts_action_generation_config(self.cfg).root_sampling_priority_for_state(
            node.state
        )

    @timed("mcts_score_actions")
    def _score_actions(
        self,
        node: MCTS_Node,
        actions: List[Action],
        valid_mask: np.ndarray,
        use_qfunction: bool,
    ) -> np.ndarray:
        scores = np.full(len(actions), -np.inf, dtype=float)
        valid_mask = np.asarray(valid_mask, dtype=bool).copy()
        for idx, action in enumerate(actions):
            diagnostics = getattr(action, "diagnostics", None)
            if diagnostics is not None:
                diagnostics.pop("mcts_rejection_reason", None)
            if not valid_mask[idx]:
                self._mark_action_rejected(action, "action_mask")
            if valid_mask[idx] and action.stone_idx < 0:
                self._mark_action_rejected(action, "invalid_stone_idx")
                valid_mask[idx] = False
            if valid_mask[idx] and self._posegen_gap_prune_failed(action):
                self._mark_action_rejected(action, "posegen_gap")
                valid_mask[idx] = False
            if valid_mask[idx] and self._execution_action_rejected(action):
                self._mark_action_rejected(action, "execution_rejected")
                valid_mask[idx] = False
        if len(actions) == 0 or not np.any(valid_mask):
            return scores
        if use_qfunction and (self.qfunction is not None):
            obs_dict = Observation.to_dict(node.obs)
            poses = np.vstack([a.pose for a in actions])
            stone_idxs = np.array([a.stone_idx for a in actions], dtype=int)
            with torch.no_grad():
                q_out = self.qfunction(
                    {
                        "obs": obs_dict,
                        "action": {"pose": poses, "stone_idx": stone_idxs},
                    }
                )
            q_arr = q_out.cpu().numpy() if torch.is_tensor(q_out) else np.asarray(q_out)
            scores[valid_mask] = q_arr.reshape(-1)[valid_mask]
            self._mark_nonfinite_scores(actions, valid_mask, scores, "qfunction")
            return scores
        scores[valid_mask] = self._heuristic_action_priors(
            node,
            actions,
            valid_mask,
        )[valid_mask]
        self._mark_nonfinite_scores(actions, valid_mask, scores, "prior")
        return scores

    def _mark_action_rejected(self, action: Action, reason: str) -> None:
        if action is None:
            return
        diagnostics = getattr(action, "diagnostics", None)
        if diagnostics is None:
            return
        diagnostics["mcts_rejection_reason"] = str(reason)

    def _execution_action_rejected(self, action: Action) -> bool:
        cfg = mcts_action_generation_config(self.cfg).execution_rejection
        if not bool(cfg.get("enabled", False)) or not bool(
            cfg.get("hard_filter", False)
        ):
            return False
        if int(action.stone_id) in getattr(
            self,
            "_execution_rejected_stone_ids",
            set(),
        ):
            return True

        xy_radius = max(float(cfg.get("xy_radius", 0.18)), 1e-6)
        z_radius = float(cfg.get("z_radius", 0.25))
        same_stone_only = bool(cfg.get("same_stone_only", True))
        use_init_pose = bool(cfg.get("use_init_pose", False))
        pose = np.asarray(
            action.init_pose if use_init_pose else action.pose, dtype=float
        )
        for rejected in getattr(self, "_execution_rejected_actions", []):
            if rejected is None:
                continue
            if same_stone_only and int(action.stone_id) != int(rejected.stone_id):
                continue
            rejected_pose = np.asarray(
                rejected.init_pose if use_init_pose else rejected.pose,
                dtype=float,
            )
            if float(np.linalg.norm(pose[:2] - rejected_pose[:2])) > xy_radius:
                continue
            if z_radius > 0.0 and abs(float(pose[2] - rejected_pose[2])) > z_radius:
                continue
            return True
        return False

    def _mark_nonfinite_scores(
        self,
        actions: List[Action],
        valid_mask: np.ndarray,
        scores: np.ndarray,
        source: str,
    ) -> None:
        for idx, action in enumerate(actions):
            if not bool(valid_mask[idx]):
                continue
            diagnostics = getattr(action, "diagnostics", {}) or {}
            if diagnostics.get("mcts_rejection_reason", None):
                continue
            if not np.isfinite(scores[idx]):
                self._mark_action_rejected(action, f"nonfinite_{source}")

    @timed("mcts_heuristic_priors")
    def _heuristic_action_priors(
        self,
        node: MCTS_Node,
        actions: list[Action],
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Score posegen equilibrium and target overlap before simulation."""
        priors = np.full(len(actions), -np.inf, dtype=float)
        weights = self.env.cfg.reward.weights
        posegen_thresh = max(float(self.env.cfg.reward.posegen_thresh), 1e-12)
        target = self.env.inventory.target_wall
        stones = self.env.inventory.stones
        for idx, action in enumerate(actions):
            if not valid_mask[idx] or action.stone_idx < 0:
                continue

            stone = stones[action.stone_idx]
            stone_iou = target.compute_IoU(
                stone,
                action.pose,
                divide_by_stone_volume=True,
                save_models=False,
            )
            stability = -min(float(action.c_feq) / posegen_thresh, 1.0)
            priors[idx] = (
                float(weights.stability) * stability
                + float(weights.stone_IoU) * stone_iou
            )
        return priors

    def _select(
        self,
        node: MCTS_Node,
        epsilon: float = 1e-1,
        include_failed: bool = False,
    ) -> MCTS_Node:
        while (not node.is_terminal()) and node.is_fully_expanded(include_failed):
            node = node.select_child(
                self.exploration_constant,
                epsilon=epsilon,
                include_failed=include_failed,
            )

        return node

    @timed("mcts_simulate_node")
    def _simulate_node(self, node: MCTS_Node, use_qfunction: bool) -> MCTS_Node:
        simulate = self._should_simulate(node)
        result = self._run_tree_candidate_simulation(
            self.env,
            node.parent.state,
            simulate=simulate,
            action=node.action,
        )
        if simulate:
            self._record_short_simulation_batch(1, parallel=False)
        return self._apply_tree_candidate_result(node, result, use_qfunction)

    def _simulate_selected_with_siblings(
        self,
        parent: MCTS_Node,
        selected: MCTS_Node,
        use_qfunction: bool,
    ) -> MCTS_Node:
        if not self._should_simulate(selected):
            return self._simulate_node(selected, use_qfunction)

        expandable = parent.expandable_children()
        if selected not in expandable:
            return self._simulate_node(selected, use_qfunction)
        n_threads = self._simulation_threads(len(expandable))
        if n_threads <= 1:
            return self._simulate_node(selected, use_qfunction)

        ordered = [selected]
        ordered.extend(child for child in expandable if child is not selected)
        batch = ordered[:n_threads]
        with phase("mcts_simulate_node"):
            results = list(
                self._simulation_executor_instance().map(
                    lambda child: self._run_tree_candidate_simulation(
                        self._thread_env(),
                        parent.state,
                        action=child.action,
                        simulate=True,
                    ),
                    batch,
                )
            )
        for child, result in zip(batch, results):
            self._apply_tree_candidate_result(child, result, use_qfunction)
        self._record_short_simulation_batch(len(batch), parallel=True)

        for sibling in batch:
            if sibling is not selected:
                self._backpropagate(sibling)
        return selected

    def _simulate_node_from_state(
        self,
        parent_state: State,
        node: MCTS_Node,
        simulate: bool,
        use_qfunction: bool,
    ) -> MCTS_Node:
        result = self._run_tree_candidate_simulation(
            self.env,
            parent_state,
            action=node.action,
            simulate=simulate,
        )
        if simulate:
            self._record_short_simulation_batch(1, parallel=False)
        return self._apply_tree_candidate_result(node, result, use_qfunction)

    def _run_tree_candidate_simulation(
        self,
        env,
        parent_state: State,
        action: Action,
        simulate: bool,
    ) -> tuple:
        legacy_profile = (
            mcts_validation_config(self.cfg).legacy_short_profile if simulate else None
        )
        env.update_from_state(parent_state)
        state, obs, done, reward, info_r = env.step(
            action,
            simulate,
            simulation_mode="short",
            simulation_overrides=legacy_profile,
        )
        info_r["simulation_type"] = "short" if simulate else "none"
        self._record_simulation_motion_info(
            env,
            state,
            action,
            info_r,
        )

        failure_reason = None
        if state.failed:
            failure_reason = self._simulation_state_failure_reason(
                env,
                state,
                action,
                info_r,
            )
            self._attach_scene_motion_failure(
                env,
                state,
                action,
                info_r,
                failure_reason,
            )
        return state, obs, done, reward, info_r, failure_reason

    def _apply_tree_candidate_result(
        self,
        node: MCTS_Node,
        result: tuple,
        use_qfunction: bool,
    ) -> MCTS_Node:
        state, obs, done, reward, info_r, failure_reason = result
        node.action = refined_action_from_state(node.action, state)

        node.update_state(state, obs, reward, done, state.failed, info_r)
        node.set_is_simulated(True)

        if use_qfunction:
            with torch.no_grad():
                value = self.qfunction({"obs": Observation.to_dict(node.obs)})
            node.value_init = value.item()
        else:
            node.value_init = node.get_estimated_reward_to_go()

        if failure_reason is not None:
            node._final_validation_scene_motion = getattr(
                state,
                "_final_validation_scene_motion",
                None,
            )
            self._mark_node_unselectable(node, failure_reason)
            node.q_value = -np.inf

        return node

    def _mark_node_unselectable(self, node: MCTS_Node, reason: str) -> None:
        if node.info is None:
            node.info = {}
        node.info["mcts_unselectable_reason"] = str(reason)

    def _long_simulation_failure_reason(
        self,
        env,
        action: Action,
        state: State,
        info_r: Dict,
        scene_state: Optional[State] = None,
    ) -> Optional[str]:
        if action is None or action.stone_idx < 0:
            return "invalid_action"
        if not bool(getattr(state, "simulation_settled", True)):
            info_r["simulation_settled"] = False
            return "simulation_unsettled"
        if not self._place_stability_margin_ok(info_r):
            return "place_robustness"
        final_pose = settled_action_pose(env, action, state)
        plane_gap_failure = place_plane_gap_failure_reason(
            env,
            action.stone_idx,
            final_pose,
            info_r,
        )
        if plane_gap_failure is not None:
            return plane_gap_failure
        scene_gap_failure = place_scene_gap_failure_reason(
            env,
            state if scene_state is None else scene_state,
            action.stone_idx,
            final_pose,
            info_r,
        )
        if scene_gap_failure is not None:
            return scene_gap_failure
        if not final_support_ok(env, state, info_r):
            return "final_support"
        allow_lower_fill_ground = support_config(
            env.inventory
        ).connected_ground_allow_lower_fill and bool(
            action.diagnostics.get("lower_fill_candidate", False)
        )
        if not allow_lower_fill_ground and not ground_placement_connected_ok(
            env.inventory,
            state,
            action.stone_idx,
            final_pose,
            support_count=int(info_r.get("support_count", 0)),
            support_has_ground=bool(info_r.get("support_has_ground", False)),
        ):
            return "isolated_ground_support"
        return None

    def _place_stability_margin_ok(self, info_r: Dict) -> bool:
        max_displacement = self._max_place_displacement_margin()
        if max_displacement is None:
            return True
        if bool(info_r.get("place_robustness_nonfinite", False)):
            return False
        displacement = float(info_r.get("place_robustness_displacement", np.inf))
        return np.isfinite(displacement) and displacement <= max_displacement

    def _max_place_displacement_margin(self) -> Optional[float]:
        return mcts_validation_config(self.cfg).max_place_displacement

    def _run_long_candidate_simulation(
        self,
        env,
        parent_state: State,
        action: Action,
    ) -> tuple:
        result = self._run_long_candidate_simulation_once(env, parent_state, action)
        state, _, _, _, _, failure_reason = result
        retry_info = self._scene_motion_retry_info(
            env,
            state,
            action,
            failure_reason,
            result[4],
        )
        if retry_info is None:
            return result

        scene_motion = self._scene_motion_record(
            state,
            retry_info["final_validation_initial_max_velocity_stone_id"],
            retry_info["final_validation_initial_max_velocity_integral"],
        )
        env.simulator.recreate_physics_context()
        retry_result = self._run_long_candidate_simulation_once(
            env,
            parent_state,
            action,
        )

        retry_info_r = retry_result[4]
        retry_info_r.update(retry_info)
        retry_info_r["final_validation_scene_motion_retry_failure"] = retry_result[5]
        retry_info_r["final_validation_scene_motion_retry_passed"] = (
            retry_result[5] is None
        )
        setattr(retry_result[0], "_final_validation_scene_motion", scene_motion)
        return retry_result

    def _run_long_candidate_simulation_once(
        self,
        env,
        parent_state: State,
        action: Action,
    ) -> tuple:
        validation = mcts_validation_config(self.cfg)
        legacy_profile = validation.legacy_long_profile
        velocity_threshold = env.simulator.simulation_limits(
            "long",
            legacy_profile,
        ).vel_thresh
        env.update_from_state(parent_state)
        state, obs, done, reward, info_r = env.step(
            action,
            simulate=True,
            simulation_mode="long",
            simulation_overrides=legacy_profile,
        )
        info_r["simulation_type"] = "long"
        info_r["long_simulation_velocity_integral_threshold"] = velocity_threshold
        self._record_simulation_motion_info(env, state, action, info_r)

        failure_reason = None
        if bool(state.failed):
            failure_reason = self._simulation_state_failure_reason(
                env,
                state,
                action,
                info_r,
            )
        else:
            failure_reason = self._long_simulation_failure_reason(
                env,
                action,
                state,
                info_r,
                scene_state=parent_state,
            )
        return state, obs, done, reward, info_r, failure_reason

    def _attach_scene_motion_failure(
        self,
        env,
        state,
        action,
        info_r: Optional[Dict] = None,
        failure_reason: str = "scene_motion_limit",
    ) -> None:
        scene_info = self._scene_motion_failure_info(
            env,
            state,
            action,
            failure_reason,
            info_r,
        )
        if scene_info is None:
            return
        if info_r is not None:
            info_r.update(scene_info)
        setattr(
            state,
            "_final_validation_scene_motion",
            self._scene_motion_record(
                state,
                scene_info["scene_motion_initial_max_velocity_stone_id"],
                scene_info["scene_motion_initial_max_velocity_integral"],
            ),
        )

    @staticmethod
    def _scene_motion_failure_info(env, state, action, failure_reason, info_r=None):
        if failure_reason != "scene_motion_limit" or state is None or action is None:
            return None
        velocity_integrals = state.latest_velocity_integrals()
        if not velocity_integrals:
            return None

        values = {
            int(stone_id): float(value)
            for stone_id, value in velocity_integrals.items()
            if not np.isnan(float(value))
        }
        target_stone_id = int(action.stone_id)
        if target_stone_id not in values:
            return None
        max_stone_id, max_value = max(values.items(), key=lambda item: item[1])
        target_value = values[target_stone_id]
        threshold = float(
            (info_r or {}).get(
                "long_simulation_velocity_integral_threshold",
                (info_r or {}).get(
                    "final_validation_velocity_integral_threshold",
                    (info_r or {}).get(
                        "simulation_velocity_integral_threshold",
                        env.cfg.reward.vel_integral_thresh,
                    ),
                ),
            )
        )
        if (
            max_stone_id == target_stone_id
            or max_value <= threshold
            or target_value > threshold
        ):
            return None
        return {
            "scene_motion_failure": True,
            "scene_motion_initial_max_velocity_integral": max_value,
            "scene_motion_initial_max_velocity_stone_id": max_stone_id,
            "scene_motion_initial_target_velocity_integral": target_value,
            "scene_motion_velocity_integral_threshold": threshold,
        }

    def _simulation_state_failure_reason(
        self,
        env,
        state,
        action,
        info_r: Optional[Dict] = None,
    ) -> str:
        info = info_r if info_r is not None else {}
        values, target_value, max_value, _ = self._record_simulation_motion_info(
            env,
            state,
            action,
            info,
        )
        threshold = float(info["simulation_velocity_integral_threshold"])

        if values and any(not np.isfinite(value) for value in values.values()):
            reason = "nonfinite_simulation"
        elif np.isfinite(target_value) and target_value > threshold:
            reason = "target_motion_limit"
        elif np.isfinite(max_value) and max_value > threshold:
            reason = "scene_motion_limit"
        elif self._simulator_place_robustness_failed(env, action):
            reason = "place_robustness"
        elif not self._place_stability_margin_ok(info):
            reason = "place_robustness"
        else:
            reason = "simulation_failure_unattributed"
        info["simulation_failure_reason"] = reason
        return reason

    @staticmethod
    def _record_simulation_motion_info(
        env,
        state,
        action,
        info: Dict,
    ) -> tuple[Dict[int, float], float, float, float]:
        values = {
            int(stone_id): float(value)
            for stone_id, value in state.latest_velocity_integrals().items()
        }
        threshold = float(
            info.get(
                "long_simulation_velocity_integral_threshold",
                info.get(
                    "final_validation_velocity_integral_threshold",
                    info.get(
                        "simulation_velocity_integral_threshold",
                        env.cfg.reward.vel_integral_thresh,
                    ),
                ),
            )
        )
        target_stone_id = int(action.stone_id)
        if target_stone_id < 0:
            target_stone_id = int(env.inventory.stones[action.stone_idx].id)
        target_value = values.get(target_stone_id, float("nan"))
        target_trajectory = None
        trajectories = getattr(state, "trajectories", None) or []
        if trajectories:
            target_trajectory = trajectories[-1].get(target_stone_id)
        finite_values = {
            stone_id: value for stone_id, value in values.items() if np.isfinite(value)
        }
        max_stone_id = None
        max_value = float("nan")
        if finite_values:
            max_stone_id, max_value = max(
                finite_values.items(),
                key=lambda item: item[1],
            )

        info.update(
            {
                "simulation_velocity_integral_threshold": threshold,
                "simulation_max_velocity_integral": max_value,
                "simulation_max_velocity_stone_id": max_stone_id,
                "simulation_target_velocity_integral": target_value,
                "simulation_target_settle_position_delta": float(
                    getattr(target_trajectory, "settle_position_delta", 0.0)
                ),
                "simulation_target_settle_path_length": float(
                    getattr(target_trajectory, "settle_path_length", 0.0)
                ),
                "simulation_settled": bool(getattr(state, "simulation_settled", True)),
            }
        )
        return values, target_value, max_value, threshold

    @staticmethod
    def _simulator_place_robustness_failed(env, action) -> bool:
        check = getattr(
            getattr(env, "simulator", None),
            "_place_robustness_failed",
            None,
        )
        if check is None:
            return False
        try:
            return bool(check(float(action.place_robustness_displacement)))
        except Exception:
            return False

    @staticmethod
    def _scene_motion_retry_info(env, state, action, failure_reason, info_r=None):
        scene_info = MonteCarloTreeSearch._scene_motion_failure_info(
            env,
            state,
            action,
            failure_reason,
            info_r,
        )
        if scene_info is None:
            return None
        return {
            **scene_info,
            "final_validation_scene_motion_retry": True,
            "final_validation_initial_max_velocity_integral": scene_info[
                "scene_motion_initial_max_velocity_integral"
            ],
            "final_validation_initial_max_velocity_stone_id": scene_info[
                "scene_motion_initial_max_velocity_stone_id"
            ],
            "final_validation_initial_target_velocity_integral": scene_info[
                "scene_motion_initial_target_velocity_integral"
            ],
            "final_validation_velocity_integral_threshold": scene_info[
                "scene_motion_velocity_integral_threshold"
            ],
        }

    @staticmethod
    def _scene_motion_record(state, stone_id, velocity_integral) -> dict:
        stone_id = int(stone_id)
        return {
            "stone_id": stone_id,
            "velocity_integral": float(velocity_integral),
            "trajectory": MonteCarloTreeSearch._stone_trajectory_poses(
                state,
                stone_id,
            ),
        }

    @staticmethod
    def _stone_trajectory_poses(state, stone_id) -> list:
        trajectories = getattr(state, "trajectories", None) or []
        if not trajectories:
            return []
        trajectory = trajectories[-1].get(int(stone_id))
        if trajectory is None:
            return []
        poses = []
        for pose in getattr(trajectory, "poses", []) or []:
            pose = np.asarray(pose, dtype=float)
            if pose.shape[0] >= 7 and np.all(np.isfinite(pose[:7])):
                poses.append(pose[:7].copy())
        return poses

    def _apply_long_candidate_result(
        self,
        node: MCTS_Node,
        result: tuple,
        use_qfunction: bool,
    ) -> MCTS_Node:
        state, obs, done, reward, info_r, failure_reason = result
        node.action = refined_action_from_state(node.action, state)
        node.update_state(state, obs, reward, done, state.failed, info_r)
        node._final_validation_scene_motion = getattr(
            state,
            "_final_validation_scene_motion",
            None,
        )
        node.set_is_simulated(True)

        if use_qfunction:
            with torch.no_grad():
                value = self.qfunction({"obs": Observation.to_dict(node.obs)})
            node.value_init = value.item()
        else:
            node.value_init = node.get_estimated_reward_to_go()

        if failure_reason is not None:
            self._mark_final_validation_failure(node, failure_reason)
            self._mark_node_unselectable(node, failure_reason)
            node.set_failed(True)
            node.q_value = -np.inf
        elif not np.isfinite(node.q_value):
            node.q_value = 0.0
            node.visits = 0
        return node

    @timed("mcts_final_validate")
    def _simulate_long_candidate(
        self,
        parent_state: State,
        node: MCTS_Node,
        use_qfunction: bool,
    ) -> MCTS_Node:
        result = self._run_long_candidate_simulation(
            self.env,
            parent_state,
            node.action,
        )
        return self._apply_long_candidate_result(node, result, use_qfunction)

    def _tree_simulation_depth(self) -> int:
        # Deepest node depth that runs short simulation; nodes below
        # this take the cheap non-dynamics step and lean on the heuristic value.
        # null/absent -> TREE_SIMULATION_DEPTH. Set >= max_depth to simulate the
        # whole tree (no simulation-skip), so backed-up values reflect real
        # physics rather than the heuristic policy.
        val = self.cfg.get("tree_simulation_depth", None)
        if val is None:
            return TREE_SIMULATION_DEPTH
        return int(val)

    def _should_simulate(self, node: MCTS_Node) -> bool:
        return node.depth <= self._tree_simulation_depth()

    def _backpropagate(self, node: MCTS_Node):
        reward_to_go = (
            (1.0 - float(node.done))
            * self.cfg.reward.discount
            * (
                node.confidence * node.value_init
                + (1 - node.confidence) * node.get_estimated_reward_to_go()
            )
        )
        if not np.isfinite(reward_to_go):
            reward_to_go = -1.0

        while node is not None:
            reward = -1.0 if node.failed else node.reward
            if reward is None or not np.isfinite(reward):
                reward = -1.0
            reward_to_go = reward + self.cfg.reward.discount * reward_to_go
            if not np.isfinite(reward_to_go):
                reward_to_go = -1.0
            node.visit((1.0 - float(node.failed)) * reward_to_go)
            node = node.parent

    def set_config(self, cfg: OmegaConf):
        self.cfg = cfg

    def close(self) -> None:
        self._shutdown_simulation_executor()
        env = getattr(self, "env", None)
        if env is not None:
            env.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
