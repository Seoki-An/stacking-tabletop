from typing import Dict, List, Optional

import numpy as np

from .node import MCTS_Node


class RootLookaheadMixin:
    @staticmethod
    def _log_root_lookahead_coverage_summary(stats: Optional[Dict]) -> None:
        if not stats or int(stats.get("eligible", 0)) == 0:
            return
        print(
            "[INFO] Root lookahead coverage: "
            f"eligible={int(stats['eligible'])}, "
            f"attempted={int(stats['attempted'])}, "
            f"generated={int(stats['generated'])}, "
            f"short_feasible={int(stats['short_feasible'])}, "
            f"long_validated={int(stats['long_validated'])}, "
            f"pending={int(stats['pending'])}, "
            f"selected_fallback={int(stats.get('selected_mean_fallback', 0))}, "
            f"selected_validated={int(stats.get('selected_validated_path', 0))}"
        )

    @staticmethod
    def _only_mean_fallback_candidates(nodes: List[MCTS_Node]) -> bool:
        return bool(nodes) and all(
            node is not None
            and bool(
                (getattr(node, "info", None) or {}).get(
                    "final_selection_used_mean_fallback",
                    False,
                )
            )
            for node in nodes
        )

    def _prepare_mean_fallback_retry(self, root: MCTS_Node) -> Dict:
        """Free root attempt slots rejected by long simulation for one retry."""
        selectable = root.selectable_children()
        selectable_ids = {id(child) for child in selectable}
        retired = [child for child in root.children if id(child) not in selectable_ids]
        stats = getattr(self, "_mean_fallback_retry_stats", None)
        if stats is None:
            stats = {
                "attempted": False,
                "retired_rejections": 0,
                "new_attempts": 0,
                "new_selectable": 0,
                "validated_path_found": False,
            }
            self._mean_fallback_retry_stats = stats
        if not retired:
            return stats

        rejected_actions = getattr(self, "_root_retry_rejected_actions", None)
        if rejected_actions is None:
            rejected_actions = []
            self._root_retry_rejected_actions = rejected_actions
        rejected_actions.extend(
            child.action for child in retired if child.action is not None
        )
        root.children = selectable
        root.set_action_generation_exhausted(False)
        stats.update(
            {
                "attempted": True,
                "retired_rejections": len(retired),
                "retained_selectable": len(selectable),
                "children_before_retry": len(root.children),
            }
        )
        return stats

    def _finish_mean_fallback_retry(
        self,
        root: MCTS_Node,
        nodes: List[MCTS_Node],
    ) -> None:
        stats = self._mean_fallback_retry_stats
        retained = int(stats.get("children_before_retry", len(root.children)))
        stats["new_attempts"] = max(len(root.children) - retained, 0)
        stats["new_selectable"] = max(
            len(root.selectable_children())
            - int(stats.get("retained_selectable", 0)),
            0,
        )
        stats["validated_path_found"] = bool(
            nodes and not self._only_mean_fallback_candidates(nodes)
        )

    @staticmethod
    def _log_mean_fallback_retry_summary(stats: Optional[Dict]) -> None:
        if not stats or not bool(stats.get("attempted", False)):
            return
        print(
            "[INFO] Mean-fallback root retry: "
            f"retired={int(stats.get('retired_rejections', 0))}, "
            f"new_attempts={int(stats.get('new_attempts', 0))}, "
            f"new_selectable={int(stats.get('new_selectable', 0))}, "
            "validated_path_found="
            f"{bool(stats.get('validated_path_found', False))}"
        )

    @staticmethod
    def _record_root_selection_diagnostics(
        child: MCTS_Node,
        validation_depth: int,
        path_score: float,
        used_mean_fallback: bool,
    ) -> None:
        if child.info is None:
            child.info = {}
        descendants = child.simulated_children(include_failed=True)
        child.info["final_selection_validation_depth"] = int(validation_depth)
        child.info["final_selection_used_mean_fallback"] = bool(used_mean_fallback)
        child.info["root_lookahead_child_count"] = int(len(child.children))
        child.info["root_lookahead_simulated_count"] = int(len(descendants))
        child.info["root_lookahead_short_feasible_count"] = int(
            sum(not descendant.failed for descendant in descendants)
        )
        child.info["root_lookahead_long_validated_count"] = int(
            sum(
                bool(
                    (descendant.info or {}).get(
                        "descendant_path_final_validated",
                        False,
                    )
                )
                for descendant in descendants
            )
        )
        child.info["final_selection_path_score_finite"] = bool(
            np.isfinite(path_score)
        )

    def _seed_root_long_values(
        self,
        root: MCTS_Node,
        children: List[MCTS_Node],
    ) -> int:
        """Back up each current authoritative root result once after a reset."""
        seeded = 0
        for child in children:
            if child.visits > 0 or not self._root_long_validation_is_current(
                root,
                child,
            ):
                continue
            self._backpropagate(child)
            if child.info is None:
                child.info = {}
            child.info["root_long_value_seeded"] = True
            seeded += 1
        return seeded

    def _next_root_lookahead_child(
        self,
        root: MCTS_Node,
    ) -> Optional[MCTS_Node]:
        """Return an uncovered feasible root only after root proposal is complete."""
        if root.can_expand_children():
            return None
        candidates = []
        for child in root.selectable_children():
            if not self._root_long_validation_is_current(root, child):
                continue
            if not child.expandable_children() and not child.can_expand_children():
                continue
            info = child.info or {}
            if child.simulated_children(include_failed=True) or bool(
                info.get("root_lookahead_attempted", False)
            ):
                continue
            candidates.append(child)
        if not candidates:
            return None
        return max(candidates, key=lambda child: float(child.q_value_init))

    def _run_root_lookahead_iteration(
        self,
        child: MCTS_Node,
        use_qfunction: bool,
        epsilon: float,
        include_failed: bool,
    ) -> None:
        """Generate and short-simulate one descendant batch for one root child."""
        if child.info is None:
            child.info = {}
        child.info["root_lookahead_attempted"] = True

        if child.is_expandable() and not child.expandable_children():
            self.expand_node(child, use_qfunction)

        expandable = child.expandable_children()
        if not expandable:
            self._backpropagate(child)
            return

        selected = child.select_child(
            self.exploration_constant,
            epsilon=epsilon,
            include_failed=include_failed,
        )
        if selected is child:
            self._backpropagate(child)
            return
        if not selected.is_simulated:
            selected = self._simulate_selected_with_siblings(
                child,
                selected,
                use_qfunction,
            )
        self._backpropagate(selected)

    def _root_lookahead_coverage_stats(self, root: MCTS_Node) -> Dict[str, int]:
        stats = {
            "eligible": 0,
            "attempted": 0,
            "generated": 0,
            "short_feasible": 0,
            "long_validated": 0,
            "pending": 0,
            "mean_fallback": 0,
            "selected_mean_fallback": 0,
            "selected_validated_path": 0,
        }
        for child in root.selectable_children():
            if not self._root_long_validation_is_current(root, child):
                continue
            if child.done or child.depth >= int(self.cfg.max_depth):
                continue
            stats["eligible"] += 1
            info = child.info or {}
            attempted = bool(info.get("root_lookahead_attempted", False)) or bool(
                child.simulated_children(include_failed=True)
            )
            stats["attempted"] += int(attempted)
            stats["generated"] += int(bool(child.children))
            descendants = child.simulated_children(include_failed=True)
            stats["short_feasible"] += int(
                any(not descendant.failed for descendant in descendants)
            )
            stats["long_validated"] += int(
                any(
                    bool(
                        (descendant.info or {}).get(
                            "descendant_path_final_validated",
                            False,
                        )
                    )
                    for descendant in descendants
                )
            )
            can_cover = bool(
                child.expandable_children()
            ) or child.can_expand_children()
            stats["pending"] += int(can_cover and not attempted)
            stats["mean_fallback"] += int(
                bool(info.get("final_selection_used_mean_fallback", False))
            )
            is_selected = bool(info.get("final_selection_best", False))
            stats["selected_mean_fallback"] += int(
                is_selected
                and bool(info.get("final_selection_used_mean_fallback", False))
            )
            stats["selected_validated_path"] += int(
                is_selected
                and bool(info.get("validated_descendant_path_available", False))
            )
        return stats
