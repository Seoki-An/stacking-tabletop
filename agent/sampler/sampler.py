import gc
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import h5py
import numpy as np
import ray
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf

from loader.heightmap_feasibility import (
    FEASIBILITY_FAILURE_REASONS,
    candidate_heightmap_stack,
    feasibility_failure_reason_id,
    resize_heightmap,
    stone_heightmaps_from_mesh,
)
from agent.mcts import MCTS_Node, MonteCarloTreeSearch
from agent.env.components.action import Action
from utils import (
    dict_list_to_nparray,
    extend_dict_of_list,
)


from utils._phase_timer import (
    phase as _timing_phase,
    reset as _timing_reset,
    format_summary as _timing_format,
)


def _grasp_access_mcts_hard_filter(grasp_cfg) -> bool:
    return bool((grasp_cfg or {}).get("mcts_hard_filter", True))


def _perturb_feasibility_pose(
    pose: np.ndarray,
    rng: np.random.Generator,
    xy_std: float,
    z_std: float,
    rotation_std_deg: float,
) -> np.ndarray:
    perturbed = np.asarray(pose, dtype=float).copy()
    perturbed[:2] += rng.normal(0.0, max(float(xy_std), 0.0), size=2)
    perturbed[2] += float(rng.normal(0.0, max(float(z_std), 0.0)))
    rotation_std = np.deg2rad(max(float(rotation_std_deg), 0.0))
    if rotation_std > 0.0:
        delta = Rotation.from_rotvec(rng.normal(0.0, rotation_std, size=3))
        base = Rotation.from_quat(perturbed[3:])
        perturbed[3:] = (delta * base).as_quat()
    return perturbed


def _init_dataset(sample: Union[dict, list]) -> Union[dict, list]:
    """Wrap every leaf of a sample dict as a single-element list."""
    if isinstance(sample, dict):
        return {k: _init_dataset(v) for k, v in sample.items()}
    return [sample]


def _build_dataset(samples: List[dict]) -> dict:
    if not samples:
        return {}
    dataset = _init_dataset(samples[0])
    for sample in samples[1:]:
        extend_dict_of_list(dataset, sample)
    return dict_list_to_nparray(dataset)


def _build_committed_multistep_samples(
    transitions: List[Tuple[MCTS_Node, Action, float]],
    final_node: MCTS_Node,
    discount: float,
) -> List[dict]:
    """Return Monte Carlo targets from committed states to the episode end."""
    if final_node is None or not final_node.done or final_node.failed:
        return []

    samples = []
    reward_to_go = 0.0
    horizon = 0
    for start_node, action, reward in reversed(transitions):
        reward_to_go = float(reward) + float(discount) * reward_to_go
        horizon += 1
        if horizon == 1:
            continue
        samples.append(
            start_node.make_multistep_sample(
                action=action,
                reward_to_go=reward_to_go,
                next_node=final_node,
                n_step=horizon,
            )
        )
    samples.reverse()
    return samples


class _StreamingH5Writer:
    """Appends batched samples to an h5 file with resizable leaf datasets.

    Schema is initialized lazily from the first non-empty batch, so callers can
    pour in step-by-step samples without staging the whole episode in RAM.
    """

    def __init__(self, path: str):
        self.path = path
        self._file: Optional[h5py.File] = None

    def write(self, samples: List[dict]) -> None:
        if not samples:
            return
        batched = _build_dataset(samples)
        if self._file is None:
            self._file = h5py.File(self.path, "w")
            self._init_schema(batched, self._file)
        self._append(batched, self._file)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def has_data(self) -> bool:
        return self._file is not None

    @staticmethod
    def _init_schema(batched: dict, parent: h5py.Group, prefix: str = "") -> None:
        for key, val in batched.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(val, dict):
                _StreamingH5Writer._init_schema(
                    val,
                    parent.create_group(key),
                    path,
                )
            else:
                if not hasattr(val, "dtype"):
                    raise TypeError(
                        f"cannot write non-array value to H5 at '{path}' "
                        f"(type={type(val).__name__}); check for variable row shapes"
                    )
                if val.dtype == object:
                    raise TypeError(f"cannot write object dtype to H5 at '{path}'")
                shape = (0,) + tuple(val.shape[1:])
                maxshape = (None,) + tuple(val.shape[1:])
                parent.create_dataset(
                    key,
                    shape=shape,
                    maxshape=maxshape,
                    dtype=val.dtype,
                    chunks=True,
                )

    @staticmethod
    def _append(batched: dict, parent: h5py.Group) -> None:
        for key, val in batched.items():
            if isinstance(val, dict):
                _StreamingH5Writer._append(val, parent[key])
            else:
                dset = parent[key]
                n = dset.shape[0]
                k = val.shape[0]
                if k == 0:
                    continue
                dset.resize(n + k, axis=0)
                dset[n:] = val


@ray.remote
class MCTS_Sampler:
    def __init__(
        self,
        env_class: gym.Env,
        env_args: Dict[str, OmegaConf],
        mcts_cfg: OmegaConf,
        id: int,
    ):
        self.mcts = MonteCarloTreeSearch(mcts_cfg, env_class, env_args)
        # Use the MCTS environment as the episode source of truth. Target-wall
        # randomization is not stored in State, so a second env can render or
        # sample against a different target from the one that produced nodes.
        self.env = self.mcts.env
        self.state, _ = self.env.reset()
        self.mcts_cfg = mcts_cfg
        self.sampler_cfg = mcts_cfg.sampler
        self.verbose = bool(mcts_cfg.get("verbose", False))
        self.id = id

    def _log(self, *args, **kwargs) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def sample_episode(
        self,
        save_dir: str,
        episode_id: int,
        epsilon: float = 1e-1,
        save_multistep: bool = False,
        log_time: bool = False,
        feasibility_cfg: Optional[OmegaConf] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], dict]:
        """Sample one episode.

        Returns paths for one-step, multistep, and feasibility rows followed by
        compact episode outcome statistics. A path is None when no rows were
        emitted.
        """
        save_dir = os.path.normpath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        hdf5_path = os.path.join(save_dir, f"actor_{self.id}_ep{episode_id}.h5")
        multi_path = os.path.join(
            save_dir,
            f"actor_{self.id}_ep{episode_id}_multi.h5",
        )
        if feasibility_cfg is None:
            feasibility_cfg = {}
        save_feasibility = bool(feasibility_cfg.get("enabled", False))
        feasibility_tmp_dir = os.path.join(save_dir, "feasibility", "_tmp")
        feasibility_path = os.path.join(
            feasibility_tmp_dir,
            f"actor_{self.id}_ep{episode_id}.h5",
        )
        if save_feasibility:
            os.makedirs(feasibility_tmp_dir, exist_ok=True)
        candidate_filter = self._make_mcts_candidate_filter(feasibility_cfg)

        _timing_reset()
        episode_started = time.perf_counter()
        phase_seconds = {
            "preserve": 0.0,
            "search": 0.0,
            "feasibility": 0.0,
            "one_step_write": 0.0,
            "multistep_write": 0.0,
            "final_figure": 0.0,
        }
        one_step_rows = 0
        feasibility_rows = 0
        multistep_rows = 0

        state, obs = self.mcts.env.reset()
        self.env = self.mcts.env
        root = MCTS_Node(cfg=self.mcts_cfg)
        root.update_state(state, obs, 0, False, False)
        node = root
        committed_transitions: List[Tuple[MCTS_Node, Action, float]] = []

        exploration_constant = float(self.mcts.cfg.exploration_constant)
        exploration_decay = float(self.sampler_cfg.exploration_decay)
        self._log(f"[actor {self.id}] episode {episode_id} starts")

        writer = _StreamingH5Writer(hdf5_path)
        feasibility_writer = (
            _StreamingH5Writer(feasibility_path) if save_feasibility else None
        )
        try:
            for n_step in range(self.env.cfg.n_stone):
                self.mcts.exploration_constant = exploration_constant
                if n_step > 0:
                    t0 = time.perf_counter()
                    preserve_stats = self.mcts.revalidate_preserved_root(
                        node,
                        use_qfunction=False,
                        max_child_drift=float(
                            self.sampler_cfg.get("preserve_max_child_drift", 0.1)
                        ),
                    )
                    phase_seconds["preserve"] += time.perf_counter() - t0
                    self._log(
                        f"[actor {self.id}] episode {episode_id} step {n_step + 1} "
                        f"preserved={preserve_stats.get('kept', 0)} "
                        f"pruned={preserve_stats.get('pruned', 0)} "
                        f"failed={preserve_stats.get('failed', 0)}"
                    )
                t0 = time.perf_counter()
                with _timing_phase("mcts_search"):
                    node_, _ = self.mcts.search(
                        root=node,
                        use_qfunction=False,
                        preserve_tree=True,
                        epsilon=epsilon,
                        eval=False,
                        candidate_filter=candidate_filter,
                    )
                    retry_idx = 0
                    while (
                        self._no_stacking_progress(node, node_)
                        and retry_idx < self._no_progress_retries()
                    ):
                        retry_idx += 1
                        self._log(
                            f"[actor {self.id}] episode {episode_id} - step {n_step + 1} "
                            f"made no stacking progress; retry {retry_idx}/"
                            f"{self._no_progress_retries()} with diversified root sampling"
                        )
                        node.reset(preserve_tree=False, depth=0)
                        node_, _ = self.mcts.search(
                            root=node,
                            use_qfunction=False,
                            preserve_tree=True,
                            epsilon=epsilon,
                            eval=False,
                            candidate_filter=candidate_filter,
                            diversify_root_sampling=True,
                        )
                step_elapsed = time.perf_counter() - t0
                phase_seconds["search"] += step_elapsed
                exploration_constant = float(
                    np.clip(
                        exploration_constant * exploration_decay,
                        self.sampler_cfg.min_exploration_constant,
                        self.sampler_cfg.max_exploration_constant,
                    )
                )
                leaf_message = (
                    f"[actor {self.id}] episode {episode_id} step {n_step + 1} "
                    f"leaves={node.number_of_leaves()}"
                )
                if self.verbose:
                    leaf_message += f" time={step_elapsed:.2f}s"
                print(leaf_message)

                if save_feasibility:
                    t0 = time.perf_counter()
                    feasibility_rows += self._write_feasibility_rows(
                        node,
                        feasibility_cfg,
                        feasibility_writer,
                        episode_id,
                        n_step + 1,
                    )
                    phase_seconds["feasibility"] += time.perf_counter() - t0
                one_step_samples = node.aggregate_tree_states()
                t0 = time.perf_counter()
                writer.write(one_step_samples)
                phase_seconds["one_step_write"] += time.perf_counter() - t0
                one_step_rows += len(one_step_samples)
                if self._no_stacking_progress(node, node_):
                    self._log(
                        f"[actor {self.id}] episode {episode_id} - step {n_step + 1} "
                        "still made no stacking progress; marking episode failed"
                    )
                    node.set_failed(True)
                    break
                if save_multistep:
                    committed_transitions.append(
                        (node, node_.action.copy(), float(node_.reward))
                    )
                node.children = []
                node = node_
                node.parent = None
                if node.done or node.failed:
                    break
                gc.collect()
        finally:
            wrote_any = writer.has_data()
            wrote_feasibility = (
                feasibility_writer is not None and feasibility_writer.has_data()
            )
            writer.close()
            if feasibility_writer is not None:
                feasibility_writer.close()

        one_out: Optional[str] = hdf5_path if wrote_any else None
        multi_out: Optional[str] = None
        feasibility_out: Optional[str] = (
            feasibility_path if wrote_feasibility else None
        )
        if save_multistep:
            multi_writer = _StreamingH5Writer(multi_path)
            try:
                multistep_samples = _build_committed_multistep_samples(
                    committed_transitions,
                    node,
                    float(self.mcts_cfg.reward.discount),
                )
                t0 = time.perf_counter()
                multi_writer.write(multistep_samples)
                phase_seconds["multistep_write"] += time.perf_counter() - t0
                multistep_rows = len(multistep_samples)
            finally:
                wrote_multi = multi_writer.has_data()
                multi_writer.close()
            if wrote_multi:
                multi_out = multi_path

        if bool(self.sampler_cfg.get("save_final_figure", True)):
            t0 = time.perf_counter()
            self._write_final_figure(save_dir, episode_id, node)
            phase_seconds["final_figure"] += time.perf_counter() - t0

        self._log(f"[actor {self.id}] episode {episode_id} ends - done: {node.done}")

        if log_time:
            print(_timing_format(f"  [actor {self.id}] episode {episode_id} timing:"))

        episode_stats = {
            "episode_id": int(episode_id),
            "done": bool(node.done),
            "failed": bool(node.failed),
            "placements": int(len(node.state.stone_seq)),
            "elapsed_seconds": time.perf_counter() - episode_started,
            "one_step_rows": one_step_rows,
            "feasibility_rows": feasibility_rows,
            "multistep_rows": multistep_rows,
            "phase_seconds": phase_seconds,
        }
        return one_out, multi_out, feasibility_out, episode_stats

    def _no_progress_retries(self) -> int:
        return max(int(self.sampler_cfg.get("no_progress_retries", 0)), 0)

    @staticmethod
    def _no_stacking_progress(parent: MCTS_Node, child: MCTS_Node) -> bool:
        if child is parent:
            return True
        if child is None or child.state is None or parent.state is None:
            return True
        return len(child.state.stone_seq) <= len(parent.state.stone_seq)

    def _write_final_figure(
        self,
        save_dir: str,
        episode_id: int,
        node: MCTS_Node,
    ) -> None:
        if node is None or node.state is None:
            return

        figure_dir = os.path.join(save_dir, "figures")
        path = os.path.join(
            figure_dir,
            f"actor_{self.id}_ep{episode_id}_final.png",
        )
        try:
            self.mcts.env.update_from_state(node.state)
            self.mcts.env.save_configuration_figure(path)
            self._log(
                f"[actor {self.id}] episode {episode_id} final figure saved: {path}"
            )
        except Exception as exc:
            print(
                f"[actor {self.id}] episode {episode_id} "
                f"final figure skipped: {exc}"
            )

    def _write_feasibility_rows(
        self,
        root: MCTS_Node,
        cfg,
        writer: _StreamingH5Writer,
        episode_id: int,
        step: int,
    ) -> int:
        rows = self._collect_feasibility_rows(root, cfg)
        rows = self._sample_feasibility_rows(rows, cfg, episode_id, step)
        rows = self._annotate_feasibility_rollouts(rows, cfg, episode_id, step)
        if not rows:
            return 0

        for row in rows:
            row["actor_id"] = int(self.id)
            row["episode_id"] = int(episode_id)
            row["step"] = int(step)
        writer.write(rows)
        return len(rows)

    def _annotate_feasibility_rollouts(
        self,
        rows: List[dict],
        cfg,
        episode_id: int,
        step: int,
    ) -> List[dict]:
        if not rows:
            return rows

        additional_rollouts = max(int(cfg.get("perturbation_rollouts", 0)), 0)
        max_perturbed = max(
            int(cfg.get("perturbation_max_rows_per_step", len(rows))), 0
        )
        rng = np.random.default_rng(
            int(cfg.get("seed", 0))
            + int(self.id) * 1_000_003
            + int(episode_id) * 997
            + int(step) * 37
            + 17
        )
        n_perturbed = min(len(rows), max_perturbed)
        perturbed_indices = (
            set(
                int(index)
                for index in rng.choice(
                    len(rows), size=n_perturbed, replace=False
                )
            )
            if additional_rollouts > 0 and n_perturbed > 0
            else set()
        )
        xy_std = float(cfg.get("perturbation_xy_std", 0.005))
        z_std = float(cfg.get("perturbation_z_std", 0.002))
        rotation_std_deg = float(
            cfg.get("perturbation_rotation_std_deg", 0.5)
        )

        for index, row in enumerate(rows):
            parent_state = row.pop("_parent_state", None)
            action = row.pop("_action", None)
            nominal_reason = row.pop("_failure_reason", None)
            simulation_type = row.pop("_simulation_type", "short")
            nominal_pass = float(row["label"]) > 0.5
            nominal_reason_id = feasibility_failure_reason_id(
                nominal_reason, failed=not nominal_pass
            )
            reason_counts = np.zeros(
                len(FEASIBILITY_FAILURE_REASONS), dtype=np.int16
            )
            reason_counts[nominal_reason_id] += 1
            passes = int(nominal_pass)
            rollouts = 1

            if (
                index in perturbed_indices
                and parent_state is not None
                and action is not None
            ):
                base_pose = np.asarray(row["candidate_pose"], dtype=float)
                for _ in range(additional_rollouts):
                    perturbed_action = action.copy()
                    perturbed_pose = _perturb_feasibility_pose(
                        base_pose,
                        rng,
                        xy_std,
                        z_std,
                        rotation_std_deg,
                    )
                    perturbed_action.pose = perturbed_pose.copy()
                    perturbed_action.solved_pose = perturbed_pose.copy()
                    if simulation_type == "long":
                        result = self.mcts._run_long_candidate_simulation(
                            self.mcts.env,
                            parent_state,
                            perturbed_action,
                        )
                    else:
                        result = self.mcts._run_tree_candidate_simulation(
                            self.mcts.env,
                            parent_state,
                            perturbed_action,
                            simulate=True,
                        )
                    state, failure_reason = result[0], result[5]
                    passed = not bool(state.failed) and failure_reason is None
                    passes += int(passed)
                    rollouts += 1
                    reason_id = feasibility_failure_reason_id(
                        failure_reason, failed=not passed
                    )
                    reason_counts[reason_id] += 1

            row["nominal_label"] = np.float32(nominal_pass)
            row["failure_reason_id"] = np.int16(nominal_reason_id)
            row["failure_reason_counts"] = reason_counts
            row["stability_passes"] = np.int16(passes)
            row["stability_rollouts"] = np.int16(rollouts)
            row["label"] = float(passes / rollouts)
        return rows

    def _feasibility_resolution(self, cfg) -> tuple[int, int]:
        resolution = cfg.get("resolution", None)
        if resolution is None:
            resolution = self.mcts.env.cfg.height_map.resolution
        if len(resolution) != 2:
            raise ValueError(f"feasibility resolution must have two entries: {resolution}")
        return tuple(int(v) for v in resolution)

    def _collect_feasibility_rows(self, root: MCTS_Node, cfg) -> List[dict]:
        roots_only = bool(cfg.get("roots_only", False))
        resolution = self._feasibility_resolution(cfg)
        pose_name = str(cfg.get("pose", "pose"))
        contact_eps = float(cfg.get("contact_eps", 0.03))
        samples_per_edge = int(cfg.get("stone_samples_per_edge", 3))
        positive_weight = float(cfg.get("positive_weight", 1.0))
        negative_weight = float(cfg.get("negative_weight", 1.0))

        grasp_cfg = cfg.get("grasp_access", {}) or {}
        grasp_evaluator = (
            self._ensure_grasp_evaluator(grasp_cfg)
            if bool(grasp_cfg.get("enabled", False))
            else None
        )
        grasp_mcts_hard_filter = _grasp_access_mcts_hard_filter(grasp_cfg)

        rows: List[dict] = []
        root_stone_count = len(root.state.stone_seq)
        parents = [root]
        while parents:
            parent = parents.pop()
            if parent.state is None:
                continue
            rows.extend(
                self._collect_parent_feasibility_rows(
                    parent,
                    resolution,
                    pose_name,
                    contact_eps,
                    samples_per_edge,
                    positive_weight,
                    negative_weight,
                    root_stone_count,
                    grasp_evaluator,
                    grasp_mcts_hard_filter,
                )
            )
            if not roots_only:
                parents.extend(
                    child
                    for child in parent.simulated_children(include_failed=True)
                    if child.children and not child.failed
                )
        return rows

    def _sample_feasibility_rows(
        self,
        rows: List[dict],
        cfg,
        episode_id: int,
        step: int,
    ) -> List[dict]:
        max_rows = cfg.get("max_rows_per_step", None)
        max_rows = None if max_rows is None else max(int(max_rows), 0)
        if max_rows == 0 or not rows:
            return []

        actor_id = int(getattr(self, "id", 0))
        rng = np.random.default_rng(
            int(cfg.get("seed", 0)) + actor_id * 1_000_003 + episode_id * 997 + step
        )
        if max_rows is None or len(rows) <= max_rows:
            order = rng.permutation(len(rows))
            return [rows[int(i)] for i in order]

        if not bool(cfg.get("balance_labels", True)):
            idx = rng.choice(len(rows), size=max_rows, replace=False)
            return [rows[int(i)] for i in idx]

        positive_fraction = float(cfg.get("positive_fraction", 0.5))
        positive_fraction = float(np.clip(positive_fraction, 0.0, 1.0))
        labels = np.asarray([float(row["label"]) > 0.5 for row in rows], dtype=bool)
        pos_idx = np.flatnonzero(labels)
        neg_idx = np.flatnonzero(~labels)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            idx = rng.choice(len(rows), size=max_rows, replace=False)
            return [rows[int(i)] for i in idx]

        n_pos = min(len(pos_idx), int(round(max_rows * positive_fraction)))
        n_neg = min(len(neg_idx), max_rows - n_pos)
        remaining = max_rows - n_pos - n_neg
        if remaining > 0:
            if len(pos_idx) - n_pos >= len(neg_idx) - n_neg:
                n_pos = min(len(pos_idx), n_pos + remaining)
            else:
                n_neg = min(len(neg_idx), n_neg + remaining)

        selected = np.concatenate(
            [
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=n_neg, replace=False),
            ]
        )
        rng.shuffle(selected)
        return [rows[int(i)] for i in selected]

    def _ensure_grasp_evaluator(self, grasp_cfg):
        """Lazily build the diffsim grasp-accessibility evaluator (reused)."""
        if getattr(self, "_grasp_evaluator", None) is None:
            from agent.sampler.grasp_access_diffsim import PlaceGraspSampler

            self._grasp_evaluator = PlaceGraspSampler(
                self.mcts.env.inventory,
                grasp_cfg,
                self.env.cfg,
                default_threads=getattr(self.env, "n_threads", 1) or 1,
            )
        return self._grasp_evaluator

    def _make_mcts_candidate_filter(self, feasibility_cfg):
        if not bool((feasibility_cfg or {}).get("enabled", False)):
            return None
        grasp_cfg = (feasibility_cfg or {}).get("grasp_access", {}) or {}
        if not bool(grasp_cfg.get("enabled", False)):
            return None
        if not _grasp_access_mcts_hard_filter(grasp_cfg):
            return None

        pose_name = str((feasibility_cfg or {}).get("pose", "pose"))

        def _filter(root: MCTS_Node, child: MCTS_Node):
            if (
                root is None
                or root.state is None
                or child is None
                or child.failed
                or child.action is None
            ):
                return child

            action = child.action
            pose = getattr(action, pose_name, action.pose)
            evaluator = self._ensure_grasp_evaluator(grasp_cfg)
            evaluator.set_scene(root.state)
            grasp_access, grasp_n, grasp_score = evaluator.score(
                action.stone_idx,
                pose,
            )
            if not grasp_access:
                self._mark_grasp_failed(child, grasp_n, grasp_score)
            else:
                self._mark_grasp_access_checked(child, grasp_n, grasp_score)
            return child

        return _filter

    def _collect_parent_feasibility_rows(
        self,
        parent: MCTS_Node,
        resolution: tuple[int, int],
        pose_name: str,
        contact_eps: float,
        samples_per_edge: int,
        positive_weight: float,
        negative_weight: float,
        root_stone_count: int,
        grasp_evaluator=None,
        grasp_mcts_hard_filter: bool = True,
    ) -> List[dict]:
        rows: List[dict] = []
        inventory = self.mcts.env.inventory
        inventory.update_from_state(parent.state)
        if grasp_evaluator is not None:
            grasp_evaluator.set_scene(parent.state)
        scene_height = parent.obs.scene_height_map if parent.obs is not None else None
        target_height = (
            parent.obs.target_height_map
            if parent.obs is not None and parent.obs.target_height_map is not None
            else None
        )
        if target_height is None or not np.any(target_height):
            scene_height = inventory.get_height_map(parent.state)
            target_height = inventory.ensure_target_height_map()
        elif scene_height is None:
            scene_height = inventory.get_height_map(parent.state)
        scene_height = resize_heightmap(scene_height, resolution)
        target_height = resize_heightmap(target_height, resolution)
        xlim = (
            parent.obs.height_map_xlim
            if parent.obs is not None
            else np.asarray(inventory.xlim, dtype=np.float32)
        )
        ylim = (
            parent.obs.height_map_ylim
            if parent.obs is not None
            else np.asarray(inventory.ylim, dtype=np.float32)
        )
        if parent.obs is not None and parent.obs.stone_ids is not None:
            scene_stone_ids = np.asarray(parent.obs.stone_ids, dtype=np.int32)
            scene_stone_poses = np.asarray(parent.obs.stone_poses, dtype=np.float32)
            scene_mask = np.asarray(parent.obs.scene_mask, dtype=bool)
        else:
            scene_stone_ids = np.asarray(parent.state.stone_set, dtype=np.int32)
            scene_stone_poses = np.zeros((len(scene_stone_ids), 7), dtype=np.float32)
            scene_stone_poses[:, 6] = 1.0
            scene_mask = np.zeros(len(scene_stone_ids), dtype=bool)
            for stone_idx in parent.state.stone_seq:
                stone_idx = int(stone_idx)
                stone_id = int(scene_stone_ids[stone_idx])
                scene_stone_poses[stone_idx] = parent.state.stone_poses[stone_id]
                scene_mask[stone_idx] = True


        for child in parent.simulated_children(include_failed=True):
            action = child.action
            if action is None:
                continue
            pose = getattr(action, pose_name, action.pose)
            if pose is None or np.asarray(pose).shape != (7,):
                continue
            stone = inventory.stones[action.stone_idx]
            vertices, faces = stone.get_lowpoly_mesh_array()
            bottom, top, mask = stone_heightmaps_from_mesh(
                vertices,
                faces,
                pose,
                xlim,
                ylim,
                resolution,
                samples_per_edge=samples_per_edge,
            )
            grasp_access = None
            grasp_n = 0
            grasp_score = 0.0
            if grasp_evaluator is not None:
                grasp_access, grasp_n, grasp_score = grasp_evaluator.score(
                    action.stone_idx, pose
                )
                if not grasp_access and grasp_mcts_hard_filter:
                    self._mark_grasp_failed(child, grasp_n, grasp_score)
                else:
                    self._mark_grasp_access_checked(child, grasp_n, grasp_score)

            label = float(child.is_simulated and not child.failed)
            child_info = child.info or {}
            failure_reason = child_info.get(
                "simulation_failure_reason",
                child_info.get("mcts_unselectable_reason"),
            )
            local_point_count = int(
                self.mcts.env.cfg.get("observation", {}).get(
                    "local_scene_dsf_points", 256
                )
            )
            local_points, local_normals, local_mask = (
                inventory.get_local_scene_dsf_samples(
                    parent.state, pose, local_point_count
                )
            )
            row = {
                "heightmaps": candidate_heightmap_stack(
                    scene_height,
                    target_height,
                    bottom,
                    top,
                    contact_eps=contact_eps,
                    stone_mask=mask,
                ),
                "label": label,
                "weight": positive_weight if label > 0.5 else negative_weight,
                "depth": max(len(parent.state.stone_seq) - root_stone_count, 0),
                "stone_idx": int(action.stone_idx),
                "stone_id": int(action.stone_id),
                "candidate_pose": np.asarray(pose, dtype=np.float32),
                "candidate_dsf_points": inventory.dsf_points[action.stone_idx].astype(
                    np.float16
                ),
                "candidate_dsf_normals": inventory.dsf_normals[action.stone_idx].astype(
                    np.float16
                ),
                "candidate_dsf_point_mask": inventory.dsf_point_mask[
                    action.stone_idx
                ].copy(),
                "candidate_physical_features": inventory.stone_physical_features[
                    action.stone_idx
                ].copy(),
                "local_scene_dsf_points": local_points.astype(np.float16),
                "local_scene_dsf_normals": local_normals.astype(np.float16),
                "local_scene_dsf_point_mask": local_mask,
                "scene_stone_ids": scene_stone_ids,
                "scene_stone_poses": scene_stone_poses,
                "scene_mask": scene_mask,
                "failed": bool(child.failed),
                "reward": float(child.reward if child.reward is not None else 0.0),
                "q_value_init": float(child.q_value_init),
                "c_feq": float(action.c_feq),
                "c_gap": float(action.c_gap),
                "_parent_state": parent.state,
                "_action": action.copy(),
                "_failure_reason": failure_reason,
                "_simulation_type": child_info.get("simulation_type", "short"),
            }
            if grasp_evaluator is not None:
                row["grasp_label"] = float(grasp_access)
                row["grasp_n"] = int(grasp_n)
                row["grasp_score"] = float(grasp_score)
            rows.append(row)
        return rows

    def _mark_grasp_failed(
        self,
        child: MCTS_Node,
        n_grasps: int,
        grasp_score: float,
    ) -> None:
        child.set_failed(True)
        child.q_value_init = -np.inf
        child.q_value = -np.inf
        if child.reward is None:
            child.reward = -1.0
        if child.state is not None:
            child.state.failed = True

        info = dict(child.info or {})
        info.update(
            {
                "grasp_access_failed": True,
                "grasp_access_checked": True,
                "grasp_failure_reason": "no_feasible_grasp",
                "grasp_n": int(n_grasps),
                "grasp_score": float(grasp_score),
            }
        )
        child.info = info

    def _mark_grasp_access_checked(
        self,
        child: MCTS_Node,
        n_grasps: int,
        grasp_score: float,
    ) -> None:
        info = dict(child.info or {})
        info.update(
            {
                "grasp_access_failed": False,
                "grasp_access_checked": True,
                "grasp_n": int(n_grasps),
                "grasp_score": float(grasp_score),
            }
        )
        child.info = info


class SamplerConfig:
    def __init__(
        self,
        env_class: gym.Env,
        env_args: Dict[str, OmegaConf] = None,
        mcts_cfg: OmegaConf = None,
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: int = 0,
        num_workers: int = 1,
    ):
        self.env_class = env_class
        self.env_args = env_args
        self.mcts_cfg = mcts_cfg
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = num_gpus_per_worker
        self.num_workers = num_workers

    def create_samplers(self) -> List[MCTS_Sampler]:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        return [
            MCTS_Sampler.options(
                num_cpus=self.num_cpus_per_worker,
                num_gpus=self.num_gpus_per_worker,
            ).remote(self.env_class, self.env_args, self.mcts_cfg, id)
            for id in range(self.num_workers)
        ]
