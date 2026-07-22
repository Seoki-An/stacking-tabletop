"""Eager-loaded heightmap dataset and grid helpers.

Mirrors stacking-RL/loader, but specialized for the height-map Q-network.
Reads merged h5 files produced by `agent.sampler.MCTS_Sampler.sample_episode`.

The dataset is self-contained: every sample carries the `height_map_xlim` /
`height_map_ylim` actually used to render its scene/target height maps (the
target wall size is randomized per episode in sampling, so the ortho-camera
bounds vary too). Binning is done per-row against those saved bounds — no
environment config is needed at training time.

Multistep mixing: a `MixedBatchSampler` guarantees a per-batch fraction of
multistep (Monte-Carlo target) rows. Multistep rows are identified by the source file suffix `_multi.h5` written by
`MCTS_Sampler.sample_episode(..., save_multistep=True)` and through
`stream_merge_h5` (which preserves the suffix). They anchor Q magnitudes
against committed-episode returns and prevent the bootstrap-only path from
running away during offline RL training. Horizon-one committed transitions are
kept only in the one-step file and are not duplicated in the multistep file.
"""

import os
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Sampler,
    Subset,
    random_split,
)


def xy_to_bin(
    xy: np.ndarray,
    xlim: Union[Tuple[float, float], np.ndarray],
    ylim: Union[Tuple[float, float], np.ndarray],
    h_out: int,
    w_out: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """World-frame (x, y) → (row, col) on the score-map grid.

    `xy` shape: (..., 2) — typically (B, 2) or (B, A, 2).
    `xlim`, `ylim` may be a length-2 tuple/array (shared bounds for all rows)
    or a (B, 2) array of per-row bounds matching `xy`'s leading dim.
    """
    xlim_arr = np.asarray(xlim, dtype=np.float64)
    ylim_arr = np.asarray(ylim, dtype=np.float64)

    if xlim_arr.ndim == 2:
        # Per-row bounds: broadcast (B,) against xy[..., 0] which is
        # (B, ...). Reshape so trailing action / sample dims broadcast.
        if xlim_arr.shape[0] != xy.shape[0]:
            raise ValueError(
                f"xy_to_bin: per-row xlim has B={xlim_arr.shape[0]} but xy has "
                f"B={xy.shape[0]}"
            )
        bcast_shape = (xy.shape[0],) + (1,) * (xy.ndim - 2)
        xlim_lo = xlim_arr[:, 0].reshape(bcast_shape)
        xlim_hi = xlim_arr[:, 1].reshape(bcast_shape)
        ylim_lo = ylim_arr[:, 0].reshape(bcast_shape)
        ylim_hi = ylim_arr[:, 1].reshape(bcast_shape)
    else:
        xlim_lo, xlim_hi = float(xlim_arr[0]), float(xlim_arr[1])
        ylim_lo, ylim_hi = float(ylim_arr[0]), float(ylim_arr[1])

    j = np.round((xy[..., 0] - xlim_lo) / (xlim_hi - xlim_lo) * (w_out - 1))
    i = np.round((xy[..., 1] - ylim_lo) / (ylim_hi - ylim_lo) * (h_out - 1))
    return (
        np.clip(i.astype(np.int64), 0, h_out - 1),
        np.clip(j.astype(np.int64), 0, w_out - 1),
    )


def _is_multistep_path(path: str) -> bool:
    """Heuristic: filename contains a `_multi` / `_multistep` token, ending
    in `.h5`. Catches both `MCTS_Sampler.sample_episode(save_multistep=True)`
    raw output (`actor_*_ep*_multi.h5`) and any `stream_merge_h5` /
    rename-style variants that preserve the token (`*_multi_v2.h5`,
    `train_multistep.h5`, `train_multistep_2.h5`, …)."""
    name = os.path.basename(path).lower()
    if not name.endswith(".h5"):
        return False
    return any(
        tok in name
        for tok in ("_multi.", "_multi_", "_multistep.", "_multistep_")
    )


class HeightmapH5Dataset(Dataset):
    """Eager-loaded dataset from a merged sample h5 file.

    Reads per-sample `height_map_xlim` / `height_map_ylim` written by the
    `Observation` builder and uses them to bin actions onto the score-map
    grid. The dataset therefore needs no environment config to interpret
    the underlying samples.

    `is_multistep` tags whether this shard contains committed-trajectory
    reward-to-go rows (filename suffix `_multi.h5`). The
    `MixedBatchSampler` reads this to enforce a per-batch MC-vs-bootstrap ratio.
    """

    def __init__(
        self,
        h5_path: str,
        output_resolution: Tuple[int, int],
    ):
        print(f"Loading dataset from {h5_path}...")
        self.is_multistep: bool = _is_multistep_path(h5_path)
        with h5py.File(h5_path, "r") as f:
            scene_hm = f["state/scene_height_map"][:]
            target_hm = f["state/target_height_map"][:]
            next_scene = f["next_state/scene_height_map"][:]
            next_target = f["next_state/target_height_map"][:]
            init_pose = f["action/init_pose"][:]
            cur_samples_pose = f["action_samples/init_pose"][:]
            next_init_pose = f["next_action_samples/init_pose"][:]
            pi_mcts = f["pi_mcts"][:]
            reward = f["reward"][:]
            done = f["done"][:]
            n_step = f["n_step"][:]
            failed = f["failed"][:] if "failed" in f else np.zeros_like(done)

            # Per-row height-map bounds (added to `Observation` so binning is
            # self-contained). Required: refuse to silently fall back to a
            # global default that would silently corrupt randomized episodes.
            missing = [
                k
                for k in ("height_map_xlim", "height_map_ylim")
                if k not in f["state"] or k not in f["next_state"]
            ]
            if missing:
                raise KeyError(
                    f"{h5_path}: missing height-map bounds in sample "
                    f"({missing}). Re-sample with the updated Observation "
                    f"that persists `height_map_xlim` / `height_map_ylim`."
                )
            xlim = f["state/height_map_xlim"][:]
            ylim = f["state/height_map_ylim"][:]
            next_xlim = f["next_state/height_map_xlim"][:]
            next_ylim = f["next_state/height_map_ylim"][:]

        # Drop rows containing NaN/Inf in any field that feeds the loss.
        # Bad rows can come from a divergent diffsim step that produced
        # corrupt poses → corrupt observations → corrupt rewards.
        def _row_finite(arr: np.ndarray) -> np.ndarray:
            flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr[:, None]
            return np.all(np.isfinite(flat), axis=1)

        mask = (
            _row_finite(reward)
            & _row_finite(scene_hm)
            & _row_finite(target_hm)
            & _row_finite(next_scene)
            & _row_finite(next_target)
            & _row_finite(init_pose)
            & _row_finite(cur_samples_pose)
            & _row_finite(next_init_pose)
            & _row_finite(pi_mcts)
            & _row_finite(xlim)
            & _row_finite(ylim)
            & _row_finite(next_xlim)
            & _row_finite(next_ylim)
        )
        n_total = int(mask.shape[0])
        n_keep = int(mask.sum())
        if n_keep < n_total:
            print(
                f"HeightmapH5Dataset: dropped {n_total - n_keep} / {n_total} "
                f"rows with NaN/Inf in {h5_path}"
            )
        scene_hm = scene_hm[mask]
        target_hm = target_hm[mask]
        next_scene = next_scene[mask]
        next_target = next_target[mask]
        init_pose = init_pose[mask]
        cur_samples_pose = cur_samples_pose[mask]
        next_init_pose = next_init_pose[mask]
        pi_mcts = pi_mcts[mask]
        reward = reward[mask]
        done = done[mask]
        n_step = n_step[mask]
        failed = failed[mask]
        xlim = xlim[mask]
        ylim = ylim[mask]
        next_xlim = next_xlim[mask]
        next_ylim = next_ylim[mask]

        h_out, w_out = output_resolution
        bin_i, bin_j = xy_to_bin(init_pose[:, :2], xlim, ylim, h_out, w_out)
        # cur_samples_pose: (B, n_actions, 7) — bins for the in-sample rank loss
        cur_bin_i, cur_bin_j = xy_to_bin(
            cur_samples_pose[..., :2], xlim, ylim, h_out, w_out
        )
        # next_init_pose: (B, n_actions, 7) — bins for the in-sample TD bootstrap.
        # Use the *next* state's bounds, since next-state heightmaps are rendered
        # against them.
        next_bin_i, next_bin_j = xy_to_bin(
            next_init_pose[..., :2], next_xlim, next_ylim, h_out, w_out
        )
        self.h_out = h_out
        self.w_out = w_out

        self.scene_hm = scene_hm.astype(np.float32)
        self.target_hm = target_hm.astype(np.float32)
        self.next_scene = next_scene.astype(np.float32)
        self.next_target = next_target.astype(np.float32)
        self.bin_i = bin_i
        self.bin_j = bin_j
        self.cur_bin_i = cur_bin_i
        self.cur_bin_j = cur_bin_j
        self.next_bin_i = next_bin_i
        self.next_bin_j = next_bin_j
        self.pi_mcts = pi_mcts.astype(np.float32)
        self.reward = reward.astype(np.float32)
        self.done = done.astype(np.float32)
        self.n_step = n_step.astype(np.float32)
        self.failed = failed.astype(np.float32)

    def __len__(self) -> int:
        return self.scene_hm.shape[0]

    def __getitem__(self, idx: int):
        return {
            "scene_hm": torch.from_numpy(self.scene_hm[idx]),
            "target_hm": torch.from_numpy(self.target_hm[idx]),
            "next_scene": torch.from_numpy(self.next_scene[idx]),
            "next_target": torch.from_numpy(self.next_target[idx]),
            "bin_i": torch.tensor(self.bin_i[idx], dtype=torch.long),
            "bin_j": torch.tensor(self.bin_j[idx], dtype=torch.long),
            "cur_bin_i": torch.from_numpy(self.cur_bin_i[idx]).long(),
            "cur_bin_j": torch.from_numpy(self.cur_bin_j[idx]).long(),
            "next_bin_i": torch.from_numpy(self.next_bin_i[idx]).long(),
            "next_bin_j": torch.from_numpy(self.next_bin_j[idx]).long(),
            "pi_mcts": torch.from_numpy(self.pi_mcts[idx]).float(),
            "reward": torch.tensor(self.reward[idx], dtype=torch.float32),
            "done": torch.tensor(self.done[idx], dtype=torch.float32),
            "n_step": torch.tensor(self.n_step[idx], dtype=torch.float32),
            "failed": torch.tensor(self.failed[idx], dtype=torch.float32),
        }


def resolve_h5_paths(path: str) -> List[str]:
    """Single .h5 file → [path]; directory → recursive .h5 scan; else []."""
    if path.endswith(".h5"):
        return [path] if os.path.isfile(path) else []
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(root, fn)
        for root, _, files in os.walk(path)
        for fn in files
        if fn.endswith(".h5")
    )


class MixedBatchSampler(Sampler[List[int]]):
    """Yields batches that always contain a fixed split between two disjoint
    index pools (multistep vs 1-step).

    Per batch:
      * `n_multi = round(batch_size * multi_ratio)` indices drawn from the
        multistep pool, and
      * `n_one  = batch_size - n_multi` indices drawn from the 1-step pool.

    Each epoch reshuffles both pools independently. The number of batches per
    epoch is determined by the *larger* pool; the smaller pool wraps with
    fresh shuffles so its rows are revisited (necessary when the pools are
    unbalanced, which is the common case — multistep is rarer than 1-step).

    `Sampler[List[int]]` rather than `Sampler[int]`: DataLoader picks this up
    as the `batch_sampler`, bypassing its own batching logic.
    """

    def __init__(
        self,
        multi_indices: Sequence[int],
        one_indices: Sequence[int],
        batch_size: int,
        multi_ratio: float,
        seed: int = 0,
    ):
        if not 0.0 < multi_ratio < 1.0:
            # 0.0 and 1.0 collapse to a single-pool sampler; the caller should
            # have detected that and used plain shuffling instead.
            raise ValueError(
                f"MixedBatchSampler: multi_ratio must be in (0, 1); got {multi_ratio}"
            )
        n_multi = int(round(batch_size * multi_ratio))
        n_one = batch_size - n_multi
        if n_multi == 0 or n_one == 0:
            raise ValueError(
                f"MixedBatchSampler: batch_size={batch_size} with "
                f"multi_ratio={multi_ratio} yields a degenerate split "
                f"(n_multi={n_multi}, n_one={n_one}). Increase batch_size."
            )
        if len(multi_indices) == 0 or len(one_indices) == 0:
            raise ValueError(
                "MixedBatchSampler: both pools must be non-empty "
                f"(got |multi|={len(multi_indices)}, |one|={len(one_indices)})"
            )

        self.multi = np.asarray(multi_indices, dtype=np.int64)
        self.one = np.asarray(one_indices, dtype=np.int64)
        self.batch_size = batch_size
        self.n_multi = n_multi
        self.n_one = n_one
        self.seed = int(seed)
        self._epoch = 0

        # One epoch's length is dictated by the *larger* per-pool budget so
        # neither pool is starved if the other is much larger. The smaller
        # pool gets reshuffled within the epoch.
        n_batches_multi = (len(self.multi) + self.n_multi - 1) // self.n_multi
        n_batches_one = (len(self.one) + self.n_one - 1) // self.n_one
        self._n_batches = max(n_batches_multi, n_batches_one)

    @staticmethod
    def _shuffled_to_length(
        pool: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return `n` indices drawn from `pool`. If `n > len(pool)`, tile by
        concatenating fresh shuffles so every block of `len(pool)` is a
        full permutation (no sample is reused within one cycle)."""
        if n <= len(pool):
            return rng.permutation(pool)[:n]
        blocks = []
        produced = 0
        while produced < n:
            blocks.append(rng.permutation(pool))
            produced += len(pool)
        return np.concatenate(blocks)[:n]

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        multi_perm = self._shuffled_to_length(
            self.multi, self._n_batches * self.n_multi, rng
        )
        one_perm = self._shuffled_to_length(
            self.one, self._n_batches * self.n_one, rng
        )
        for i in range(self._n_batches):
            batch = np.concatenate(
                [
                    multi_perm[i * self.n_multi : (i + 1) * self.n_multi],
                    one_perm[i * self.n_one : (i + 1) * self.n_one],
                ]
            )
            yield batch.tolist()

    def __len__(self) -> int:
        return self._n_batches


def _split_indices_by_part(
    parts: Sequence[HeightmapH5Dataset],
    subset_indices: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Given a `ConcatDataset(parts)` and a list of indices into it (e.g.
    from `random_split` → `Subset.indices`), return `(multi_subset_idx,
    one_subset_idx)`: indices **into the Subset** (i.e. positions in
    `subset_indices`) that point at multistep vs 1-step rows.

    DataLoader sees the Subset as its dataset; sampler indices are Subset-
    local, not global. We translate global→Subset by inverting the indices.
    """
    cum = np.cumsum([len(p) for p in parts])  # exclusive on the left
    # For each global index, find its part by binary search.
    global_idx = np.asarray(subset_indices, dtype=np.int64)
    part_of = np.searchsorted(cum, global_idx, side="right")
    multi_mask = np.array(
        [parts[p].is_multistep for p in part_of], dtype=bool
    )
    subset_pos = np.arange(len(global_idx), dtype=np.int64)
    return (
        subset_pos[multi_mask].tolist(),
        subset_pos[~multi_mask].tolist(),
    )


def get_heightmap_dataloaders(
    data_dir: str,
    output_resolution: Tuple[int, int],
    batch_size: int,
    val_frac: float = 0.1,
    seed: int = 0,
    num_workers: int = 0,
    multi_ratio: Optional[float] = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Build train/val DataLoaders from a single file or directory of h5 shards.

    `multi_ratio`: if set in (0, 1), enforces a per-batch fraction of
    multistep rows via `MixedBatchSampler` (Monte-Carlo target rows anchor
    Q magnitudes against the bootstrap-driven 1-step rows). Falls back to
    plain uniform shuffling when `multi_ratio` is None, 0, 1, or one of the
    two pools is empty.
    """
    h5_paths = resolve_h5_paths(data_dir)
    if not h5_paths:
        return None, None
    # Skip corrupt files with a warning rather than aborting the whole run.
    # Bad files can come from interrupted writes / disk errors on long
    # sampling jobs; re-sampling them is the fix, but we'd like training on
    # the surviving shards to proceed in the meantime.
    parts: List[HeightmapH5Dataset] = []
    skipped: List[Tuple[str, str]] = []
    for p in h5_paths:
        try:
            parts.append(HeightmapH5Dataset(p, output_resolution))
        except (OSError, KeyError, ValueError) as exc:
            skipped.append((p, type(exc).__name__ + ": " + str(exc)))
            print(f"[HeightmapH5Dataset] WARNING: skipping {p} ({exc!r})")
    if not parts:
        raise FileNotFoundError(
            f"all .h5 files under {data_dir} failed to load:\n"
            + "\n".join(f"  {p}: {err}" for p, err in skipped)
        )
    if skipped:
        print(
            f"[HeightmapH5Dataset] skipped {len(skipped)} of {len(h5_paths)} "
            f"file(s) due to load errors; re-sample to recover:"
        )
        for p, err in skipped:
            print(f"  - {p}: {err}")
    dataset: Dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)

    n_multi = sum(len(p) for p in parts if p.is_multistep)
    n_one = sum(len(p) for p in parts if not p.is_multistep)
    print(
        f"Heightmap dataset: {n_multi + n_one} rows "
        f"({n_multi} multistep, {n_one} 1-step) across {len(parts)} file(s)"
    )

    if val_frac > 0 and len(dataset) > 1:
        n_val = max(1, int(round(len(dataset) * val_frac)))
        n_train = len(dataset) - n_val
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
        train_loader = _build_train_loader(
            train_ds=train_ds,
            parts=parts,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            multi_ratio=multi_ratio,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return train_loader, val_loader

    train_loader = _build_train_loader(
        train_ds=dataset,
        parts=parts,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        multi_ratio=multi_ratio,
    )
    return train_loader, None


def _build_train_loader(
    train_ds: Dataset,
    parts: Sequence[HeightmapH5Dataset],
    batch_size: int,
    num_workers: int,
    seed: int,
    multi_ratio: Optional[float],
) -> DataLoader:
    """Plain shuffled loader when no ratio is requested, else a
    `MixedBatchSampler`-backed loader with a hard per-batch multi/1-step split."""
    if multi_ratio is None or multi_ratio <= 0 or multi_ratio >= 1:
        return DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    if isinstance(train_ds, Subset):
        subset_indices = list(train_ds.indices)
    else:
        # Whole-dataset training (no val split). Sampler indexes the dataset
        # directly; build a Subset-equivalent index list of length len(ds).
        subset_indices = list(range(len(train_ds)))

    multi_pos, one_pos = _split_indices_by_part(parts, subset_indices)
    if not multi_pos or not one_pos:
        # One of the two pools is empty after the split — falling back to
        # uniform shuffling rather than crash. Common when the user only
        # has 1-step data, or asked for ratio mixing on a multistep-only set.
        print(
            f"Heightmap dataloader: multi_ratio={multi_ratio} requested but "
            f"|multi|={len(multi_pos)}, |one|={len(one_pos)} after train/val "
            f"split — using plain uniform shuffling."
        )
        return DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    batch_sampler = MixedBatchSampler(
        multi_indices=multi_pos,
        one_indices=one_pos,
        batch_size=batch_size,
        multi_ratio=multi_ratio,
        seed=seed,
    )
    print(
        f"Heightmap dataloader: MixedBatchSampler "
        f"(multi/one per batch = {batch_sampler.n_multi}/{batch_sampler.n_one}, "
        f"{batch_sampler._n_batches} batches/epoch)"
    )
    return DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
    )
