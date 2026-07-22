"""GNN-side offline-RL dataset.

Faithful port of `stacking-RL/loader/loader.py:RLDataset`, consuming the h5
files emitted by `agent.sampler.MCTS_Sampler.sample_episode`. Each row is a
(state, action, reward, next_state, action_samples, next_action_samples,
q_mcts, pi_mcts, done, failed, n_step) tuple; per-row we subsample the top-K
candidate actions ranked by `pi_mcts` (with tiny noise for tie-breaking) so
that downstream batching has a fixed action axis.
"""

import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
from torch.utils import data

from utils.data_handle import convert_h5py_group_to_dict


class RLDataset(data.Dataset):
    def __init__(self, path: str, n_action_samples: int = 16):
        dataset = {}
        with h5py.File(path, "r") as file:
            for key, val in file.items():
                if isinstance(val, h5py.Group):
                    dataset[key] = convert_h5py_group_to_dict(val)
                else:
                    dataset[key] = val[()]

        rewards = dataset["reward"]
        failed = dataset.get("failed", np.zeros_like(rewards))
        if "failed" not in dataset:
            dataset["failed"] = failed

        self.dataset = dataset
        self.n_action_samples = int(n_action_samples)
        self.len = len(rewards)
        self.mean_reward = float(np.mean(rewards))
        self.max_reward = float(np.max(rewards))
        self.min_reward = float(np.min(rewards))
        print(f"[RLDataset] {path}: length={self.len}")
        print(
            f"[RLDataset] reward stats — mean: {self.mean_reward:.4f}, "
            f"max: {self.max_reward:.4f}, min: {self.min_reward:.4f}, "
            f"var: {float(np.var(rewards)):.4f}"
        )
        print(f"[RLDataset] # failures: {int(np.count_nonzero(failed))}")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):
        sample = self._index_into(self.dataset, idx)

        # Top-K subsample by pi_mcts; tiny noise breaks ties.
        pi_mcts = sample["pi_mcts"]
        k = min(self.n_action_samples, len(pi_mcts))
        noise = np.random.rand(len(pi_mcts)) * 1e-6
        sel = np.argsort(pi_mcts + noise)[-k:]

        action_samples = sample["action_samples"]
        for key, val in action_samples.items():
            action_samples[key] = val[sel]
        sample["action_samples"] = action_samples
        sample["pi_mcts"] = sample["pi_mcts"][sel]
        sample["q_mcts"] = sample["q_mcts"][sel]

        # action_mask is per-candidate; align with the subsampled set.
        if "action_mask" in sample.get("state", {}):
            sample["state"]["action_mask"] = sample["state"]["action_mask"][sel]
        if "action_mask" in sample.get("next_state", {}):
            # Next-state mask is shared shape with current; subsample identically
            # (the planner pads to a fixed candidate axis per row).
            sample["next_state"]["action_mask"] = sample["next_state"]["action_mask"][sel]
        return sample

    @classmethod
    def _index_into(cls, data, idx):
        if isinstance(data, dict):
            return {k: cls._index_into(v, idx) for k, v in data.items()}
        return data[idx]


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


def get_rl_dataloaders(
    data_dir: str,
    batch_size: int,
    val_frac: float = 0.1,
    seed: int = 0,
    num_workers: int = 0,
    n_action_samples: int = 16,
) -> Tuple[Optional[data.DataLoader], Optional[data.DataLoader]]:
    """Build train/val DataLoaders. Mirrors `loader.get_heightmap_dataloaders`."""
    import torch

    paths = resolve_h5_paths(data_dir)
    if not paths:
        return None, None
    parts = [RLDataset(p, n_action_samples=n_action_samples) for p in paths]
    dataset = parts[0] if len(parts) == 1 else data.ConcatDataset(parts)
    if val_frac > 0 and len(dataset) > 1:
        n_val = max(1, int(round(len(dataset) * val_frac)))
        n_train = len(dataset) - n_val
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = data.random_split(
            dataset, [n_train, n_val], generator=gen
        )
        train_loader = data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader
    return (
        data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        None,
    )
