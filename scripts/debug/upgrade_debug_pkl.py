#!/usr/bin/env python3
"""
Re-extract stone meshes in an existing debug_state.pkl using StoneGeometry.get_mesh()
(full-resolution DSF) instead of the lowpoly arrays that were saved originally.

The run directory must contain a config.yml alongside the pkl so the environment
can be re-initialised to access the stone objects.

Usage:
    python scripts/debug/upgrade_debug_pkl.py <debug_state.pkl>
    python scripts/debug/upgrade_debug_pkl.py .debug/mcts/<run>/debug_state.pkl
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.env import StoneStackingEnv


def _extract_hires_meshes(env: StoneStackingEnv) -> dict:
    stone_meshes = {}
    for idx in range(len(env.inventory.stone_set)):
        stone = env.inventory.stones[idx]
        all_v, all_t, offset = [], [], 0
        for geom in stone.geometries:
            m = geom.get_mesh()
            v = np.asarray(m.vertices, dtype=float)
            t = np.asarray(m.triangles, dtype=int)
            all_v.append(v)
            all_t.append(t + offset)
            offset += len(v)
        stone_meshes[int(stone.id)] = (
            np.concatenate(all_v, axis=0),
            np.concatenate(all_t, axis=0),
        )
    return stone_meshes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl", help="path to debug_state.pkl")
    args = parser.parse_args()

    pkl_path = Path(args.pkl).resolve()
    cfg_path = pkl_path.parent / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yml not found next to pkl: {cfg_path}")

    print(f"loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"loading config from {cfg_path} ...")
    cfg = OmegaConf.load(cfg_path)

    print("initialising environment to access stone geometries ...")
    env = StoneStackingEnv({"cfg": cfg.environment, "n_threads": 1})
    env.reset()

    print("re-extracting stone meshes with StoneGeometry.get_mesh() ...")
    data["stone_meshes"] = _extract_hires_meshes(env)

    print(f"saving upgraded pkl to {pkl_path} ...")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    print(f"done — {len(data['stone_meshes'])} stone meshes upgraded.")


if __name__ == "__main__":
    main()
