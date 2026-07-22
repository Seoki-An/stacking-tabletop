from __future__ import annotations

from typing import Optional

import numpy as np


def inward_xy(inventory, xy: np.ndarray) -> np.ndarray:
    origin = np.asarray(inventory.target_wall.origin[:2], dtype=float)
    inward = origin - np.asarray(xy[:2], dtype=float)
    norm = np.linalg.norm(inward)
    if norm < 1e-9:
        return inward
    return inward / norm


def top_exposed_surface_normal(
    world_normals: np.ndarray,
    areas: np.ndarray,
    min_z: float = 0.0,
) -> Optional[np.ndarray]:
    candidate_mask = world_normals[:, 2] >= min_z
    if not np.any(candidate_mask):
        return None

    candidate_indices = np.flatnonzero(candidate_mask)
    weights = np.asarray(areas[candidate_indices], dtype=float)
    if len(weights) != len(candidate_indices) or float(np.sum(weights)) <= 1e-12:
        weights = np.ones(len(candidate_indices), dtype=float)
    normal = np.average(world_normals[candidate_indices], axis=0, weights=weights)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    return normal / norm
