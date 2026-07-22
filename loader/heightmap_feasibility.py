from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


_OPTIONAL_INPUT_KEYS = (
    "candidate_pose",
    "candidate_dsf_points",
    "candidate_dsf_normals",
    "candidate_dsf_point_mask",
    "local_scene_dsf_points",
    "local_scene_dsf_normals",
    "local_scene_dsf_point_mask",
    "candidate_physical_features",
    "c_feq",
    "c_gap",
    "depth",
    "scene_stone_ids",
    "scene_stone_poses",
    "scene_mask",
    "failure_reason_id",
    "failure_reason_counts",
    "nominal_label",
    "stability_passes",
    "stability_rollouts",
)

FEASIBILITY_FAILURE_REASONS = (
    "pass",
    "target_motion_limit",
    "scene_motion_limit",
    "place_robustness",
    "nonfinite_simulation",
    "simulation_unsettled",
    "other",
)
FEASIBILITY_FAILURE_OTHER_ID = len(FEASIBILITY_FAILURE_REASONS) - 1


def feasibility_failure_reason_id(reason, failed: bool = False) -> int:
    if not failed:
        return 0
    try:
        return FEASIBILITY_FAILURE_REASONS.index(str(reason))
    except ValueError:
        return FEASIBILITY_FAILURE_OTHER_ID



def candidate_heightmap_stack(
    scene_height: np.ndarray,
    target_height: np.ndarray,
    stone_bottom: np.ndarray,
    stone_top: np.ndarray,
    contact_eps: float = 0.03,
    stone_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a float16 candidate tensor for feasibility H5 storage."""
    scene = np.asarray(scene_height, dtype=np.float32)
    target = np.asarray(target_height, dtype=np.float32)
    bottom = np.asarray(stone_bottom, dtype=np.float32)
    top = np.asarray(stone_top, dtype=np.float32)
    if stone_mask is None:
        footprint = np.abs(top - bottom) > 1e-6
    else:
        footprint = np.asarray(stone_mask, dtype=bool)

    bottom = np.where(footprint, bottom, scene)
    top = np.where(footprint, top, scene)

    target_minus_scene = np.maximum(target - scene, 0.0)
    clearance = np.where(footprint, bottom - scene, 0.0)
    penetration = np.where(footprint, np.maximum(scene - bottom, 0.0), 0.0)
    contact_band = (
        footprint & (np.abs(clearance) <= float(contact_eps))
    ).astype(np.float32)

    return np.stack(
        [
            scene,
            target,
            target_minus_scene,
            bottom,
            top,
            clearance,
            penetration,
            contact_band,
        ],
        axis=0,
    ).astype(np.float16)


def resize_heightmap(heightmap: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize for small training tensors without extra deps."""
    heightmap = np.asarray(heightmap, dtype=np.float32)
    out_h, out_w = int(resolution[0]), int(resolution[1])
    if heightmap.shape == (out_h, out_w):
        return heightmap.copy()
    y_idx = np.linspace(0, heightmap.shape[0] - 1, out_h).round().astype(int)
    x_idx = np.linspace(0, heightmap.shape[1] - 1, out_w).round().astype(int)
    return heightmap[np.ix_(y_idx, x_idx)].astype(np.float32)


def stone_heightmaps_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    pose: np.ndarray,
    xlim: np.ndarray,
    ylim: np.ndarray,
    resolution: tuple[int, int],
    samples_per_edge: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize a posed stone mesh into bottom/top/footprint heightmaps."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=int)
    pose = np.asarray(pose, dtype=float)
    resolution = (int(resolution[0]), int(resolution[1]))

    bottom = np.zeros(resolution, dtype=np.float32)
    top = np.zeros(resolution, dtype=np.float32)
    mask = np.zeros(resolution, dtype=bool)

    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        return bottom, top, mask
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return bottom, top, mask
    vertices, faces = _drop_padded_vertices(vertices, faces)
    if len(vertices) == 0:
        return bottom, top, mask

    quat = pose[3:]
    if np.linalg.norm(quat) <= 1e-12:
        return bottom, top, mask

    rot = Rotation.from_quat(pose[3:]).as_matrix()
    world_vertices = vertices @ rot.T + pose[:3]
    samples = _sample_mesh_points(world_vertices, faces, samples_per_edge)
    if len(samples) == 0:
        return bottom, top, mask
    finite = np.all(np.isfinite(samples), axis=1)
    samples = samples[finite]
    if len(samples) == 0:
        return bottom, top, mask

    xlim = np.asarray(xlim, dtype=float)
    ylim = np.asarray(ylim, dtype=float)
    if xlim.shape != (2,) or ylim.shape != (2,):
        return bottom, top, mask
    if not (np.all(np.isfinite(xlim)) and np.all(np.isfinite(ylim))):
        return bottom, top, mask
    x_span = max(float(xlim[1] - xlim[0]), 1e-6)
    y_span = max(float(ylim[1] - ylim[0]), 1e-6)
    xs = ((samples[:, 0] - xlim[0]) / x_span * (resolution[1] - 1)).round().astype(int)
    ys = ((samples[:, 1] - ylim[0]) / y_span * (resolution[0] - 1)).round().astype(int)
    inside = (
        (xs >= 0)
        & (xs < resolution[1])
        & (ys >= 0)
        & (ys < resolution[0])
        & np.isfinite(samples[:, 2])
    )
    xs = xs[inside]
    ys = ys[inside]
    zs = samples[inside, 2].astype(np.float32)
    for x, y, z in zip(xs, ys, zs):
        if not mask[y, x]:
            bottom[y, x] = z
            top[y, x] = z
            mask[y, x] = True
        else:
            bottom[y, x] = min(bottom[y, x], z)
            top[y, x] = max(top[y, x], z)
    return bottom, top, mask


def _drop_padded_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_vertices = np.all(np.isfinite(vertices), axis=1)
    if not np.any(finite_vertices):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=int)

    index_map = -np.ones(len(vertices), dtype=int)
    index_map[finite_vertices] = np.arange(int(finite_vertices.sum()))

    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        valid_faces = np.empty((0, 3), dtype=int)
    else:
        in_range = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
        valid_faces = faces[in_range]
        if len(valid_faces) > 0:
            valid_faces = valid_faces[
                np.all(finite_vertices[valid_faces], axis=1)
            ]
            valid_faces = index_map[valid_faces]

    return vertices[finite_vertices], valid_faces


def _sample_mesh_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    samples_per_edge: int,
) -> np.ndarray:
    valid_faces = faces[np.all(faces >= 0, axis=1)] if len(faces) else faces
    if len(valid_faces) == 0:
        return vertices

    n = max(int(samples_per_edge), 1)
    points = [vertices]
    tri = vertices[valid_faces]
    bary = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            a = i / n
            b = j / n
            c = 1.0 - a - b
            bary.append((a, b, c))
    weights = np.asarray(bary, dtype=np.float32)
    sampled = (
        weights[:, 0, None, None] * tri[None, :, 0]
        + weights[:, 1, None, None] * tri[None, :, 1]
        + weights[:, 2, None, None] * tri[None, :, 2]
    )
    points.append(sampled.reshape(-1, 3))
    return np.concatenate(points, axis=0)


class HeightmapFeasibilityDataset(Dataset):
    """Hybrid heightmap and optional DSF-geometry feasibility dataset.

    The preferred integrated format is one HDF5 file, usually
    `<sample-dir>/feasibility/feasibility.h5`. Legacy `.npz` shard directories
    are still accepted.

    Each file is expected to contain:
      - `heightmaps`: (N, C, H, W) float32
      - `label`: (N,) or (N, 1), where 1 means feasible/pass

    Optional geometry inputs:
      - candidate DSF points, support-direction normals, mask, pose, and
        physical features
      - nearby scene DSF points, normals, and mask in the candidate frame
      - compact scene stone IDs, poses, and mask

    Optional labels:
      - `weight`: per-row loss weight
      - `grasp_label`: (N,) grasp-accessibility target (1 = a grasp was found),
        present only when the sampler ran with `feasibility.grasp_access` enabled.

    `label_key` selects the supervised target column:
      - `"label"` (default): simulation feasibility.
      - `"grasp_label"`: grasp accessibility.
      - `"combined"`: `label * grasp_label` (must pass both gates).
      - `"grasp_count"`: soft target from normalized `grasp_n`.
      - `"combined_count"`: `label * grasp_count`.
    """

    def __init__(
        self,
        path: str,
        label_key: str = "label",
        grasp_count_normalizer: float = 32.0,
        cache_heightmaps: bool = False,
    ):
        self.paths = _resolve_feasibility_paths(path)
        if not self.paths:
            raise FileNotFoundError(f"no feasibility files found at {path}")
        self.label_key = label_key
        self.grasp_count_normalizer = max(float(grasp_count_normalizer), 1.0)
        self.cache_heightmaps = bool(cache_heightmaps)

        self._heightmap_arrays = []
        self._heightmap_paths = []
        labels = []
        sim_labels = []
        grasp_labels = []
        weights = []
        row_counts = []
        optional_inputs = {key: [] for key in _OPTIONAL_INPUT_KEYS}
        episode_groups = []
        has_episode_groups = True
        heightmap_shape = None
        for file_index, item in enumerate(self.paths):
            data = _load_array_file(item, load_heightmaps=False)
            n_rows, item_shape = _heightmap_info(item, data)
            if "actor_id" in data and "episode_id" in data:
                actor_ids = np.asarray(data["actor_id"], dtype=np.int64).reshape(-1)
                episode_ids = np.asarray(data["episode_id"], dtype=np.int64).reshape(-1)
                if len(actor_ids) != n_rows or len(episode_ids) != n_rows:
                    raise ValueError(
                        f"{item} episode metadata rows do not match heightmaps "
                        f"{n_rows}"
                    )
                episode_groups.append(
                    np.column_stack(
                        [
                            np.full(n_rows, file_index, dtype=np.int64),
                            actor_ids,
                            episode_ids,
                        ]
                    )
                )
            else:
                has_episode_groups = False
            if heightmap_shape is None:
                heightmap_shape = item_shape
            elif item_shape[1:] != heightmap_shape[1:]:
                raise ValueError(
                    f"{item} heightmaps shape tail {item_shape[1:]} does not "
                    f"match {heightmap_shape[1:]}"
                )

            label = self._load_label(data, item)
            sim_label = self._load_named_label(data, item, "label")
            if "grasp_label" in data:
                grasp_label = self._load_named_label(data, item, "grasp_label")
            else:
                grasp_label = np.ones_like(label, dtype=np.float32)
            if "weight" in data:
                weight = np.asarray(data["weight"], dtype=np.float32).reshape(-1)
            else:
                weight = np.ones_like(label, dtype=np.float32)

            for name, arr in (
                ("label", label),
                ("sim_label", sim_label),
                ("grasp_label", grasp_label),
                ("weight", weight),
            ):
                if len(arr) != n_rows:
                    raise ValueError(
                        f"{item} {name} rows {len(arr)} do not match heightmaps {n_rows}"
                    )

            labels.append(label)
            sim_labels.append(sim_label)
            grasp_labels.append(grasp_label)
            weights.append(weight)
            row_counts.append(n_rows)
            default_reason_ids = np.where(
                sim_label > 0.5, 0, FEASIBILITY_FAILURE_OTHER_ID
            ).astype(np.int16)
            default_reason_counts = np.zeros(
                (n_rows, len(FEASIBILITY_FAILURE_REASONS)), dtype=np.int16
            )
            default_reason_counts[np.arange(n_rows), default_reason_ids] = 1
            diagnostic_defaults = {
                "failure_reason_id": default_reason_ids,
                "failure_reason_counts": default_reason_counts,
                "nominal_label": sim_label.astype(np.float32),
                "stability_passes": (sim_label > 0.5).astype(np.int16),
                "stability_rollouts": np.ones(n_rows, dtype=np.int16),
            }
            for key in _OPTIONAL_INPUT_KEYS:
                optional_inputs[key].append(
                    np.asarray(data[key])
                    if key in data
                    else diagnostic_defaults.get(key)
                )
            if item.endswith((".h5", ".hdf5")):
                self._heightmap_paths.append(item)
                self._heightmap_arrays.append(None)
            else:
                self._heightmap_paths.append(None)
                self._heightmap_arrays.append(np.asarray(data["heightmaps"], dtype=np.float32))

        self.labels = np.concatenate(labels, axis=0)
        self.sim_labels = np.concatenate(sim_labels, axis=0)
        self.grasp_labels = np.concatenate(grasp_labels, axis=0)
        self.weights = np.concatenate(weights, axis=0)
        self.optional_inputs = {
            key: np.concatenate(parts, axis=0)
            for key, parts in optional_inputs.items()
            if all(part is not None for part in parts)
        }
        self.episode_groups = (
            np.concatenate(episode_groups, axis=0)
            if has_episode_groups and len(episode_groups) == len(self.paths)
            else None
        )

        self._row_offsets = np.cumsum([0] + row_counts)
        self._h5_files = [None for _ in self._heightmap_paths]
        self.heightmaps = _HeightmapShapeProxy((int(self._row_offsets[-1]),) + heightmap_shape[1:])
        if self.cache_heightmaps:
            self._cache_all_heightmaps()

        mask = (
            np.isfinite(self.labels)
            & np.isfinite(self.sim_labels)
            & np.isfinite(self.grasp_labels)
            & np.isfinite(self.weights)
        )
        if not np.all(mask):
            keep = np.flatnonzero(mask)
            self._index = keep.astype(np.int64)
            self.labels = self.labels[keep]
            self.sim_labels = self.sim_labels[keep]
            self.grasp_labels = self.grasp_labels[keep]
            self.weights = self.weights[keep]
            if self.episode_groups is not None:
                self.episode_groups = self.episode_groups[keep]
            self.optional_inputs = {
                key: value[keep] for key, value in self.optional_inputs.items()
            }
        else:
            self._index = None

    def _cache_all_heightmaps(self) -> None:
        for i, path in enumerate(self._heightmap_paths):
            if path is None or self._heightmap_arrays[i] is not None:
                continue
            import h5py

            with h5py.File(path, "r") as file:
                self._heightmap_arrays[i] = np.asarray(file["heightmaps"], dtype=np.float32)

    def _load_label(self, data, item: str) -> np.ndarray:
        if self.label_key == "grasp_count":
            return self._load_grasp_count_label(data, item)
        if self.label_key == "combined_count":
            label = self._load_named_label(data, item, "label")
            return label * self._load_grasp_count_label(data, item)

        keys = (
            ("label", "grasp_label")
            if self.label_key == "combined"
            else (self.label_key,)
        )
        out = None
        for key in keys:
            col = self._load_named_label(data, item, key)
            out = col if out is None else out * col
        return out

    def _load_named_label(self, data, item: str, key: str) -> np.ndarray:
        if key not in data:
            raise KeyError(
                f"{item} has no '{key}' (keys: {list(data.keys())}). "
                "Was the data generated with grasp_access enabled?"
            )
        return np.asarray(data[key], dtype=np.float32).reshape(-1)

    def _load_grasp_count_label(self, data, item: str) -> np.ndarray:
        if "grasp_n" not in data:
            raise KeyError(
                f"{item} has no 'grasp_n' (keys: {list(data.keys())}). "
                "Was the data generated with grasp_access enabled?"
            )
        grasp_n = np.asarray(data["grasp_n"], dtype=np.float32).reshape(-1)
        denom = np.log1p(self.grasp_count_normalizer)
        return np.clip(np.log1p(np.maximum(grasp_n, 0.0)) / denom, 0.0, 1.0)

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        source_idx = int(self._index[idx]) if self._index is not None else int(idx)
        return self._make_item(idx, self._load_heightmap(source_idx))

    def _make_item(self, idx: int, heightmap: np.ndarray) -> dict:
        item = {
            "heightmaps": torch.from_numpy(heightmap),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "sim_label": torch.tensor(self.sim_labels[idx], dtype=torch.float32),
            "grasp_label": torch.tensor(self.grasp_labels[idx], dtype=torch.float32),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
        }
        item.update(
            {
                key: torch.as_tensor(value[idx])
                for key, value in self.optional_inputs.items()
            }
        )
        return item

    def __getitems__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        indices = [int(i) for i in indices]
        source_indices = (
            self._index[indices].astype(np.int64)
            if self._index is not None
            else np.asarray(indices, dtype=np.int64)
        )
        heightmaps = self._load_heightmaps(source_indices)
        return [
            self._make_item(idx, heightmaps[i]) for i, idx in enumerate(indices)
        ]

    def _load_heightmap(self, idx: int) -> np.ndarray:
        return self._load_heightmaps(np.asarray([idx], dtype=np.int64))[0]

    def _load_heightmaps(self, indices: np.ndarray) -> np.ndarray:
        out = np.empty((len(indices),) + self.heightmaps.shape[1:], dtype=np.float32)
        if len(indices) == 0:
            return out

        file_indices = np.searchsorted(self._row_offsets, indices, side="right") - 1
        for file_idx in np.unique(file_indices):
            batch_pos = np.flatnonzero(file_indices == file_idx)
            local_indices = indices[batch_pos] - self._row_offsets[file_idx]
            array = self._heightmap_arrays[file_idx]
            if array is not None:
                out[batch_pos] = np.asarray(array[local_indices], dtype=np.float32)
                continue

            h5_file = self._h5_files[file_idx]
            if h5_file is None:
                import h5py

                h5_file = h5py.File(self._heightmap_paths[file_idx], "r")
                self._h5_files[file_idx] = h5_file

            order = np.argsort(local_indices, kind="stable")
            sorted_local = local_indices[order]
            sorted_batch_pos = batch_pos[order]
            unique_local, inverse = np.unique(sorted_local, return_inverse=True)
            start = int(unique_local[0])
            stop = int(unique_local[-1]) + 1
            span = stop - start
            if span <= max(len(unique_local) * 2, len(unique_local) + 512):
                values = np.asarray(h5_file["heightmaps"][start:stop], dtype=np.float32)
                out[sorted_batch_pos] = values[unique_local[inverse] - start]
            else:
                values = np.asarray(h5_file["heightmaps"][unique_local], dtype=np.float32)
                out[sorted_batch_pos] = values[inverse]
        return out


HeightmapFeasibilityNPZDataset = HeightmapFeasibilityDataset


class _HeightmapShapeProxy:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def _load_array_file(path: str, load_heightmaps: bool = True) -> dict[str, np.ndarray]:
    if path.endswith((".h5", ".hdf5")):
        import h5py

        with h5py.File(path, "r") as file:
            return {
                key: val[()]
                for key, val in file.items()
                if isinstance(val, h5py.Dataset)
                and (load_heightmaps or key != "heightmaps")
            }
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _heightmap_info(path: str, data: dict[str, np.ndarray]) -> tuple[int, tuple[int, ...]]:
    if path.endswith((".h5", ".hdf5")):
        import h5py

        with h5py.File(path, "r") as file:
            shape = tuple(int(v) for v in file["heightmaps"].shape)
        return shape[0], shape
    heightmaps = np.asarray(data["heightmaps"])
    return int(heightmaps.shape[0]), tuple(int(v) for v in heightmaps.shape)


def _resolve_feasibility_paths(path: str) -> List[str]:
    if path.endswith((".h5", ".hdf5", ".npz")):
        return [path] if os.path.isfile(path) else []
    if not os.path.isdir(path):
        return []

    integrated_candidates = [
        os.path.join(path, "feasibility.h5"),
        os.path.join(path, "feasibility", "feasibility.h5"),
    ]
    for candidate in integrated_candidates:
        if os.path.isfile(candidate):
            return [candidate]

    integrated_paths = sorted(
        os.path.join(root, fn)
        for root, _, files in os.walk(path)
        for fn in files
        if fn == "feasibility.h5"
    )
    if integrated_paths:
        return integrated_paths

    return sorted(
        os.path.join(root, fn)
        for root, _, files in os.walk(path)
        for fn in files
        if fn.endswith(".npz")
    )


def _resolve_npz_paths(path: str) -> List[str]:
    return [p for p in _resolve_feasibility_paths(path) if p.endswith(".npz")]
