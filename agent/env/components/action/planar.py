import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from agent.rl_models.heightmap_value import HeightmapValueModel

from ..inventory import InventoryManager, pick_pose_position_quat
from .floor_fill import sample_scored_xy, score_xy_debug_map
from .orientation import inward_xy, top_exposed_surface_normal
from ..state import State


class PlanarPoseSampler:
    """Picks promising (x, y) anchors from the height-map gap to the target wall."""

    _debug_dump_counts: Dict[int, int] = {}
    _feasible_region_printed: set = set()

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.score_model_kind = cfg.action.planar.score_model
        self.last_score_maps: dict[int, dict] = {}
        if self.score_model_kind == "score":
            pass
        elif self.score_model_kind == "heuristic":
            self._init_heuristic()
        elif self.score_model_kind == "cnn":
            self._init_cnn()
        else:
            raise ValueError(
                f"unknown action.planar.score_model: {self.score_model_kind!r} "
                f"(expected 'score', 'heuristic', or 'cnn')"
            )

    def _init_heuristic(self):
        self.sobel_x = torch.tensor(
            [
                [
                    [
                        [-5, -4, 0, 4, 5],
                        [-8, -10, 0, 10, 8],
                        [-10, -20, 0, 20, 10],
                        [-8, -10, 0, 10, 8],
                        [-5, -4, 0, 4, 5],
                    ]
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.sobel_y = torch.tensor(
            [
                [
                    [
                        [5, 8, 10, 8, 5],
                        [4, 10, 20, 10, 4],
                        [0, 0, 0, 0, 0],
                        [-4, -10, -20, -10, -4],
                        [-5, -8, -10, -8, -5],
                    ]
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.sobel_padding = self.sobel_x.shape[-1] // 2
        self.smooth_kernel = torch.ones(1, 1, 5, 5, device=self.device) / 25.0

    def _init_cnn(self):
        cnn_cfg = OmegaConf.load(self.cfg.action.planar.cnn.config)
        self.cnn = HeightmapValueModel(cnn_cfg)
        weights_path = self.cfg.action.planar.cnn.weights
        if weights_path:
            payload = torch.load(
                weights_path, map_location=self.device, weights_only=False
            )
            state_dict = (
                payload.get("model_state", payload)
                if isinstance(payload, dict)
                else payload
            )
            self.cnn.load_state_dict(state_dict)
        self.cnn.set_device(self.device)
        self.cnn.eval()

    def sample(
        self,
        inventory: InventoryManager,
        state: State,
        scene_height_map: Optional[np.ndarray] = None,
        min_count: int = 1,
        stone_idx: Optional[int] = None,
        random_start: bool = False,
    ) -> List[np.ndarray]:
        min_count = max(int(min_count), 1)
        if self.score_model_kind == "score":
            score_debug = score_xy_debug_map(
                inventory,
                state,
                stone_idx,
                scene_height_map=scene_height_map,
            )
            candidates = score_debug.get("candidates", [])
            xy_coords = (
                np.empty((0, 2), dtype=float)
                if not candidates
                else sample_scored_xy(
                    inventory,
                    candidates,
                    min_count,
                    random_start,
                )
            )
            score_debug["selected_xy"] = np.asarray(xy_coords, dtype=float).copy()
            self.last_score_maps[self._score_map_key(stone_idx)] = (
                self._compact_score_debug(score_debug, inventory, stone_idx)
            )
            if len(xy_coords) == 0:
                return []
            if self.cfg.action.pose_from_scan:
                return self._mixed_scan_poses(
                    xy_coords,
                    min_count,
                    inventory=inventory,
                    stone_idx=stone_idx,
                )
            return self._mixed_grid_poses(
                xy_coords,
                min_count,
                inventory=inventory,
                stone_idx=stone_idx,
            )

        height_map = (
            scene_height_map
            if scene_height_map is not None
            else inventory.get_height_map(state)
        )
        score_map = self._score_map(inventory, state, height_map)
        score_map = self._mask_score_map_outside_target(inventory, score_map)
        self._maybe_print_feasible_region(inventory, score_map)
        self._maybe_dump_debug_maps(inventory, state, height_map, score_map)
        xy_coords = self._boltzmann_xy(
            inventory,
            score_map,
            k=min_count,
        )

        if self.cfg.action.pose_from_scan:
            return self._mixed_scan_poses(xy_coords, min_count)
        return self._mixed_grid_poses(xy_coords, min_count)

    @staticmethod
    def _score_map_key(stone_idx: Optional[int]) -> int:
        return -1 if stone_idx is None else int(stone_idx)

    def score_map_for_stone(self, stone_idx: Optional[int]) -> dict | None:
        return self.last_score_maps.get(self._score_map_key(stone_idx))

    def score_at_xy(
        self,
        stone_idx: Optional[int],
        xy: np.ndarray,
    ) -> tuple[float, float] | None:
        """Return the nearest score-grid value and its per-stone normalization."""
        score_map = self.score_map_for_stone(stone_idx)
        if score_map is None:
            return None

        x_coords = np.asarray(score_map.get("x_coords", []), dtype=float)
        y_coords = np.asarray(score_map.get("y_coords", []), dtype=float)
        scores = np.asarray(score_map.get("scores", []), dtype=float)
        candidate_mask = np.asarray(
            score_map.get("candidate_mask", np.zeros_like(scores, dtype=bool)),
            dtype=bool,
        )
        if (
            x_coords.size == 0
            or y_coords.size == 0
            or scores.shape != candidate_mask.shape
            or scores.shape != (y_coords.size, x_coords.size)
        ):
            return None

        xy = np.asarray(xy, dtype=float).reshape(-1)
        if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
            return None
        x_idx = int(np.argmin(np.abs(x_coords - xy[0])))
        y_idx = int(np.argmin(np.abs(y_coords - xy[1])))
        if not candidate_mask[y_idx, x_idx] or not np.isfinite(
            scores[y_idx, x_idx]
        ):
            return None

        valid_scores = scores[candidate_mask & np.isfinite(scores)]
        if valid_scores.size == 0:
            return None
        raw_score = float(scores[y_idx, x_idx])
        score_min = float(np.min(valid_scores))
        score_span = float(np.max(valid_scores) - score_min)
        normalized = (
            0.0 if score_span <= 1e-12 else (raw_score - score_min) / score_span
        )
        return raw_score, float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def _compact_score_debug(
        score_debug: dict,
        inventory: InventoryManager,
        stone_idx: Optional[int],
    ) -> dict:
        compact = {
            key: np.asarray(score_debug[key]).copy()
            for key in (
                "x_coords",
                "y_coords",
                "scores",
                "valid",
                "candidate_mask",
                "height",
                "height_term",
                "connectedness",
                "open_area",
                "fill_area",
                "frontier",
                "excavator_distance",
                "selected_xy",
            )
            if key in score_debug
        }
        compact["weights"] = dict(score_debug.get("weights", {}) or {})
        compact["h_min"] = float(score_debug.get("h_min", 0.0))
        compact["h_span"] = float(score_debug.get("h_span", 0.0))
        compact["stone_idx"] = None if stone_idx is None else int(stone_idx)
        compact["stone_id"] = (
            None if stone_idx is None else int(inventory.stone_set[int(stone_idx)])
        )
        compact["n_candidates"] = int(len(score_debug.get("candidates", []) or []))
        return compact

    def _score_map(
        self,
        inventory: InventoryManager,
        state: State,
        scene_height_map: Optional[np.ndarray] = None,
        scale: int = 4,
    ) -> np.ndarray:
        height_map = (
            scene_height_map
            if scene_height_map is not None
            else inventory.get_height_map(state)
        )
        if self.score_model_kind == "cnn":
            score_map = self.cnn.score_map(height_map, inventory.ensure_target_height_map())
            return self._resize_score_map(score_map)
        return self._heuristic_score_map(
            height_map, inventory.ensure_target_height_map(), scale
        )

    def _maybe_dump_debug_maps(
        self,
        inventory: InventoryManager,
        state: State,
        height_map: np.ndarray,
        score_map: np.ndarray,
    ) -> None:
        debug_cfg = self.cfg.action.planar.get("debug_maps", {})
        if not bool(debug_cfg.get("enabled", False)):
            return

        current_step = len(state.stone_seq) + 1
        target_steps = {int(step) for step in debug_cfg.get("steps", [4, 5])}
        if current_step not in target_steps:
            return

        max_dumps = int(debug_cfg.get("max_dumps_per_step", 1))
        dump_count = self._debug_dump_counts.get(current_step, 0)
        if max_dumps >= 0 and dump_count >= max_dumps:
            return

        output_dir = Path(debug_cfg.get("output_dir", ".debug/height_score_maps"))
        output_dir.mkdir(parents=True, exist_ok=True)

        dump_index = dump_count + 1
        self._debug_dump_counts[current_step] = dump_index
        prefix = output_dir / f"step_{current_step:02d}_dump_{dump_index:03d}"

        np.save(f"{prefix}_height_map.npy", np.asarray(height_map))
        np.save(
            f"{prefix}_target_height_map.npy",
            np.asarray(inventory.target_height_map),
        )
        np.save(f"{prefix}_score_map.npy", np.asarray(score_map))
        self._save_debug_map_png(
            prefix,
            height_map,
            inventory.target_height_map,
            score_map,
        )
        print(f"Saved planar debug maps for step {current_step}: {prefix}_*.png/.npy")

    @staticmethod
    def _save_debug_map_png(
        prefix: Path,
        height_map: np.ndarray,
        target_height_map: np.ndarray,
        score_map: np.ndarray,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        items = (
            ("Scene height map", height_map, "viridis"),
            ("Target height map", target_height_map, "viridis"),
            ("Planar score map", score_map, "magma"),
        )
        for ax, (title, data, cmap) in zip(axes, items):
            image = ax.imshow(np.asarray(data), origin="lower", cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(f"{prefix}_maps.png", dpi=160)
        plt.close(fig)

    def _heuristic_score_map(
        self,
        height_map: np.ndarray,
        target_height_map: np.ndarray,
        scale: int,
    ) -> np.ndarray:
        height = self._to_tensor(height_map)
        target = self._to_tensor(target_height_map)

        score_shape = self._configured_score_map_shape()
        if score_shape is None:
            height = F.avg_pool2d(height, kernel_size=scale, stride=scale)
            target = F.avg_pool2d(target, kernel_size=scale, stride=scale)
        else:
            height = F.adaptive_avg_pool2d(height, output_size=score_shape)
            target = F.adaptive_avg_pool2d(target, output_size=score_shape)

        gx = F.conv2d(height, self.sobel_x, padding=self.sobel_padding)
        gy = F.conv2d(height, self.sobel_y, padding=self.sobel_padding)
        contrast_score = -torch.sqrt(gx**2 + gy**2)

        local_mean = F.conv2d(height, self.smooth_kernel, padding=2)
        support = local_mean - height

        heuristic_terms = {
            "low": 1.5,
            "contrast": 1.0,
            "support": 1.0,
            "target_height_map": 3.0,
        }
        score = (
            heuristic_terms["low"] * (-height)
            + heuristic_terms["contrast"] * contrast_score
            + heuristic_terms["support"] * support
            + heuristic_terms["target_height_map"] * target
        )
        return score.squeeze().cpu().numpy()

    def _resize_score_map(self, score_map: np.ndarray) -> np.ndarray:
        score_shape = self._configured_score_map_shape()
        if score_shape is None or tuple(score_map.shape) == score_shape:
            return score_map

        score = self._to_tensor(score_map)
        score = F.interpolate(
            score,
            size=score_shape,
            mode="bilinear",
            align_corners=False,
        )
        return score.squeeze().cpu().numpy()

    def _configured_score_map_shape(self) -> Optional[Tuple[int, int]]:
        size = self.cfg.action.planar.get("score_map_size", None)
        if size is None:
            return None
        if isinstance(size, int):
            width = height = int(size)
        else:
            if len(size) != 2:
                raise ValueError("action.planar.score_map_size must be [width, height]")
            width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError("action.planar.score_map_size values must be positive")
        return (height, width)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return (
            torch.tensor(arr.copy(), dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def _boltzmann_xy(
        self,
        inventory: InventoryManager,
        score_map: np.ndarray,
        k: int,
    ) -> np.ndarray:
        xy_grid = self._xy_grid(inventory, score_map.shape)
        flat = score_map.reshape(-1)
        valid = np.isfinite(flat)
        if not np.any(valid):
            raise ValueError("no finite planar scores remain after target masking")

        temperature = float(self.cfg.action.planar.get("boltzmann_temperature", 1.0))
        if temperature <= 0.0:
            raise ValueError("action.planar.boltzmann_temperature must be positive")

        valid_logits = flat[valid] / temperature
        valid_logits = valid_logits - np.max(valid_logits)
        probs = np.exp(valid_logits)
        probs = probs / np.sum(probs)
        min_probability = float(
            self.cfg.action.planar.get("boltzmann_min_probability", 0.0)
        )
        if min_probability < 0.0:
            raise ValueError("action.planar.boltzmann_min_probability must be >= 0")
        if min_probability > 0.0:
            probs = np.maximum(probs, min_probability)
            probs = probs / np.sum(probs)

        valid_idx = np.flatnonzero(valid)
        k = max(int(k), 1)
        replace = k > len(valid_idx)
        idx = np.random.choice(valid_idx, size=k, replace=replace, p=probs)

        _, W = score_map.shape
        return xy_grid[idx // W, idx % W]

    @staticmethod
    def _mean_stone_aabb_extent(inventory: InventoryManager) -> np.ndarray:
        extents = [
            np.asarray(stone.local_aabb_extent(), dtype=float)
            for stone in inventory.stones
        ]
        if not extents:
            return np.ones(3, dtype=float)
        return np.mean(extents, axis=0)

    def _mask_score_map_outside_target(
        self,
        inventory: InventoryManager,
        score_map: np.ndarray,
    ) -> np.ndarray:
        mask = self._target_xy_mask(inventory, score_map.shape)
        masked = np.asarray(score_map, dtype=float).copy()
        masked[~mask] = -np.inf
        return masked

    def _maybe_print_feasible_region(
        self,
        inventory: InventoryManager,
        score_map: np.ndarray,
    ) -> None:
        if not bool(self.cfg.action.planar.get("print_feasible_region", False)):
            return

        feasible = np.isfinite(score_map)
        n_cells = int(np.count_nonzero(feasible))
        if n_cells == 0:
            print(f"Planar feasible xy region: 0 cells / {score_map.size}")
            return

        xlim, ylim = inventory.xlim, inventory.ylim
        height, width = score_map.shape
        dx = (xlim[1] - xlim[0]) / max(width - 1, 1)
        dy = (ylim[1] - ylim[0]) / max(height - 1, 1)
        area = n_cells * dx * dy

        xy = self._xy_grid(inventory, score_map.shape)[feasible]
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)

        print_once = bool(
            self.cfg.action.planar.get("print_feasible_region_once", True)
        )
        key = (
            score_map.shape,
            round(float(xlim[0]), 6),
            round(float(xlim[1]), 6),
            round(float(ylim[0]), 6),
            round(float(ylim[1]), 6),
            round(float(xy_min[0]), 6),
            round(float(xy_min[1]), 6),
            round(float(xy_max[0]), 6),
            round(float(xy_max[1]), 6),
            n_cells,
        )
        if print_once and key in self._feasible_region_printed:
            return
        self._feasible_region_printed.add(key)

        print(
            "Planar feasible xy region: "
            f"{n_cells}/{score_map.size} cells, "
            f"approx area={area:.3f} m^2, "
            f"cell=({dx:.3f}, {dy:.3f}) m, "
            f"x=[{xy_min[0]:.3f}, {xy_max[0]:.3f}], "
            f"y=[{xy_min[1]:.3f}, {xy_max[1]:.3f}]"
        )

    def _target_xy_mask(
        self,
        inventory: InventoryManager,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        xy_grid = self._xy_grid(inventory, shape)
        wall = inventory.target_wall
        origin = np.asarray(wall.origin[:2], dtype=float)
        half_extent = np.array([wall.width, wall.length], dtype=float) / 2.0
        margin = float(self.cfg.action.planar.get("target_mask_margin", 0.0))
        lower = origin - half_extent - margin
        upper = origin + half_extent + margin
        return np.all((xy_grid >= lower) & (xy_grid <= upper), axis=-1)

    @staticmethod
    def _xy_grid(
        inventory: InventoryManager,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        xlim, ylim = inventory.xlim, inventory.ylim
        height, width = shape
        x = np.linspace(xlim[0], xlim[1], width)
        y = np.linspace(ylim[0], ylim[1], height)
        X, Y = np.meshgrid(x, y, indexing="xy")
        return np.stack([X, Y], axis=-1)

    def _mixed_scan_poses(
        self,
        xy_coords: np.ndarray,
        n: int,
        inventory: Optional[InventoryManager] = None,
        stone_idx: Optional[int] = None,
    ) -> List[np.ndarray]:
        poses: List[np.ndarray] = []
        variance = self.cfg.action.planar.variance
        for i in range(n):
            xy = xy_coords[i % len(xy_coords)]
            angle_z = self._yaw_angle_for_xy(inventory, stone_idx, xy, i)
            pose = np.zeros(6)
            pose[:2] = xy + np.random.randn(2) * variance
            pose[2:] = Rotation.from_euler(
                "xyz", [0, 0, angle_z], degrees=True
            ).as_quat()
            poses.append(pose)
        return poses

    def _mixed_grid_poses(
        self,
        xy_coords: np.ndarray,
        n: int,
        inventory: Optional[InventoryManager] = None,
        stone_idx: Optional[int] = None,
    ) -> List[np.ndarray]:
        poses: List[np.ndarray] = []
        angles = [
            (angle_x, angle_z)
            for angle_x in self.cfg.action.rotation.angles_x
            for angle_z in self.cfg.action.rotation.angles_z
        ]
        face_orient_cache: Dict[Tuple[float, float], List[np.ndarray]] = {}
        for i in range(n):
            xy = xy_coords[i % len(xy_coords)]
            pose = np.zeros(6)
            pose[:2] = xy
            if stone_idx is not None and self._uses_floor_fill_orientation(inventory, xy):
                # Face-normal → wall-normal orientation (paper §3.3 fit estimation)
                xy_key = (round(float(xy[0]), 6), round(float(xy[1]), 6))
                if xy_key not in face_orient_cache:
                    face_orient_cache[xy_key] = self._face_normal_orientations(
                        inventory, stone_idx, xy
                    )
                face_quats = face_orient_cache[xy_key]
                pose[2:] = face_quats[i % len(face_quats)]
            elif self._uses_floor_fill_orientation(inventory, xy):
                angle_x = list(self.cfg.action.rotation.angles_x)[
                    i % len(self.cfg.action.rotation.angles_x)
                ]
                angle_z = self._floor_fill_yaw_angle(inventory, None, xy, i)
                pose[2:] = Rotation.from_euler(
                    "xyz", [angle_x, 0, angle_z], degrees=True
                ).as_quat()
            else:
                angle_x, angle_z = angles[i % len(angles)]
                r_angle = Rotation.from_euler(
                    "xyz", [angle_x, 0, angle_z], degrees=True
                )
                r_align = self._principal_axis_rotation(inventory, stone_idx)
                # Re-align stone by principal axes first, then apply the angles.
                pose[2:] = (r_angle * r_align).as_quat()
            poses.append(pose)
        return poses

    def _principal_axis_rotation(
        self,
        inventory: Optional[InventoryManager],
        stone_idx: Optional[int],
    ) -> Rotation:
        """Rotation aligning the stone's principal axes (long/mid/short) to X/Y/Z.

        Falls back to identity when no specific stone is available (e.g. the
        non-score path that samples without a stone index).
        """
        if inventory is None or stone_idx is None:
            return Rotation.identity()
        stone = inventory.stones[stone_idx]
        return Rotation.from_quat(stone.principal_axis_alignment())

    def _yaw_angle_for_xy(
        self,
        inventory: Optional[InventoryManager],
        stone_idx: Optional[int],
        xy: np.ndarray,
        sample_idx: int,
    ) -> float:
        angles_z = list(self.cfg.action.rotation.angles_z)
        if not self._uses_floor_fill_orientation(inventory, xy):
            return float(angles_z[sample_idx % len(angles_z)])
        return self._floor_fill_yaw_angle(inventory, stone_idx, xy, sample_idx)

    def _floor_fill_yaw_angle(
        self,
        inventory: InventoryManager,
        stone_idx: Optional[int],
        xy: np.ndarray,
        sample_idx: int,
    ) -> float:
        inward = inward_xy(inventory, xy)
        if np.linalg.norm(inward) < 1e-9:
            angles_z = list(self.cfg.action.rotation.angles_z)
            return float(angles_z[sample_idx % len(angles_z)])

        base_yaw = np.rad2deg(np.arctan2(inward[1], inward[0]))
        if self.cfg.action.pose_from_scan and stone_idx is not None:
            base_yaw -= self._scan_upper_face_heading(inventory, stone_idx)

        offsets = self._floor_fill_orientation_offsets()
        return float(base_yaw + offsets[sample_idx % len(offsets)])

    def _uses_floor_fill_orientation(
        self,
        inventory: Optional[InventoryManager],
        xy: np.ndarray,
    ) -> bool:
        return (
            inventory is not None
            and self.score_model_kind == "score"
            and self._floor_fill_orientation_enabled()
            and self._is_boundary_xy(inventory, xy)
        )

    def _floor_fill_orientation_enabled(self) -> bool:
        floor_fill_cfg = self.cfg.action.planar.get("floor_fill", {})
        orient_cfg = floor_fill_cfg.get("orientation", {})
        return bool(orient_cfg.get("enabled", True))

    def _floor_fill_orientation_offsets(self) -> List[float]:
        floor_fill_cfg = self.cfg.action.planar.get("floor_fill", {})
        orient_cfg = floor_fill_cfg.get("orientation", {})
        offsets = orient_cfg.get("offsets_deg", [0.0, -45.0, 45.0, 180.0])
        return [float(offset) for offset in offsets] or [0.0]

    def _face_normal_orientations(
        self,
        inventory: InventoryManager,
        stone_idx: int,
        xy: np.ndarray,
    ) -> List[np.ndarray]:
        """Quaternions that align each stone face to the wall inward direction at xy.

        For each representative face normal n_face, computes R such that
        R.apply(n_face) = ê_y (wall inward), then rotates about ê_y by each
        configured offset to vary the edge direction along the rim path.
        Falls back to the discrete angle grid when face data is unavailable.
        """
        fallback = [
            Rotation.from_euler("xyz", [ax, 0, az], degrees=True).as_quat()
            for ax in self.cfg.action.rotation.angles_x
            for az in self.cfg.action.rotation.angles_z
        ]

        stone = inventory.stones[stone_idx]
        normals, _ = stone.representative_face_normals()
        if len(normals) == 0:
            return fallback

        inward_2d = inward_xy(inventory, xy)
        if np.linalg.norm(inward_2d) < 1e-9:
            return fallback
        e_y = np.array([inward_2d[0], inward_2d[1], 0.0], dtype=float)

        offsets = self._floor_fill_orientation_offsets()
        quats: List[np.ndarray] = []
        for n_face in normals:
            n_face = np.asarray(n_face, dtype=float)
            norm = np.linalg.norm(n_face)
            if norm < 1e-9:
                continue
            n_face = n_face / norm
            # Skip strongly downward-facing normals: aligning them to the wall
            # inward direction would require flipping the stone upside-down.
            if n_face[2] < -0.5:
                continue
            try:
                R_base, _ = Rotation.align_vectors([e_y], [n_face])
            except Exception:
                continue
            for offset_deg in offsets:
                # Rotate about ê_y to vary the stone edge direction along the rim
                # path (0° and 180° give the paper's two canonical orientations).
                R_offset = Rotation.from_rotvec(np.deg2rad(offset_deg) * e_y)
                quats.append((R_offset * R_base).as_quat())

        return quats if quats else fallback

    def _is_boundary_xy(self, inventory: InventoryManager, xy: np.ndarray) -> bool:
        floor_fill_cfg = self.cfg.action.planar.get("floor_fill", {})
        orient_cfg = floor_fill_cfg.get("orientation", {})
        margin_scale = float(orient_cfg.get("boundary_margin_radius_scale", 1.5))
        mean_extent = self._mean_stone_aabb_extent(inventory)
        margin = margin_scale * 0.5 * float(np.min(mean_extent[:2]))

        wall = inventory.target_wall
        origin = np.asarray(wall.origin[:2], dtype=float)
        half = np.array([wall.width, wall.length], dtype=float) / 2.0
        distance_to_edge = half - np.abs(np.asarray(xy[:2], dtype=float) - origin)
        return bool(np.min(distance_to_edge) <= margin)

    def _scan_upper_face_heading(
        self,
        inventory: InventoryManager,
        stone_idx: int,
    ) -> float:
        _, scan_quat = pick_pose_position_quat(
            inventory.pick_poses[inventory.stone_set[stone_idx]]
        )
        stone = inventory.stones[stone_idx]
        normals, areas = stone.representative_face_normals()
        if len(normals) == 0:
            scan_up = Rotation.from_quat(scan_quat).as_matrix()[:, 2]
            return self._heading_from_vector(scan_up)

        scan_rot = Rotation.from_quat(scan_quat).as_matrix()
        world_normals = normals @ scan_rot.T
        upper_normal = self._top_exposed_surface_normal(world_normals, areas)
        if upper_normal is None:
            upper_normal = scan_rot[:, 2]
        return self._heading_from_vector(upper_normal)

    def _top_exposed_surface_normal(
        self,
        world_normals: np.ndarray,
        areas: np.ndarray,
    ) -> Optional[np.ndarray]:
        floor_fill_cfg = self.cfg.action.planar.get("floor_fill", {})
        orient_cfg = floor_fill_cfg.get("orientation", {})
        min_z = float(orient_cfg.get("upper_face_min_z", 0.0))
        return top_exposed_surface_normal(world_normals, areas, min_z=min_z)

    def floor_fill_upper_face_inward_ok(
        self,
        inventory: InventoryManager,
        stone_idx: int,
        pose: np.ndarray,
    ) -> bool:
        if not self._uses_floor_fill_orientation(inventory, pose[:2]):
            return True

        normal = self._upper_face_normal_at_pose(inventory, stone_idx, pose)
        if normal is None:
            return True

        heading = np.asarray(normal[:2], dtype=float)
        heading_norm = np.linalg.norm(heading)
        floor_fill_cfg = self.cfg.action.planar.get("floor_fill", {})
        orient_cfg = floor_fill_cfg.get("orientation", {})
        min_upward = float(orient_cfg.get("min_upward_dot", 0.7))
        if float(normal[2]) <= min_upward:
            return False

        min_horizontal = float(orient_cfg.get("upper_face_min_horizontal", 0.05))
        if heading_norm < min_horizontal:
            return True

        inward = inward_xy(inventory, pose[:2])
        if np.linalg.norm(inward) < 1e-9:
            return True

        max_angle = float(orient_cfg.get("max_inward_angle_deg", 90.0))
        min_dot = float(np.cos(np.deg2rad(max_angle)))
        if np.isclose(max_angle, 90.0):
            min_dot = 0.0
        heading = heading / heading_norm
        return float(np.dot(heading, inward)) >= min_dot

    def _upper_face_normal_at_pose(
        self,
        inventory: InventoryManager,
        stone_idx: int,
        pose: np.ndarray,
    ) -> Optional[np.ndarray]:
        stone = inventory.stones[stone_idx]
        normals, areas = stone.representative_face_normals()
        if len(normals) == 0:
            return None
        world_normals = normals @ Rotation.from_quat(pose[3:]).as_matrix().T
        return self._top_exposed_surface_normal(world_normals, areas)

    @staticmethod
    def _heading_from_vector(vector: np.ndarray) -> float:
        heading = np.asarray(vector[:2], dtype=float)
        if np.linalg.norm(heading) < 1e-6:
            return 0.0
        return float(np.rad2deg(np.arctan2(heading[1], heading[0])))
