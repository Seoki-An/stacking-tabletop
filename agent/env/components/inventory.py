from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING, Tuple

import os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import re
import pickle
import open3d as o3d

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from matplotlib import pyplot as plt

from .wall import TargetWall
from .stone import StoneObject

if TYPE_CHECKING:
    from .state import State

from utils._phase_timer import timed


def pick_pose_position_quat(pose) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(position, quaternion)` from supported field-pose formats."""
    if isinstance(pose, dict):
        if "pose" in pose:
            return pick_pose_position_quat(pose["pose"])
        pos = pose.get("pos", pose.get("position"))
        quat = pose.get("quat", pose.get("quaternion"))
        if pos is not None and quat is not None:
            return (
                np.asarray(pos, dtype=float).copy(),
                np.asarray(quat, dtype=float).copy(),
            )

    if isinstance(pose, (tuple, list)) and len(pose) == 2:
        pos = np.asarray(pose[0], dtype=float).reshape(-1)
        quat = np.asarray(pose[1], dtype=float).reshape(-1)
        if pos.size == 3 and quat.size == 4:
            return pos.copy(), quat.copy()

    arr = np.asarray(pose, dtype=float).reshape(-1)
    if arr.size >= 7:
        return arr[:3].copy(), arr[3:7].copy()
    if arr.size == 4:
        return np.zeros(3, dtype=float), arr.copy()
    raise ValueError(f"unsupported pick pose format with shape {arr.shape}")


class InventoryManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.target_wall: TargetWall = TargetWall(self.cfg.target)
        self.model_dir = cfg.data.load_dir
        self.stones: List[StoneObject] = []
        self.points: np.ndarray = np.zeros((0, 0, 3))
        self.faces: np.ndarray = np.zeros((0, 0, 3), dtype=int)
        self.sharpness: np.ndarray = np.zeros(0)
        self.dsf_points: np.ndarray = np.zeros((0, 0, 3), dtype=np.float32)
        self.dsf_normals: np.ndarray = np.zeros((0, 0, 3), dtype=np.float32)
        self.dsf_point_mask: np.ndarray = np.zeros((0, 0), dtype=bool)
        self.stone_physical_features: np.ndarray = np.zeros(
            (0, 14), dtype=np.float32
        )
        self.mean_radius: float = 0.0
        self.mean_extent: np.ndarray = np.zeros((3, 2))
        self._set_height_map_bounds()

        self.pick_poses: Dict[int, np.ndarray] = {}
        if self.cfg.action.pose_from_scan:
            with open(self.cfg.action.pose_data_path, "rb") as f:
                raw_pick_poses: Dict[int, np.ndarray] = pickle.load(f)
            self.pick_poses = {
                int(stone_id): pose for stone_id, pose in raw_pick_poses.items()
            }

        self.target_height_map: np.ndarray | None = None
        if self._uses_height_map_for_xy():
            self.ensure_target_height_map()

    def reset(self):
        self.stones: List[StoneObject] = []

        available_stone_ids: List[int] = []
        for file in Path(self.model_dir).glob("model_*.obj"):
            match = re.search(r"\d+", file.stem)
            if match:
                available_stone_ids.append(int(match.group()))

        fixed_stone_set = self.cfg.data.get("fixed_stone_set", None)
        if fixed_stone_set is not None:
            # Keep downstream sampling on the same RNG stream as ordinary reset,
            # where the initial stone order is produced by this permutation.
            np.random.permutation(available_stone_ids)
            available = set(available_stone_ids)
            self.stone_set = [int(stone_id) for stone_id in fixed_stone_set]
            missing = [
                stone_id for stone_id in self.stone_set if stone_id not in available
            ]
            if missing:
                raise ValueError(f"fixed_stone_set contains missing stone ids: {missing}")
        else:
            self.stone_set = available_stone_ids

        if self.cfg.action.pose_from_scan:
            pose_ids = set(self.pick_poses)
            self.stone_set = [
                stone_id for stone_id in self.stone_set if stone_id in pose_ids
            ]

        if fixed_stone_set is not None:
            self.stone_set = np.asarray(self.stone_set, dtype=int)
        else:
            self.stone_set = np.random.permutation(self.stone_set)
        self.stone_set = self.stone_set[: self.cfg.data.limit_candidates]
        if len(self.stone_set) == 0:
            self.points = np.zeros((0, 0, 3))
            self.faces = np.zeros((0, 0, 3), dtype=int)
            self.sharpness = np.zeros(0)
            self._refresh_dsf_observation_arrays()
            self.mean_radius = 0.0
            self.mean_extent = np.zeros((3, 2))
            self._set_height_map_bounds()
            self.target_height_map = None
            if self._uses_height_map_for_xy():
                self.ensure_target_height_map()
            return

        stone_bound_radii = []
        stone_bound_extents = []

        max_n_points = 0
        max_n_faces = 0
        for id in self.stone_set:
            path = os.path.join(self.model_dir, f"model_{id}.obj")
            stone = StoneObject(path, id=id)
            self.stones.append(stone)
            stone_bound_radii.append(stone.bound_radius)
            stone_bound_extents.append(stone.bound_extent)
            points, faces = stone.get_lowpoly_mesh_array()
            max_n_points = max(max_n_points, points.shape[0])
            max_n_faces = max(max_n_faces, faces.shape[0])

        self.points = np.inf * np.ones((len(self.stones), max_n_points, 3))
        self.sharpness = np.zeros(len(self.stones))
        # Pad faces with -1 sentinel so consumers can drop padded rows without
        # confusing them with the legitimate vertex index 0.
        self.faces = -np.ones((len(self.stones), max_n_faces, 3), dtype=int)

        for i, stone in enumerate(self.stones):
            points, faces = stone.get_lowpoly_mesh_array()
            n_points = points.shape[0]
            n_faces = faces.shape[0]
            self.points[i, :n_points] = points
            self.sharpness[i] = stone.sharpness
            self.faces[i, :n_faces] = faces

        self.mean_radius = np.mean(stone_bound_radii)
        self.mean_extent = np.stack(stone_bound_extents).mean(axis=0)
        self._refresh_dsf_observation_arrays()
        self._set_height_map_bounds()
        self.target_height_map = None
        if self._uses_height_map_for_xy():
            self.ensure_target_height_map()

    def copy(self):
        inventory = InventoryManager(self.cfg)
        inventory.target_wall = self.target_wall.copy()
        inventory.stones = [stone.copy() for stone in self.stones]
        inventory.stone_set = self.stone_set.copy()
        inventory.points = self.points.copy()
        inventory.faces = self.faces.copy()
        inventory.sharpness = self.sharpness.copy()
        inventory.dsf_points = self.dsf_points.copy()
        inventory.dsf_normals = self.dsf_normals.copy()
        inventory.dsf_point_mask = self.dsf_point_mask.copy()
        inventory.stone_physical_features = self.stone_physical_features.copy()
        inventory.mean_radius = self.mean_radius
        inventory.mean_extent = self.mean_extent.copy()
        inventory.pick_poses = self.pick_poses.copy()
        inventory.target_height_map = (
            None if self.target_height_map is None else self.target_height_map.copy()
        )

        return inventory

    def update_from_state(self, state: State):
        # Each Simulator picks its own random stone subset on reset, so a state
        # built by one env may reference ids absent from another env's
        # inventory. Cache what we have, lazy-load the rest, and reorder the
        # geometry arrays so they line up with the new stone order.
        stone_by_id = {stone.id: stone for stone in self.stones}
        stones: List[StoneObject] = []
        for id in state.stone_set:
            sid = int(id)
            stone = stone_by_id.get(sid)
            if stone is None:
                path = os.path.join(self.model_dir, f"model_{sid}.obj")
                stone = StoneObject(path, id=sid)
                stone_by_id[sid] = stone
            stones.append(stone)

        self.stones = stones
        self.stone_set = np.asarray(state.stone_set)
        self._refresh_geometry_arrays()

        for idx in state.stone_seq:
            st = self.stones[idx]
            try:
                st.pose = state.stone_poses[st.id]
            except KeyError:
                print(
                    f"Warning: Stone pose for stone id {st.id} not found in state. Using default pose."
                )
                print(f"state.stone_poses keys: {list(state.stone_poses.keys())}")

    def _refresh_geometry_arrays(self):
        if not self.stones:
            self.points = np.zeros((0, 0, 3))
            self.faces = np.zeros((0, 0, 3), dtype=int)
            self.sharpness = np.zeros(0)
            self._refresh_dsf_observation_arrays()
            self.mean_radius = 0.0
            self.mean_extent = np.zeros((3, 2))
            return

        mesh_arrays = [stone.get_lowpoly_mesh_array() for stone in self.stones]
        max_n_points = max(points.shape[0] for points, _ in mesh_arrays)
        max_n_faces = max(faces.shape[0] for _, faces in mesh_arrays)

        n = len(self.stones)
        points = np.inf * np.ones((n, max_n_points, 3))
        faces = -np.ones((n, max_n_faces, 3), dtype=int)
        sharpness = np.zeros(n)
        bound_radii = []
        bound_extents = []
        for i, (stone, (stone_points, stone_faces)) in enumerate(
            zip(self.stones, mesh_arrays)
        ):
            points[i, : stone_points.shape[0]] = stone_points
            faces[i, : stone_faces.shape[0]] = stone_faces
            sharpness[i] = stone.sharpness
            bound_radii.append(stone.bound_radius)
            bound_extents.append(stone.bound_extent)

        self.points = points
        self.faces = faces
        self.sharpness = sharpness
        self.mean_radius = float(np.mean(bound_radii))
        self.mean_extent = np.stack(bound_extents).mean(axis=0)
        self._refresh_dsf_observation_arrays()

    def _refresh_dsf_observation_arrays(self) -> None:
        point_count = max(
            int(self.cfg.get("observation", {}).get("dsf_surface_points", 128)),
            1,
        )
        if not self.stones:
            self.dsf_points = np.zeros((0, point_count, 3), dtype=np.float32)
            self.dsf_normals = np.zeros((0, point_count, 3), dtype=np.float32)
            self.dsf_point_mask = np.zeros((0, point_count), dtype=bool)
            self.stone_physical_features = np.zeros((0, 14), dtype=np.float32)
            return

        samples = [
            stone.get_dsf_surface_samples(point_count) for stone in self.stones
        ]
        self.dsf_points = np.stack([sample[0] for sample in samples])
        self.dsf_normals = np.stack([sample[1] for sample in samples])
        self.dsf_point_mask = np.stack([sample[2] for sample in samples])
        self.stone_physical_features = np.stack(
            [stone.physical_features() for stone in self.stones]
        ).astype(np.float32)

    def get_local_scene_dsf_samples(
        self,
        state: "State",
        candidate_pose: np.ndarray,
        n_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return nearby placed-stone DSF samples in the candidate frame."""
        n_points = max(int(n_points), 1)
        out_points = np.zeros((n_points, 3), dtype=np.float32)
        out_normals = np.zeros((n_points, 3), dtype=np.float32)
        out_mask = np.zeros(n_points, dtype=bool)
        pose = np.asarray(candidate_pose, dtype=float)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            return out_points, out_normals, out_mask

        candidate_rotation = Rotation.from_quat(pose[3:])
        candidate_rotation_inv = candidate_rotation.inv()
        points = []
        normals = []
        for stone_idx in state.stone_seq:
            stone_idx = int(stone_idx)
            stone = self.stones[stone_idx]
            stone_pose = state.stone_poses.get(stone.id)
            if stone_pose is None:
                continue
            stone_pose = np.asarray(stone_pose, dtype=float)
            stone_rotation = Rotation.from_quat(stone_pose[3:])
            valid = self.dsf_point_mask[stone_idx]
            world_points = (
                stone_rotation.apply(self.dsf_points[stone_idx, valid])
                + stone_pose[:3]
            )
            world_normals = stone_rotation.apply(self.dsf_normals[stone_idx, valid])
            points.append(candidate_rotation_inv.apply(world_points - pose[:3]))
            normals.append(candidate_rotation_inv.apply(world_normals))

        if not points:
            return out_points, out_normals, out_mask
        points = np.concatenate(points, axis=0)
        normals = np.concatenate(normals, axis=0)
        order = np.argsort(np.sum(points**2, axis=1), kind="stable")[:n_points]
        out_points[: len(order)] = points[order]
        out_normals[: len(order)] = normals[order]
        out_mask[: len(order)] = True
        return out_points, out_normals, out_mask

    def _uses_height_map_for_xy(self) -> bool:
        if bool(self.cfg.get("observation", {}).get("render_height_map", False)):
            return True
        planar_cfg = self.cfg.action.get("planar", {})
        score_model = str(planar_cfg.get("score_model", "heuristic"))
        return score_model in {"heuristic", "cnn"}

    def _set_height_map_bounds(self) -> None:
        center = np.asarray(self.target_wall.origin, dtype=float).copy()
        w, l = self.target_wall.width, self.target_wall.length
        max_length = max(w, l) + self.cfg.height_map.margin
        self.xlim = [-max_length / 2 + center[0], max_length / 2 + center[0]]
        self.ylim = [-max_length / 2 + center[1], max_length / 2 + center[1]]
        self.z_top = self.target_wall.height + 1.0

    def ensure_target_height_map(self) -> np.ndarray:
        if self.target_height_map is None:
            self.target_height_map = self.get_target_height_map()
        return self.target_height_map

    def _raycast_height_map(
        self,
        meshes: List[o3d.geometry.TriangleMesh],
    ) -> np.ndarray:
        """Return metric top-surface heights without creating an EGL renderer."""
        height, width = (int(value) for value in self.cfg.height_map.resolution)
        out = np.zeros((height, width), dtype=np.float32)
        meshes = [mesh for mesh in meshes if len(mesh.triangles) > 0]
        if not meshes:
            return out

        max_z = max(
            float(np.asarray(mesh.vertices)[:, 2].max()) for mesh in meshes
        )
        ray_origin_z = max(float(self.z_top), max_z + 1.0)
        x = np.linspace(self.xlim[0], self.xlim[1], width, dtype=np.float32)
        y = np.linspace(self.ylim[0], self.ylim[1], height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        rays = np.zeros((height * width, 6), dtype=np.float32)
        rays[:, 0] = xx.reshape(-1)
        rays[:, 1] = yy.reshape(-1)
        rays[:, 2] = ray_origin_z
        rays[:, 5] = -1.0

        scene = o3d.t.geometry.RaycastingScene()
        for mesh in meshes:
            scene.add_triangles(
                o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            )
        distance = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit = np.isfinite(distance)
        out.reshape(-1)[hit] = np.maximum(
            ray_origin_z - distance[hit],
            0.0,
        )
        return out

    @staticmethod
    def _show_height_map(height: np.ndarray) -> None:
        plt.imshow(height, origin="lower")
        plt.colorbar()
        plt.title("Height Map")
        plt.show()

    @timed("render")
    def get_target_height_map(self, visualize: bool = False) -> np.ndarray:
        meshes = [
            geometry.get_mesh() for geometry in self.target_wall.geometries
        ]
        height = self._raycast_height_map(meshes)
        if visualize:
            self._show_height_map(height)
        return height.astype(np.float16)

    @timed("render")
    def get_height_map(self, state: State, visualize: bool = False) -> np.ndarray:
        meshes = []
        for idx in state.stone_seq:
            stone = self.stones[idx]
            for geometry in stone.geometries:
                mesh = geometry.get_lowpoly_mesh()
                mesh.transform(stone.get_pose_matrix())
                meshes.append(mesh)

        height = self._raycast_height_map(meshes)
        if visualize:
            self._show_height_map(height)
        return height.astype(np.float16)
