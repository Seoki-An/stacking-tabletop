from typing import List, Tuple
import numpy as np
from omegaconf import OmegaConf
import open3d as o3d
import pyvista as pv

from utils.geometry import (
    half_to_vertices,
    rigidTrans_half,
    compute_intersection,
    vertices_to_hull,
    fix_normals_outward,
)
from .stone import StoneObject


class TargetWall:

    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg

        w, l, h, t = cfg.width, cfg.length, cfg.height, cfg.taper
        if self.cfg.randomize:
            a = np.sqrt(w * l)
            w = a + a * 0.5 * (2 * np.random.rand(1).item() - 1)
            l = a * a / w
            t += 2 * np.random.rand(1).item() - 1.0
        t *= np.pi / 180
        self.origin = np.array([cfg.origin[0], cfg.origin[1], 0.0])
        self.geometries: List[TargetWallGeometry] = [
            TargetWallGeometry(width=w, length=l, height=h, taper=t, origin=cfg.origin)
        ]
        self.volume: float = 0.0
        for geometry in self.geometries:
            self.volume += geometry.volume
        self.width: float = w
        self.length: float = l
        self.height: float = h
        self.taper: float = t

        self._iou_grid_res: float = 0.04
        self._iou_grid_wall_pts: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self._iou_voxel_volume: float = 0.0
        self._precompute_iou_grid()

    def _precompute_iou_grid(self, resolution: float = 0.04) -> None:
        ox, oy = float(self.cfg.origin[0]), float(self.cfg.origin[1])
        margin = resolution
        xs = np.arange(ox - self.width / 2 - margin, ox + self.width / 2 + margin + 1e-9, resolution)
        ys = np.arange(oy - self.length / 2 - margin, oy + self.length / 2 + margin + 1e-9, resolution)
        zs = np.arange(-margin, self.height + margin + 1e-9, resolution)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)
        inside = np.zeros(len(pts), dtype=bool)
        for geom in self.geometries:
            A, b = geom.support_planes  # A: (F, 3), b: (F,) — polytope is {x: Ax <= b}
            inside |= np.all(pts @ A.T <= b, axis=1)
        self._iou_grid_res = resolution
        self._iou_grid_wall_pts = pts[inside]
        self._iou_voxel_volume = float(resolution ** 3)

    def copy(self):
        wall = TargetWall(self.cfg)
        wall.geometries = [geometry.copy() for geometry in self.geometries]
        wall.volume = self.volume
        wall.width = self.width
        wall.length = self.length
        wall.height = self.height
        wall.taper = self.taper
        wall._iou_grid_res = self._iou_grid_res
        wall._iou_grid_wall_pts = self._iou_grid_wall_pts
        wall._iou_voxel_volume = self._iou_voxel_volume
        return wall

    def _compute_iou_fast(
        self,
        stone: StoneObject,
        pose: np.ndarray,
        divide_by_stone_volume: bool,
    ) -> float:
        wall_pts = self._iou_grid_wall_pts
        if len(wall_pts) == 0:
            return 0.0
        # AABB pre-filter: worst-case rotated extent is the half-diagonal
        half_extent = float(np.linalg.norm(stone.bound_extent)) / 2 + self._iou_grid_res
        aabb_mask = np.all(np.abs(wall_pts - pose[:3]) <= half_extent, axis=1)
        candidates = wall_pts[aabb_mask]
        if len(candidates) == 0:
            return 0.0
        inside = np.zeros(len(candidates), dtype=bool)
        for geom in stone.geometries:
            A, b = geom.support_planes
            Aw, bw = rigidTrans_half(pose, A, b)
            inside |= np.all(candidates @ Aw.T <= bw.ravel(), axis=1)
        intersection_volume = float(inside.sum()) * self._iou_voxel_volume
        if divide_by_stone_volume:
            return intersection_volume / stone.volume
        return intersection_volume / self.volume

    def compute_IoU(
        self,
        stone: StoneObject,
        pose: np.ndarray,
        divide_by_stone_volume: bool = True,
        save_models: bool = False,
    ) -> float:
        if not save_models:
            return self._compute_iou_fast(stone, pose, divide_by_stone_volume)

        stone_support_planes = []
        for geometry in stone.geometries:
            A, b = geometry.support_planes
            stone_support_planes.append(rigidTrans_half(pose, A, b))

        volumes = []
        wall_support_planes = [geometry.support_planes for geometry in self.geometries]
        v_list = compute_intersection(wall_support_planes, stone_support_planes)
        for v in v_list:
            if v.size == 0:
                continue
            vtv = v.transpose() @ v
            eigvals, _ = np.linalg.eig(vtv)
            eigvals_spread = np.sqrt(eigvals.max() / eigvals.min())
            if np.linalg.det(vtv) < 1e-2 or eigvals_spread > 100.0:
                continue
            model, vol = vertices_to_hull(v, calc_vol=True)
            center = model.points.mean(0)
            model.points = 1.10 * (model.points - center[None, :]) + center[None, :]
            stone.add_IoU_model(
                model.points.copy(), model.faces.copy().reshape(-1, 4)[:, 1:]
            )
            volumes.append(vol)

        if len(volumes) == 0:
            return 0.0
        else:
            if divide_by_stone_volume:
                return np.array(volumes).sum() / stone.volume
            else:
                return np.array(volumes).sum() / self.volume

    def sample_internal_points(self, n_points: int) -> np.ndarray:
        sampled_points = []
        for model, half_space in zip(self.models, self.support_planes):
            bounds = model.bounds
            random_points = np.random.uniform(
                low=[bounds[0], bounds[2], bounds[4]],
                high=[bounds[1], bounds[3], bounds[5]],
                size=(10 * n_points, 3),
            ).transpose()
            A, b = half_space
            sampled_points.append(
                random_points[:, np.all((A @ random_points - b[:, None]) <= 0, axis=0)]
            )

        sampled_points = np.concatenate(sampled_points, axis=-1)

        return sampled_points[:, :n_points]


class TargetWallGeometry:
    def __init__(
        self,
        width: float,
        length: float,
        height: float,
        taper: float,
        origin: Tuple[float, float] = (0.0, 0.0),
    ):
        w, l, h, t = width, length, height, taper
        x0, y0 = origin

        plane_w = np.array([x0 - w / 2, y0, 0, -np.cos(t), 0, np.sin(t)])
        plane_s = np.array([x0, y0 - l / 2, 0, 0, -np.cos(t), np.sin(t)])
        plane_e = np.array([x0 + w / 2, y0, 0, np.cos(t), 0, np.sin(t)])
        plane_n = np.array([x0, y0 + l / 2, 0, 0, np.cos(t), np.sin(t)])
        plane_b = np.array([x0, y0, -1e-1, 0, 0, -1])
        plane_t = np.array([x0, y0, h, 0, 0, 1])
        planes = np.array([plane_w, plane_s, plane_e, plane_n, plane_b, plane_t])
        A = planes[:, 3:]
        b = (planes[:, :3] * A).sum(-1)
        self.support_planes = (A, b)

        model: pv.PolyData
        self.volume: float
        model, self.volume = vertices_to_hull(
            half_to_vertices([self.support_planes])[0], calc_vol=True
        )
        model = model.subdivide(3)
        model.compute_normals(
            point_normals=True,
            cell_normals=False,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=True,
            inplace=True,
        )
        self.points: np.ndarray = np.array(model.points.copy())
        self.normals: np.ndarray = np.array(model.point_normals.copy())
        self.faces: np.ndarray = np.array(model.faces.copy()).reshape(-1, 4)[:, 1:]

        self.width: float = w
        self.length: float = l
        self.height: float = h
        self.taper: float = t
        self.origin: Tuple[float, float] = origin

    def get_mesh(self):

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.points),
            o3d.utility.Vector3iVector(self.faces),
        )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return fix_normals_outward(mesh)

    def get_mesh_array(self):
        return self.points.copy(), self.faces.copy()

    def copy(self):
        wall = TargetWallGeometry(
            width=self.width,
            length=self.length,
            height=self.height,
            taper=self.taper,
            origin=(self.origin[0], self.origin[1]),
        )
        return wall
