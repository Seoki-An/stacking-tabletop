from typing import List, Tuple

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull

from diffsimpy import diffsim
from utils.geometry import (
    vertices_to_hull,
    compute_polytope_halfspaces,
    fix_normals_outward,
)
from utils.dsf import DiffSupportSimple

DENSITY = 1000.0
REPRESENTATIVE_NORMAL_MERGE_ANGLE_DEG = 50.0
REPRESENTATIVE_NORMAL_MERGE_DISTANCE_SCALE = 0.5


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministically select a spatially distributed subset."""
    points = np.asarray(points, dtype=float)
    count = min(max(int(count), 0), len(points))
    if count == 0:
        return np.empty(0, dtype=int)

    center = np.mean(points, axis=0)
    first = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    selected = np.empty(count, dtype=int)
    selected[0] = first
    min_distance = np.sum((points - points[first]) ** 2, axis=1)
    for i in range(1, count):
        idx = int(np.argmax(min_distance))
        selected[i] = idx
        distance = np.sum((points - points[idx]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return selected


class StoneObject:
    def __init__(self, model_path: str, id: int = None):
        self.model_path = model_path
        self.id = id
        self.object_config = diffsim.BodyConfig(model_path)

        self.geometries: List[StoneGeometry] = [
            StoneGeometry(geometry) for geometry in self.object_config.geometry
        ]
        if not self.geometries:
            raise ValueError(f"Stone model has no geometry: {model_path}")

        all_vertices = np.concatenate(
            [geometry.vertices_dsf for geometry in self.geometries], axis=0
        )
        self.bound_extent = all_vertices.max(axis=0) - all_vertices.min(axis=0)
        self.bound_radius = np.mean(self.bound_extent) / 2
        self.volume = np.sum([geometry.volume for geometry in self.geometries])
        if self.volume > 0.0:
            self.sharpness = np.average(
                [geometry.sharpness for geometry in self.geometries],
                weights=[geometry.volume for geometry in self.geometries],
            )
        else:
            self.sharpness = np.mean(
                [geometry.sharpness for geometry in self.geometries]
            )

        self.mass = DENSITY * self.object_config.inertial.mass
        self.inertia = DENSITY * np.array(self.object_config.inertial.inertia)
        self.object_config.inertial.mass = self.mass
        self.object_config.inertial.set_inertia(self.inertia)

        self.IoU_models: List[Tuple[np.ndarray, np.ndarray]] = []
        self._representative_face_normals = None
        self._representative_face_centers_normals = None
        self._global_hull_mesh_array = None
        self._representative_hull_mesh_array = None
        self._local_aabb = None
        self._principal_axis_alignment = None
        self._dsf_surface_sample_cache = {}

    def copy(self):
        stone = StoneObject(self.model_path, id=self.id)

        stone.pose = self.pose.copy()
        stone.motion = self.motion.copy()
        stone.object_config.error_reduction_ratio = (
            self.object_config.error_reduction_ratio
        )
        stone.IoU_models = self.IoU_models.copy()
        stone._representative_face_normals = self._representative_face_normals
        stone._representative_face_centers_normals = (
            self._representative_face_centers_normals
        )
        stone._global_hull_mesh_array = self._global_hull_mesh_array
        stone._representative_hull_mesh_array = self._representative_hull_mesh_array
        stone._local_aabb = self._local_aabb
        stone._principal_axis_alignment = self._principal_axis_alignment
        stone._dsf_surface_sample_cache = {
            key: tuple(value.copy() for value in sample)
            for key, sample in self._dsf_surface_sample_cache.items()
        }

        return stone

    def add_IoU_model(self, points: np.ndarray, faces: np.ndarray):
        self.IoU_models.append((points, faces))

    def clear_IoU_models(self):
        self.IoU_models = []

    def get_IoU_models(self):
        meshes = []
        for points, faces in self.IoU_models:
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(points),
                o3d.utility.Vector3iVector(faces),
            )
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            meshes.append(fix_normals_outward(mesh))
        return meshes

    @property
    def pose(self) -> np.ndarray:
        return self.object_config.pose.vectorized().copy()

    @pose.setter
    def pose(self, pose: np.ndarray):
        if isinstance(pose, np.ndarray):
            mat = np.eye(4)
            mat[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
            mat[:3, -1] = pose[:3]
            pose = diffsim.Pose().from_matrix(mat)

        self.object_config.pose = pose

    @property
    def motion(self) -> np.ndarray:
        return self.object_config.motion.vectorized().copy()

    @motion.setter
    def motion(self, motion: np.ndarray):
        if isinstance(motion, np.ndarray):
            motion = diffsim.Motion(motion.tolist())

        self.object_config.motion = motion

    @property
    def config(self):
        return self.object_config

    def get_pose_matrix(self):
        return self.object_config.pose.as_matrix()

    def get_lowpoly_mesh(self):
        points, faces = self.get_lowpoly_mesh_array()
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return fix_normals_outward(mesh)

    def get_lowpoly_mesh_array(self):
        points = []
        faces = []
        offset = 0
        for geometry in self.geometries:
            g_points, g_faces = geometry.get_lowpoly_mesh_array()
            points.append(g_points)
            faces.append(g_faces + offset)
            offset += g_points.shape[0]
        return np.concatenate(points, axis=0), np.concatenate(faces, axis=0)

    def get_dsf_mesh_array(self):
        points = []
        faces = []
        offset = 0
        for geometry in self.geometries:
            g_points, g_faces = geometry.get_dsf_mesh_array()
            points.append(g_points)
            faces.append(g_faces + offset)
            offset += g_points.shape[0]
        return np.concatenate(points, axis=0), np.concatenate(faces, axis=0)

    def get_dsf_surface_samples(
        self,
        n_points: int = 128,
        resolution: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the exterior of the union of this stone's DSF components."""
        n_points = max(int(n_points), 1)
        key = (n_points, int(resolution))
        cached = self._dsf_surface_sample_cache.get(key)
        if cached is not None:
            return tuple(value.copy() for value in cached)

        component_samples = [
            geometry.get_dsf_surface_samples(resolution)
            for geometry in self.geometries
        ]
        hulls = [ConvexHull(points) for points, _ in component_samples]
        tolerance = max(float(np.linalg.norm(self.bound_extent)) * 1e-4, 1e-7)
        exterior_points = []
        exterior_normals = []
        for i, (points, normals) in enumerate(component_samples):
            keep = np.ones(len(points), dtype=bool)
            for j, hull in enumerate(hulls):
                if i == j:
                    continue
                equations = hull.equations
                signed = points @ equations[:, :3].T + equations[:, 3]
                probes = points + 10.0 * tolerance * normals
                probe_signed = (
                    probes @ equations[:, :3].T + equations[:, 3]
                )
                covered = (
                    (np.max(signed, axis=1) < -tolerance)
                    | (np.max(probe_signed, axis=1) < -tolerance)
                )
                keep &= ~covered
            exterior_points.append(points[keep])
            exterior_normals.append(normals[keep])

        points = np.concatenate(exterior_points, axis=0)
        normals = np.concatenate(exterior_normals, axis=0)
        if len(points) == 0:
            points = np.concatenate(
                [sample[0] for sample in component_samples], axis=0
            )
            normals = np.concatenate(
                [sample[1] for sample in component_samples], axis=0
            )

        selected = _farthest_point_indices(points, n_points)
        out_points = np.zeros((n_points, 3), dtype=np.float32)
        out_normals = np.zeros((n_points, 3), dtype=np.float32)
        out_mask = np.zeros(n_points, dtype=bool)
        out_points[: len(selected)] = points[selected]
        out_normals[: len(selected)] = normals[selected]
        out_mask[: len(selected)] = True
        sample = (out_points, out_normals, out_mask)
        self._dsf_surface_sample_cache[key] = sample
        return tuple(value.copy() for value in sample)

    def physical_features(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray([self.mass, self.volume], dtype=np.float32),
                np.asarray(self.inertia, dtype=np.float32).reshape(-1),
                np.asarray(self.bound_extent, dtype=np.float32),
            ]
        )

    def local_aabb(self) -> np.ndarray:
        """Return [xmin, xmax, ymin, ymax, zmin, zmax] over all geometry DSFs."""
        if self._local_aabb is not None:
            return self._local_aabb.copy()

        axes = np.eye(3)
        aabb = np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf])
        for geometry in self.geometries:
            for axis_idx, direction in enumerate(axes):
                try:
                    _, p_min = geometry.dsf.support(-direction)
                    _, p_max = geometry.dsf.support(direction)
                    min_value = float(p_min[axis_idx, 0])
                    max_value = float(p_max[axis_idx, 0])
                except Exception:
                    min_value = float(np.min(geometry.vertices_dsf[:, axis_idx]))
                    max_value = float(np.max(geometry.vertices_dsf[:, axis_idx]))
                aabb[2 * axis_idx] = min(aabb[2 * axis_idx], min_value)
                aabb[2 * axis_idx + 1] = max(aabb[2 * axis_idx + 1], max_value)

        self._local_aabb = aabb.copy()
        return aabb

    def local_aabb_extent(self) -> np.ndarray:
        aabb = self.local_aabb()
        return np.array(
            [aabb[1] - aabb[0], aabb[3] - aabb[2], aabb[5] - aabb[4]],
            dtype=float,
        )

    def xy_aabb_radius(self) -> float:
        extent = self.local_aabb_extent()
        return float(0.5 * np.linalg.norm(extent[:2]))

    def mean_xy_aabb_radius(self) -> float:
        extent = self.local_aabb_extent()
        return float(0.25 * (extent[0] + extent[1]))

    def get_global_hull_mesh(self):
        points, faces = self.get_global_hull_mesh_array()
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return fix_normals_outward(mesh)

    def get_global_hull_mesh_array(self):
        """Return one convex hull over all decomposed stone geometry vertices."""
        if self._global_hull_mesh_array is not None:
            points, faces = self._global_hull_mesh_array
            return points.copy(), faces.copy()

        vertices = np.concatenate(
            [geometry.vertices_dsf for geometry in self.geometries], axis=0
        )
        hull = ConvexHull(vertices)
        hull_vertices = vertices[hull.vertices]

        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
        faces = np.asarray(
            [[remap[int(idx)] for idx in simplex] for simplex in hull.simplices],
            dtype=int,
        )

        self._global_hull_mesh_array = (hull_vertices.copy(), faces.copy())
        return hull_vertices, faces

    def principal_axis_alignment(self) -> np.ndarray:
        """Quaternion (x, y, z, w) re-aligning the stone's principal axes to world.

        The principal axes come from PCA of the convex-hull vertices. The returned
        rotation maps the longest principal axis to +X, the middle to +Y, and the
        shortest to +Z. Cached because the geometry is fixed.
        """
        if self._principal_axis_alignment is not None:
            return self._principal_axis_alignment.copy()

        points, _ = self.get_global_hull_mesh_array()
        centered = points - points.mean(axis=0)
        _, eigvecs = np.linalg.eigh(centered.T @ centered)
        # eigh yields ascending eigenvalues (spread); reverse to longest-first.
        axes = eigvecs[:, ::-1]  # columns: [longest, middle, shortest] in local frame
        # World-from-local rotation has the principal axes as its rows.
        matrix = axes.T
        if np.linalg.det(matrix) < 0.0:
            matrix[2] *= -1.0  # keep a proper right-handed rotation
        self._principal_axis_alignment = Rotation.from_matrix(matrix).as_quat()
        return self._principal_axis_alignment.copy()

    def get_representative_hull_mesh_array(self):
        """Return the triangulated global hull used to build representative faces."""
        if self._representative_hull_mesh_array is not None:
            points, faces = self._representative_hull_mesh_array
            return points.copy(), faces.copy()

        points, faces = self.get_global_hull_mesh_array()
        self._representative_hull_mesh_array = (points.copy(), faces.copy())
        return points, faces

    def get_representative_hull_mesh(self):
        points, faces = self.get_representative_hull_mesh_array()
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return fix_normals_outward(mesh)

    def representative_face_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return outward normals from merged convex-hull face patches."""
        if self._representative_face_normals is not None:
            normals, areas = self._representative_face_normals
            return normals.copy(), areas.copy()

        _, normals, areas = self.representative_face_centers_normals()
        self._representative_face_normals = (normals.copy(), areas.copy())
        return normals, areas

    def representative_face_centers_normals(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return centers, normals, and areas for merged convex-hull face patches."""
        if self._representative_face_centers_normals is not None:
            centers, normals, areas = self._representative_face_centers_normals
            return centers.copy(), normals.copy(), areas.copy()

        points, faces = self.get_representative_hull_mesh_array()
        if len(points) < 4 or len(faces) == 0:
            centers = np.empty((0, 3), dtype=float)
            normals = np.empty((0, 3), dtype=float)
            areas = np.empty(0, dtype=float)
            self._representative_face_centers_normals = (centers, normals, areas)
            self._representative_face_normals = (normals, areas)
            return centers.copy(), normals.copy(), areas.copy()

        centers, normals, areas = self._merged_face_centers_normals_areas(
            points,
            faces,
            REPRESENTATIVE_NORMAL_MERGE_ANGLE_DEG,
            REPRESENTATIVE_NORMAL_MERGE_DISTANCE_SCALE,
        )
        self._representative_face_centers_normals = (
            centers.copy(),
            normals.copy(),
            areas.copy(),
        )
        self._representative_face_normals = (normals.copy(), areas.copy())
        return centers, normals, areas

    @classmethod
    def _merged_face_centers_normals_areas(
        cls,
        points: np.ndarray,
        faces: np.ndarray,
        merge_angle_deg: float,
        merge_distance_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers, normals, areas = cls._face_centers_normals_and_areas(points, faces)
        if len(normals) == 0:
            return centers, normals, areas

        cos_thresh = float(np.cos(np.deg2rad(merge_angle_deg)))
        hull_extent = np.ptp(points, axis=0)
        distance_thresh = float(merge_distance_scale) * float(np.mean(hull_extent))
        assigned = np.zeros(len(normals), dtype=bool)
        merged_centers = []
        merged_normals = []
        merged_areas = []
        order = np.argsort(areas)[::-1]

        for seed_idx in order:
            if assigned[seed_idx]:
                continue
            group = cls._nearby_normal_group(
                centers,
                normals,
                assigned,
                seed_idx,
                cos_thresh,
                distance_thresh,
            )
            assigned[group] = True

            weights = areas[group]
            area = float(np.sum(weights))
            if area <= 1e-12:
                weights = np.ones(len(group), dtype=float)
                area = float(len(group))
            center = np.average(centers[group], axis=0, weights=weights)
            normal = np.average(normals[group], axis=0, weights=weights)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            merged_centers.append(center)
            merged_normals.append(normal / norm)
            merged_areas.append(area)

        return (
            np.asarray(merged_centers, dtype=float),
            np.asarray(merged_normals, dtype=float),
            np.asarray(merged_areas, dtype=float),
        )

    @staticmethod
    def _nearby_normal_group(
        centers: np.ndarray,
        normals: np.ndarray,
        assigned: np.ndarray,
        seed_idx: int,
        cos_thresh: float,
        distance_thresh: float,
    ) -> np.ndarray:
        unassigned = ~assigned
        aligned = np.einsum("ij,j->i", normals, normals[seed_idx]) >= cos_thresh
        distances = np.linalg.norm(centers - centers[seed_idx], axis=1)
        nearby = distances <= distance_thresh
        return np.flatnonzero(unassigned & aligned & nearby)

    @staticmethod
    def _face_centers_normals_and_areas(
        points: np.ndarray,
        faces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tri = points[faces]
        raw_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(raw_normals, axis=1)
        valid = norm > 1e-12
        if not np.any(valid):
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=float),
                np.empty(0, dtype=float),
            )

        tri = tri[valid]
        raw_normals = raw_normals[valid]
        norm = norm[valid]
        normals = raw_normals / norm[:, None]
        areas = 0.5 * norm

        mesh_center = np.mean(points, axis=0)
        face_centers = np.mean(tri, axis=1)
        outward = np.einsum("ij,ij->i", normals, face_centers - mesh_center) >= 0.0
        normals[~outward] *= -1.0
        return face_centers, normals, areas


class StoneGeometry:
    def __init__(self, geometry: diffsim.BodyConfig.geometry):
        model, self.volume = vertices_to_hull(
            np.transpose(geometry.nodes), calc_vol=True
        )

        self.vertices_dsf: np.ndarray = np.array(model.points.copy())
        self.faces_dsf: np.ndarray = np.array(model.faces.copy()).reshape(-1, 4)[:, 1:]
        self.sharpness: float = geometry.sharpness

        self.support_planes, _ = compute_polytope_halfspaces(self.vertices_dsf)
        self.bound_extent: np.ndarray = self.vertices_dsf.max(
            axis=0
        ) - self.vertices_dsf.min(axis=0)
        self.bound_radius: float = np.mean(self.bound_extent) / 2

        self.dsf: DiffSupportSimple = DiffSupportSimple(
            self.vertices_dsf.T, self.sharpness
        )

    def copy(self):
        geometry = StoneGeometry.__new__(StoneGeometry)
        geometry.vertices_dsf = self.vertices_dsf.copy()
        geometry.faces_dsf = self.faces_dsf.copy()
        geometry.sharpness = self.sharpness
        geometry.support_planes = (
            self.support_planes[0].copy(),
            self.support_planes[1].copy(),
        )
        geometry.bound_extent = self.bound_extent.copy()
        geometry.bound_radius = self.bound_radius
        geometry.dsf = DiffSupportSimple(
            geometry.vertices_dsf.T, geometry.sharpness
        )  # reinitialize dsf
        return geometry

    def get_mesh(self):
        points, faces = self.get_dsf_mesh_array()

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return fix_normals_outward(mesh)

    def get_dsf_mesh_array(self):
        points, faces = self.dsf.get_mesh(resolution=4)
        return (
            np.asarray(points, dtype=float).copy(),
            np.asarray(faces, dtype=int).copy(),
        )

    def get_dsf_surface_samples(
        self, resolution: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        points, normals = self.dsf.sample_surface(resolution)
        return points.copy(), normals.copy()

    def get_lowpoly_mesh(self):
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices_dsf.copy()),
            o3d.utility.Vector3iVector(self.faces_dsf.copy()),
        )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return fix_normals_outward(mesh)

    def get_lowpoly_mesh_array(self):
        return self.vertices_dsf.copy(), self.faces_dsf.copy()
