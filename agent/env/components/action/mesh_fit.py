from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from agent.config_views import mesh_fit_config
from utils._phase_timer import timed

if TYPE_CHECKING:
    from ..inventory import InventoryManager
    from ..state import State
    from ..stone import StoneObject


@dataclass
class MeshFaceData:
    vertices: np.ndarray
    triangles: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray


_PLY_FACE_CACHE: dict[str, Optional[MeshFaceData]] = {}


def ply_vertices(stone: "StoneObject") -> Optional[np.ndarray]:
    data = _ply_faces(stone)
    if data is None:
        return None
    return data.vertices.copy()


@timed("ply_face_fit")
def refine_pose_by_ply_faces(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Refine an initial pose by aligning real-mesh support faces before posegen."""
    debug = {
        "enabled": False,
        "changed": False,
        "reason": "",
        "support_faces": 0,
        "support_bodies": 0,
        "icp_fitness": 0.0,
        "icp_rmse": np.inf,
        "icp_pre_min_distance": np.inf,
        "icp_post_min_distance": np.inf,
    }

    def finish(out_pose: np.ndarray, reason: str = ""):
        debug["reason"] = reason
        debug["changed"] = bool(np.linalg.norm(out_pose[:7] - pose[:7]) > 1e-6)
        if return_debug:
            return out_pose, debug
        return out_pose

    cfg = mesh_fit_config(inventory)
    if not cfg.enabled:
        return finish(pose, "disabled")
    debug["enabled"] = True
    min_supports = cfg.min_support_bodies
    activate_after = cfg.activate_after
    if len(state.stone_seq) < max(min_supports, activate_after, 1):
        return finish(pose, "inactive_step")

    target = inventory.stones[stone_idx]
    target_faces = _ply_faces(target)
    if target_faces is None:
        return finish(pose, "missing_target_ply")

    support_faces = _support_face_candidates(inventory, state, pose, cfg)
    support_body_count = len({face["stone_idx"] for face in support_faces})
    debug["support_faces"] = len(support_faces)
    debug["support_bodies"] = support_body_count
    if support_body_count < max(min_supports, 1):
        return finish(pose, "not_enough_support_bodies")

    support_faces = support_faces[: cfg.max_support_faces]
    support_patch = _aggregate_support_patch(support_faces, cfg)
    if support_patch is None:
        return finish(pose, "empty_support_patch")

    refined, fit_debug = _best_face_fit_pose(target_faces, support_patch, pose, cfg)
    debug.update(fit_debug)
    return finish(refined, debug.get("reason", ""))


def _ply_faces(stone: "StoneObject") -> Optional[MeshFaceData]:
    path = _ply_path(stone)
    key = str(path)
    if key in _PLY_FACE_CACHE:
        return _PLY_FACE_CACHE[key]
    if not path.exists():
        _PLY_FACE_CACHE[key] = None
        return None

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        _PLY_FACE_CACHE[key] = None
        return None

    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=int)
    tri_vertices = vertices[triangles]
    centers = np.mean(tri_vertices, axis=1)
    cross = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    lengths = np.linalg.norm(cross, axis=1)
    valid = lengths > 1e-12
    if not np.any(valid):
        _PLY_FACE_CACHE[key] = None
        return None

    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / lengths[valid, None]
    normals = _orient_face_normals_outward(vertices, centers, normals, valid)
    areas = 0.5 * lengths
    data = MeshFaceData(
        vertices=vertices,
        triangles=triangles,
        centers=centers,
        normals=normals,
        areas=areas,
    )
    _PLY_FACE_CACHE[key] = data
    return data


def _orient_face_normals_outward(
    vertices: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Make PLY face normals usable even when triangle winding is inconsistent."""
    oriented = np.asarray(normals, dtype=float).copy()
    if len(vertices) == 0 or len(centers) == 0:
        return oriented

    centroid = np.mean(np.asarray(vertices, dtype=float), axis=0)
    radial = np.asarray(centers, dtype=float) - centroid[None, :]
    dots = np.sum(oriented * radial, axis=1)
    flip = valid & np.isfinite(dots) & (dots < 0.0)
    oriented[flip] *= -1.0
    return oriented


def _ply_path(stone: "StoneObject") -> Path:
    path = Path(stone.model_path)
    return path.with_name(f"{path.stem}_mesh.ply")


def _support_face_candidates(
    inventory: "InventoryManager",
    state: "State",
    pose: np.ndarray,
    cfg,
) -> list[dict]:
    min_up = cfg.support_min_normal_z
    search_radius = cfg.support_search_radius
    max_z_gap = cfg.support_max_z_gap
    max_per_stone = cfg.max_faces_per_support
    preselect_per_stone = cfg.preselect_faces_per_support

    candidates: list[dict] = []
    pose_xy = np.asarray(pose[:2], dtype=float)
    for placed_idx in state.stone_seq:
        placed = inventory.stones[placed_idx]
        data = _ply_faces(placed)
        if data is None:
            continue
        placed_pose = np.asarray(
            state.stone_poses.get(placed.id, placed.pose), dtype=float
        )
        placed_rot = Rotation.from_quat(placed_pose[3:])
        centers = placed_rot.apply(data.centers) + placed_pose[:3]
        normals = placed_rot.apply(data.normals)
        upward = normals[:, 2] >= min_up
        if not np.any(upward):
            continue

        upward_indices = np.nonzero(upward)[0]
        top_order = upward_indices[
            np.argsort(data.areas[upward_indices] + 0.25 * centers[upward_indices, 2])[
                ::-1
            ]
        ][:preselect_per_stone]
        if len(top_order) == 0:
            continue

        xy_dist = np.linalg.norm(centers[top_order, :2] - pose_xy, axis=1)
        close_local = xy_dist <= search_radius
        if not np.any(close_local):
            continue

        close_indices = top_order[close_local]
        close_dist = xy_dist[close_local]
        top_z = float(np.max(centers[close_indices, 2]))
        z_close = centers[close_indices, 2] >= top_z - max_z_gap
        indices = close_indices[z_close]
        index_dist = close_dist[z_close]
        if len(indices) == 0:
            continue

        scores = data.areas[indices] + 0.25 * centers[indices, 2] - 0.1 * index_dist
        order = indices[np.argsort(scores)[::-1]][:max_per_stone]
        for idx in order:
            dist = float(np.linalg.norm(centers[idx, :2] - pose_xy))
            tri_vertices = (
                placed_rot.apply(data.vertices[data.triangles[idx]]) + placed_pose[:3]
            )
            candidates.append(
                {
                    "stone_idx": int(placed_idx),
                    "center": centers[idx],
                    "normal": _unit(normals[idx]),
                    "area": float(data.areas[idx]),
                    "xy_dist": dist,
                    "vertices": tri_vertices,
                }
            )

    candidates.sort(
        key=lambda item: (
            item["area"] + 0.25 * item["center"][2] - 0.1 * item["xy_dist"]
        ),
        reverse=True,
    )
    return candidates


def _aggregate_support_patch(support_faces: list[dict], cfg) -> Optional[dict]:
    if not support_faces:
        return None
    weights = np.asarray([max(float(face["area"]), 1e-9) for face in support_faces])
    centers = np.asarray([face["center"] for face in support_faces], dtype=float)
    normals = np.asarray([face["normal"] for face in support_faces], dtype=float)
    normal = _unit(np.average(normals, axis=0, weights=weights))
    if np.linalg.norm(normal) < 1e-9:
        return None
    patch_points = []
    patch_normals = []
    samples_per_face = cfg.icp_samples_per_face
    for face in support_faces:
        points = _sample_triangle(face["vertices"], samples_per_face)
        patch_points.append(points)
        patch_normals.append(np.repeat(face["normal"][None, :], len(points), axis=0))

    return {
        "center": np.average(centers, axis=0, weights=weights),
        "normal": normal,
        "area": float(np.sum(weights)),
        "stone_count": len({face["stone_idx"] for face in support_faces}),
        "points": np.concatenate(patch_points, axis=0),
        "normals": np.concatenate(patch_normals, axis=0),
    }


def _best_face_fit_pose(
    target_faces: MeshFaceData,
    support_patch: dict,
    pose: np.ndarray,
    cfg,
) -> tuple[np.ndarray, dict]:
    debug = {
        "reason": "no_downward_target_face",
        "downward_target_faces": 0,
        "angle_rejected_faces": 0,
        "protrusion_rejected_faces": 0,
        "fit_candidate_faces": 0,
        "icp_fitness": 0.0,
        "icp_rmse": np.inf,
        "icp_pre_min_distance": np.inf,
        "icp_post_min_distance": np.inf,
    }
    R0 = Rotation.from_quat(pose[3:])
    world_normals = R0.apply(target_faces.normals)
    down_indices = np.nonzero(
        world_normals[:, 2] <= -cfg.target_min_down_z
    )[0]
    debug["downward_target_faces"] = int(len(down_indices))
    if len(down_indices) == 0:
        return pose, debug
    debug["reason"] = "no_face_fit_candidate"

    down_indices = down_indices[np.argsort(target_faces.areas[down_indices])[::-1]]
    down_indices = down_indices[: cfg.max_target_faces]

    max_angle = cfg.max_rotation_rad
    clearance = cfg.clearance
    xy_weight = cfg.xy_regularization
    area_weight = cfg.area_weight
    max_lower_protrusion = cfg.max_lower_protrusion
    patch_snap_enabled = cfg.pre_icp_patch_centroid_snap

    best_score = -np.inf
    best_pose = pose
    best_debug = debug
    for target_idx in down_indices:
        local_normal = _unit(target_faces.normals[target_idx])
        local_center = target_faces.centers[target_idx]
        target_area = float(target_faces.areas[target_idx])
        desired_normal = -_unit(support_patch["normal"])
        current_normal = _unit(R0.apply(local_normal))
        delta = _rotation_between(current_normal, desired_normal)
        if delta is None or delta.magnitude() > max_angle:
            debug["angle_rejected_faces"] += 1
            continue

        R_new = delta * R0
        rotated_center = R_new.apply(local_center)
        rotated_vertices = R_new.apply(target_faces.vertices)
        lower_protrusion = float(rotated_center[2] - np.min(rotated_vertices[:, 2]))
        if lower_protrusion > max_lower_protrusion:
            debug["protrusion_rejected_faces"] += 1
            continue
        debug["fit_candidate_faces"] += 1

        support_center = np.asarray(support_patch["center"], dtype=float)
        new_pose = pose.copy()
        new_pose[3:] = R_new.as_quat()
        face_center_xy = new_pose[:2] + rotated_center[:2]
        xy_delta = support_center[:2] - face_center_xy
        max_pre_xy = cfg.max_pre_icp_xy_translation
        if max_pre_xy is not None:
            norm = float(np.linalg.norm(xy_delta))
            if norm > float(max_pre_xy) > 0.0:
                xy_delta = xy_delta * (float(max_pre_xy) / norm)
        new_pose[:2] = new_pose[:2] + xy_delta
        new_pose[2] = float(support_center[2] + clearance - rotated_center[2])
        patch_shift = np.zeros(3, dtype=float)
        if patch_snap_enabled:
            new_pose, patch_shift = _snap_source_patch_to_support_patch(
                target_faces,
                target_idx,
                new_pose,
                support_patch,
                cfg,
            )
        fit_pose, icp_score, icp_debug = _refine_pose_by_patch_icp(
            target_faces,
            target_idx,
            new_pose,
            support_patch,
            cfg,
        )

        fit_rot = Rotation.from_quat(fit_pose[3:])
        fit_center = fit_rot.apply(local_center)
        drift = float(
            np.linalg.norm((fit_pose[:2] + fit_center[:2]) - support_center[:2])
        )
        normal_score = float(np.dot(fit_rot.apply(local_normal), desired_normal))
        score = (
            normal_score
            + area_weight * min(target_area, float(support_patch["area"]))
            + 0.1 * float(support_patch["stone_count"])
            + icp_score
            - xy_weight * drift
            - 0.01 * delta.magnitude()
        )
        if score > best_score:
            best_score = score
            best_pose = fit_pose
            best_debug = {
                "reason": "refined",
                "downward_target_faces": debug["downward_target_faces"],
                "angle_rejected_faces": debug["angle_rejected_faces"],
                "protrusion_rejected_faces": debug["protrusion_rejected_faces"],
                "fit_candidate_faces": debug["fit_candidate_faces"],
                **icp_debug,
                "face_center_xy_shift": float(np.linalg.norm(xy_delta)),
                "patch_centroid_shift": float(np.linalg.norm(patch_shift)),
            }

    return best_pose, best_debug


def _refine_pose_by_patch_icp(
    target_faces: MeshFaceData,
    target_idx: int,
    pose: np.ndarray,
    support_patch: dict,
    cfg,
) -> tuple[np.ndarray, float, dict]:
    """Paper-style local ICP refinement on reduced face patches.

    The Hutter et al. planner first reduces candidates with geometric
    heuristics, then refines each remaining candidate with ICP. Here the
    already-selected lower target face is the source patch and the nearby
    upward support faces are the target patch.
    """
    debug = {
        "icp_fitness": 0.0,
        "icp_rmse": np.inf,
        "icp_pre_snap_min_distance": np.inf,
        "icp_pre_min_distance": np.inf,
        "icp_post_min_distance": np.inf,
        "icp_accepted": False,
        "icp_reject_reason": None,
        "icp_patch_max_translation": 0.0,
        "icp_patch_max_xy_translation": 0.0,
        "icp_rotation_deg": 0.0,
        "pre_icp_contact_shift": 0.0,
    }
    if not cfg.icp_enabled:
        return pose, 0.0, debug

    target_points = np.asarray(support_patch.get("points", []), dtype=float)
    estimation = cfg.icp_estimation
    target_normals = np.asarray(support_patch.get("normals", []), dtype=float)
    needs_normals = estimation == "point_to_plane"
    if len(target_points) < 3 or (
        needs_normals and len(target_normals) != len(target_points)
    ):
        return pose, 0.0, debug

    source_local = _sample_triangle(
        target_faces.vertices[target_faces.triangles[target_idx]],
        cfg.icp_samples_per_face,
    )
    R_pose = Rotation.from_quat(pose[3:])
    source_points = R_pose.apply(source_local) + pose[:3]
    if len(source_points) < 3:
        return pose, 0.0, debug

    threshold = cfg.icp_max_correspondence_distance
    debug["icp_pre_min_distance"] = _min_pairwise_distance(source_points, target_points)
    debug["icp_pre_snap_min_distance"] = debug["icp_pre_min_distance"]
    pose_for_icp = pose.copy()
    if cfg.pre_icp_contact_snap:
        pose_for_icp, source_points, shift = _snap_source_patch_to_contact_distance(
            pose_for_icp,
            source_points,
            target_points,
            threshold,
            cfg,
        )
        debug["pre_icp_contact_shift"] = float(np.linalg.norm(shift))
        debug["icp_pre_min_distance"] = _min_pairwise_distance(
            source_points, target_points
        )

    source = _point_cloud(source_points)
    target = _point_cloud(target_points, target_normals if needs_normals else None)
    if estimation == "point_to_plane":
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif estimation == "point_to_point":
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        raise ValueError(
            "mesh_face_fit.icp_estimation must be 'point_to_point' or 'point_to_plane'"
        )
    iterations = cfg.icp_max_iteration
    try:
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            np.eye(4),
            estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max(iterations, 1)
            ),
        )
    except Exception:
        return pose, 0.0, debug

    debug["icp_fitness"] = float(result.fitness)
    debug["icp_rmse"] = float(result.inlier_rmse)
    min_fitness = cfg.icp_min_fitness
    if (
        not np.isfinite(debug["icp_fitness"])
        or not np.isfinite(debug["icp_rmse"])
        or debug["icp_fitness"] < min_fitness
    ):
        return pose, 0.0, debug

    T = np.asarray(result.transformation, dtype=float)
    R_delta = Rotation.from_matrix(T[:3, :3])
    t_delta = T[:3, 3]
    refined_source = R_delta.apply(source_points) + t_delta
    displacement = refined_source - source_points
    debug["icp_patch_max_translation"] = float(
        np.max(np.linalg.norm(displacement, axis=1))
    )
    debug["icp_patch_max_xy_translation"] = float(
        np.max(np.linalg.norm(displacement[:, :2], axis=1))
    )
    debug["icp_rotation_deg"] = float(np.rad2deg(R_delta.magnitude()))
    reject_reason = _icp_delta_reject_reason(
        R_delta,
        t_delta,
        cfg,
        source_points,
        refined_source,
    )
    if reject_reason is not None:
        debug["icp_reject_reason"] = reject_reason
        return pose, 0.0, debug

    refined = pose_for_icp.copy()
    refined[:3] = R_delta.apply(pose_for_icp[:3]) + t_delta
    refined[3:] = (R_delta * Rotation.from_quat(pose_for_icp[3:])).as_quat()
    debug["icp_post_min_distance"] = _min_pairwise_distance(
        refined_source, target_points
    )
    debug["icp_accepted"] = True

    fitness_weight = cfg.icp_fitness_weight
    rmse_weight = cfg.icp_rmse_weight
    score = fitness_weight * float(result.fitness) - rmse_weight * float(
        result.inlier_rmse
    )
    return refined, score, debug


def _snap_source_patch_to_contact_distance(
    pose: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    threshold: float,
    cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source_points) == 0 or len(target_points) == 0 or threshold <= 0.0:
        return pose, source_points, np.zeros(3, dtype=float)

    source_idx, target_idx, distance = _closest_pair_indices(
        source_points,
        target_points,
    )
    target_distance = cfg.pre_icp_contact_distance
    if not np.isfinite(distance) or distance <= target_distance:
        return pose, source_points, np.zeros(3, dtype=float)

    vector = target_points[target_idx] - source_points[source_idx]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return pose, source_points, np.zeros(3, dtype=float)

    shift = vector * ((distance - target_distance) / norm)
    max_shift = cfg.max_pre_icp_contact_translation
    if max_shift is not None:
        shift_norm = float(np.linalg.norm(shift))
        if shift_norm > float(max_shift) > 0.0:
            shift = shift * (float(max_shift) / shift_norm)

    shifted_pose = pose.copy()
    shifted_pose[:3] = shifted_pose[:3] + shift
    return shifted_pose, source_points + shift[None, :], shift


def _closest_pair_indices(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[int, int, float]:
    diff = source_points[:, None, :] - target_points[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    flat_idx = int(np.argmin(dist2))
    source_idx, target_idx = np.unravel_index(flat_idx, dist2.shape)
    return int(source_idx), int(target_idx), float(np.sqrt(dist2[source_idx, target_idx]))


def _snap_source_patch_to_support_patch(
    target_faces: MeshFaceData,
    target_idx: int,
    pose: np.ndarray,
    support_patch: dict,
    cfg,
) -> tuple[np.ndarray, np.ndarray]:
    source_local = _sample_triangle(
        target_faces.vertices[target_faces.triangles[target_idx]],
        cfg.icp_samples_per_face,
    )
    target_points = np.asarray(support_patch.get("points", []), dtype=float)
    if len(source_local) == 0 or len(target_points) == 0:
        return pose, np.zeros(3, dtype=float)

    rot = Rotation.from_quat(pose[3:])
    source_points = rot.apply(source_local) + pose[:3]
    shift = np.mean(target_points, axis=0) - np.mean(source_points, axis=0)
    max_shift = cfg.max_pre_icp_patch_translation
    if max_shift is not None:
        norm = float(np.linalg.norm(shift))
        if norm > float(max_shift) > 0.0:
            shift = shift * (float(max_shift) / norm)

    shifted = pose.copy()
    shifted[:3] = shifted[:3] + shift
    return shifted, shift


def _icp_delta_ok(
    R_delta: Rotation,
    t_delta: np.ndarray,
    cfg,
    source_points: Optional[np.ndarray] = None,
    refined_source_points: Optional[np.ndarray] = None,
) -> bool:
    return (
        _icp_delta_reject_reason(
            R_delta,
            t_delta,
            cfg,
            source_points,
            refined_source_points,
        )
        is None
    )


def _icp_delta_reject_reason(
    R_delta: Rotation,
    t_delta: np.ndarray,
    cfg,
    source_points: Optional[np.ndarray] = None,
    refined_source_points: Optional[np.ndarray] = None,
) -> Optional[str]:
    max_translation = cfg.icp_max_translation
    max_xy = cfg.icp_max_xy_translation
    if source_points is not None and refined_source_points is not None:
        displacement = np.asarray(refined_source_points) - np.asarray(source_points)
        if float(np.max(np.linalg.norm(displacement, axis=1))) > max_translation:
            return "max_patch_translation"
        if max_xy is not None and float(
            np.max(np.linalg.norm(displacement[:, :2], axis=1))
        ) > float(max_xy):
            return "max_patch_xy_translation"
    else:
        if float(np.linalg.norm(t_delta)) > max_translation:
            return "max_translation"
        if max_xy is not None and float(np.linalg.norm(t_delta[:2])) > float(max_xy):
            return "max_xy_translation"
    if R_delta.magnitude() > cfg.icp_max_rotation_rad:
        return "max_rotation"
    return None


def _point_cloud(points: np.ndarray, normals: Optional[np.ndarray] = None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    return cloud


def _min_pairwise_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.inf
    diff = a[:, None, :] - b[None, :, :]
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=-1))))


def _sample_triangle(vertices: np.ndarray, n: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    points = [
        vertices[0],
        vertices[1],
        vertices[2],
        np.mean(vertices, axis=0),
    ]
    if n <= 4:
        return np.asarray(points[: max(n, 1)], dtype=float)

    # Deterministic low-discrepancy barycentric samples. This avoids random
    # timing/noise inside the action sampler while giving ICP more than corners.
    for i in range(n - 4):
        a = (i + 1) / (n - 3)
        b = ((2 * i + 1) % (n - 3) + 1) / (n - 2)
        if a + b > 1.0:
            a = 1.0 - a
            b = 1.0 - b
        c = 1.0 - a - b
        points.append(a * vertices[0] + b * vertices[1] + c * vertices[2])
    return np.asarray(points, dtype=float)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> Optional[Rotation]:
    a = _unit(a)
    b = _unit(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return Rotation.identity()
    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return None
    return Rotation.from_rotvec(axis / axis_norm * np.arccos(dot))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm
