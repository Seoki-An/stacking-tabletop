from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from agent.config_views import support_config

if TYPE_CHECKING:
    from ..inventory import InventoryManager
    from ..state import State


def extract_contact_points(result) -> np.ndarray:
    """Extract world-frame contact points from a posegen Solution."""
    pts: list = []
    try:
        for v in result.contact_point.values():
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 1 and len(arr) >= 3:
                pts.append(arr[:3])
            elif arr.ndim == 2 and arr.shape[1] >= 3:
                pts.extend(arr[:, :3].tolist())
    except Exception:
        pass
    return np.array(pts, dtype=float) if pts else np.zeros((0, 3), dtype=float)


def support_constraint_ok(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
) -> bool:
    support = support_config(inventory)
    if not support.enabled:
        return True
    if len(state.stone_seq) == 0:
        return True

    target = inventory.stones[stone_idx]
    target_bottom, _ = _stone_world_z_bounds(target, pose)
    ground_z = support.ground_z
    z_tolerance = support.z_tolerance
    if abs(target_bottom - ground_z) <= z_tolerance:
        return True

    supports = supporting_stones(
        inventory,
        state,
        stone_idx,
        pose,
        xy_factor=support.xy_factor,
        z_tolerance=z_tolerance,
    )
    return _support_sources_ok(inventory, state, stone_idx, pose, supports, support)


def posegen_contact_support_ok(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    result,
    body_id_to_stone_idx: Optional[dict[int, int]] = None,
) -> bool:
    """Check support using posegen's actual contacts before falling back to geometry.

    Posegen returns contact points keyed by body id. When those ids match the scene
    bodies, this counts actual contacted supporting stones. If the binding or result
    does not expose useful ownership, contact points are classified against placed
    stone footprints.
    """
    support = support_config(inventory)
    if not support.enabled:
        return True
    if not support.use_posegen_contacts:
        return support_constraint_ok(inventory, state, stone_idx, pose)
    if len(state.stone_seq) == 0:
        return True

    target = inventory.stones[stone_idx]
    target_bottom, _ = _stone_world_z_bounds(target, pose)
    ground_z = support.ground_z
    z_tolerance = support.z_tolerance
    if abs(target_bottom - ground_z) <= z_tolerance:
        return True

    sources = posegen_contact_support_sources(
        inventory,
        state,
        stone_idx,
        pose,
        result,
        body_id_to_stone_idx=body_id_to_stone_idx,
    )
    if sources:
        return _support_sources_ok(
            inventory, state, stone_idx, pose, sources, support
        )

    if support.contact_fallback_to_geometry:
        return support_constraint_ok(inventory, state, stone_idx, pose)
    return False


def posegen_contact_support_sources(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    result,
    body_id_to_stone_idx: Optional[dict[int, int]] = None,
) -> List[int]:
    support = support_config(inventory)
    z_tolerance = support.contact_z_tolerance
    xy_margin = support.contact_xy_margin

    sources = set(
        _contact_sources_from_body_ids(
            inventory,
            state,
            stone_idx,
            pose,
            result,
            body_id_to_stone_idx or {},
            xy_margin=xy_margin,
            z_tolerance=z_tolerance,
            match_tolerance=support.contact_match_tolerance,
        )
    )

    contact_points = extract_contact_points(result)
    if len(contact_points) == 0:
        return sorted(sources)

    sources.update(
        _contact_sources_from_points(
            inventory,
            state,
            stone_idx,
            pose,
            contact_points,
            xy_margin=xy_margin,
            z_tolerance=z_tolerance,
        )
    )
    return sorted(sources)


def _contact_sources_from_body_ids(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    result,
    body_id_to_stone_idx: dict[int, int],
    xy_margin: float,
    z_tolerance: float,
    match_tolerance: float,
) -> List[int]:
    if not body_id_to_stone_idx:
        return []

    sources: set[int] = set()
    target = inventory.stones[stone_idx]
    target_bottom, target_top = _stone_world_z_bounds(target, pose)

    try:
        items = list(result.contact_point.items())
    except Exception:
        return []
    candidate_points = _candidate_contact_points(items, body_id_to_stone_idx)

    for body_id, contacts in items:
        try:
            placed_idx = body_id_to_stone_idx[int(body_id)]
        except Exception:
            continue
        arr = np.asarray(contacts, dtype=float)
        if arr.size == 0:
            continue
        arr = np.atleast_2d(arr)
        placed = inventory.stones[placed_idx]
        placed_pose = np.asarray(
            state.stone_poses.get(placed.id, placed.pose), dtype=float
        )
        placed_bottom, placed_top = _stone_world_z_bounds(placed, placed_pose)
        for point in arr:
            if len(point) < 3:
                continue
            point = np.asarray(point[:3], dtype=float)
            if not _contact_point_in_z_bounds(
                point,
                target_bottom,
                target_top,
                placed_bottom,
                placed_top,
                z_tolerance,
            ):
                continue
            if len(candidate_points) and not _matches_any_point(
                point,
                candidate_points,
                match_tolerance,
            ):
                continue
            if not _point_inside_stone_xy_footprint(target, pose, point[:2], xy_margin):
                continue
            if _point_inside_stone_xy_footprint(
                placed, placed_pose, point[:2], xy_margin
            ):
                sources.add(int(placed_idx))
                break
    return sorted(sources)


def _candidate_contact_points(
    contact_items,
    body_id_to_stone_idx: dict[int, int],
) -> np.ndarray:
    points: list = []
    for body_id, contacts in contact_items:
        try:
            body_id_int = int(body_id)
        except Exception:
            continue
        if body_id_int == -1 or body_id_int in body_id_to_stone_idx:
            continue
        arr = np.asarray(contacts, dtype=float)
        if arr.size == 0:
            continue
        arr = np.atleast_2d(arr)
        points.extend(arr[:, :3].tolist())
    return np.asarray(points, dtype=float) if points else np.zeros((0, 3), dtype=float)


def _matches_any_point(point: np.ndarray, points: np.ndarray, tolerance: float) -> bool:
    return bool(np.any(np.linalg.norm(points - point, axis=1) <= tolerance))


def _contact_sources_from_points(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    contact_points: np.ndarray,
    xy_margin: float,
    z_tolerance: float,
) -> List[int]:
    sources: set[int] = set()
    target = inventory.stones[stone_idx]
    target_bottom, target_top = _stone_world_z_bounds(target, pose)

    for placed_idx in state.stone_seq:
        placed = inventory.stones[placed_idx]
        placed_pose = np.asarray(
            state.stone_poses.get(placed.id, placed.pose), dtype=float
        )
        placed_bottom, placed_top = _stone_world_z_bounds(placed, placed_pose)
        for point in contact_points:
            point = np.asarray(point[:3], dtype=float)
            if not _contact_point_in_z_bounds(
                point,
                target_bottom,
                target_top,
                placed_bottom,
                placed_top,
                z_tolerance,
            ):
                continue
            if not _point_inside_stone_xy_footprint(target, pose, point[:2], xy_margin):
                continue
            if _point_inside_stone_xy_footprint(
                placed, placed_pose, point[:2], xy_margin
            ):
                sources.add(int(placed_idx))
                break
    return sorted(sources)


def _contact_point_in_z_bounds(
    point: np.ndarray,
    target_bottom: float,
    target_top: float,
    placed_bottom: float,
    placed_top: float,
    tolerance: float,
) -> bool:
    """True when a contact point is plausible for both contacting stones.

    Posegen contacts should replace the old target-bottom/placed-top heuristic.
    Keep only a broad z-range sanity check so sloped or side-face contacts are not
    discarded just because they are not on the placed stone's top surface.
    """
    z = float(point[2])
    return bool(
        target_bottom - tolerance <= z <= target_top + tolerance
        and placed_bottom - tolerance <= z <= placed_top + tolerance
    )


def _point_inside_stone_xy_footprint(
    stone,
    pose: np.ndarray,
    point_xy: np.ndarray,
    margin: float,
) -> bool:
    R = Rotation.from_quat(pose[3:]).as_matrix()
    t = pose[:3]
    verts_local, _ = stone.get_global_hull_mesh_array()
    verts_xy = (verts_local @ R.T + t)[:, :2]

    try:
        hull = ConvexHull(verts_xy)
        equations = hull.equations
        return bool(
            np.all(
                equations[:, :2] @ np.asarray(point_xy, dtype=float) + equations[:, 2]
                <= margin
            )
        )
    except Exception:
        xy_limit = float(stone.xy_aabb_radius()) + margin
        return bool(
            np.linalg.norm(np.asarray(point_xy, dtype=float) - pose[:2]) <= xy_limit
        )


def _stone_world_z_bounds(stone, pose: np.ndarray) -> tuple[float, float]:
    verts_local, _ = stone.get_global_hull_mesh_array()
    R = Rotation.from_quat(pose[3:]).as_matrix()
    verts_z = (verts_local @ R.T + pose[:3])[:, 2]
    return float(np.min(verts_z)), float(np.max(verts_z))


def _support_sources_ok(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    supports: List[int],
    support,
) -> bool:
    min_supports = support.min_sources
    if len(supports) >= min_supports:
        return _support_spread_ok(
            inventory, state, stone_idx, pose, supports, support
        )
    if len(supports) == 0:
        return False
    if not support.allow_single_support:
        return False

    target = inventory.stones[stone_idx]
    below = inventory.stones[supports[0]]
    large_ratio = support.large_below_volume_ratio
    return not (below.volume >= large_ratio * max(target.volume, 1e-12))


def _support_spread_ok(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    supports: List[int],
    support,
) -> bool:
    """Reject tower-like supports whose support bodies are too concentrated."""
    if not support.spread_check_enabled:
        return True

    support_indices = [
        int(idx) for idx in supports if isinstance(idx, (int, np.integer))
    ]
    if len(support_indices) < 2:
        return False

    centers = []
    for idx in support_indices:
        stone = inventory.stones[idx]
        placed_pose = np.asarray(
            state.stone_poses.get(stone.id, stone.pose), dtype=float
        )
        centers.append(placed_pose[:2])
    centers = np.asarray(centers, dtype=float)
    target_xy = np.asarray(pose[:2], dtype=float)
    target_radius = max(
        float(np.linalg.norm(0.5 * _stone_xy_extent(inventory.stones[stone_idx]))),
        1e-6,
    )

    pair_distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :],
        axis=-1,
    )
    max_separation = float(np.max(pair_distances))
    min_separation = support.spread_min_separation_scale * target_radius
    if max_separation < min_separation:
        return False

    if len(centers) == 2:
        line_distance = _point_segment_distance(target_xy, centers[0], centers[1])
        max_line_distance = support.spread_max_line_distance_scale * target_radius
        return bool(line_distance <= max_line_distance)

    try:
        hull = ConvexHull(centers)
    except Exception:
        return False

    min_area = support.spread_min_area_scale * target_radius * target_radius
    if float(hull.volume) < min_area:
        return False

    margin = support.spread_com_margin
    signed = hull.equations[:, :2] @ target_xy + hull.equations[:, 2]
    return bool(np.all(signed <= -margin))


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def supporting_stones(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    xy_factor: float,
    z_tolerance: float,
) -> List[int]:
    target = inventory.stones[stone_idx]
    target_bottom, _ = _stone_world_z_bounds(target, pose)

    supports = []
    for placed_idx in state.stone_seq:
        placed = inventory.stones[placed_idx]
        placed_pose = state.stone_poses.get(placed.id, placed.pose)
        _, placed_top = _stone_world_z_bounds(placed, placed_pose)
        if abs(target_bottom - placed_top) > z_tolerance:
            continue
        xy_limit = xy_factor * np.minimum(
            0.5 * (_stone_xy_extent(target) + _stone_xy_extent(placed)),
            _stone_xy_extent(placed),
        )
        if _inside_xy_limit(pose[:2] - placed_pose[:2], xy_limit):
            supports.append(placed_idx)
    return supports


def placement_support_sources(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    xy_factor: float,
    z_tolerance: float,
    ground_z: float = 0.0,
) -> List[int | str]:
    target = inventory.stones[stone_idx]
    target_bottom, _ = _stone_world_z_bounds(target, pose)

    sources: List[int | str] = []
    if abs(target_bottom - float(ground_z)) <= z_tolerance:
        sources.append("ground")
    sources.extend(
        supporting_stones(
            inventory,
            state,
            stone_idx,
            pose,
            xy_factor=xy_factor,
            z_tolerance=z_tolerance,
        )
    )
    return sources


def diffsim_contact_support_sources(
    state: "State",
    stone_idx: int,
    max_gap: float,
) -> List[int | str]:
    """Return support sources for the latest stone from diffsim contacts.

    The simulator stores contacts as Python-native records after the final
    stability simulation. Count only contacts involving the newly placed stone,
    and deduplicate by supporting body.
    """
    sources: set[int | str] = set()
    for contact in getattr(state, "contact_points", []) or []:
        try:
            gap = float(contact.get("gap", np.inf))
        except Exception:
            continue
        if not np.isfinite(gap) or gap > max_gap:
            continue

        idx_1 = contact.get("stone_idx_1", None)
        idx_2 = contact.get("stone_idx_2", None)
        ground_1 = bool(contact.get("is_ground_1", False))
        ground_2 = bool(contact.get("is_ground_2", False))

        if idx_1 == stone_idx:
            if ground_2:
                sources.add("ground")
            elif idx_2 is not None and idx_2 != stone_idx:
                sources.add(int(idx_2))
        elif idx_2 == stone_idx:
            if ground_1:
                sources.add("ground")
            elif idx_1 is not None and idx_1 != stone_idx:
                sources.add(int(idx_1))
    return sorted(sources, key=lambda source: -1 if source == "ground" else int(source))


def diffsim_contact_support_xy(
    state: "State",
    stone_idx: int,
    max_gap: float,
) -> np.ndarray:
    """Return XY contact points where the placed stone is supported."""
    points = []
    for contact in getattr(state, "contact_points", []) or []:
        try:
            gap = float(contact.get("gap", np.inf))
        except Exception:
            continue
        if not np.isfinite(gap) or gap > max_gap:
            continue

        idx_1 = contact.get("stone_idx_1", None)
        idx_2 = contact.get("stone_idx_2", None)
        ground_1 = bool(contact.get("is_ground_1", False))
        ground_2 = bool(contact.get("is_ground_2", False))
        if idx_1 != stone_idx and idx_2 != stone_idx:
            continue
        if idx_1 == stone_idx and idx_2 is None and not ground_2:
            continue
        if idx_2 == stone_idx and idx_1 is None and not ground_1:
            continue

        pts = []
        for key in ("s_1", "s_2"):
            value = np.asarray(contact.get(key, []), dtype=float)
            if value.shape[0] >= 3 and np.all(np.isfinite(value[:3])):
                pts.append(value[:3])
        if not pts:
            continue
        points.append(np.mean(np.asarray(pts, dtype=float), axis=0)[:2])

    if not points:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def placement_support_score(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
) -> tuple[float, int, bool]:
    support = support_config(inventory)
    reward_cfg = support.reward
    xy_factor = support.score_xy_factor
    z_tolerance = support.score_z_tolerance
    ground_z = support.ground_z
    desired_sources = max(support.desired_sources, 1)

    use_diffsim_contacts = bool(reward_cfg.get("use_diffsim_contacts", True))
    fallback_to_geometry = bool(reward_cfg.get("contact_fallback_to_geometry", True))
    sources: List[int | str] = []
    if use_diffsim_contacts and getattr(state, "contact_points", None):
        sources = diffsim_contact_support_sources(
            state,
            stone_idx,
            max_gap=support.contact_gap_tolerance,
        )
        if not sources and not fallback_to_geometry:
            count = 0
            return 0.0, count, False

    if not sources:
        sources = placement_support_sources(
            inventory,
            state,
            stone_idx,
            pose,
            xy_factor=xy_factor,
            z_tolerance=z_tolerance,
            ground_z=ground_z,
        )
    count = len(sources)
    has_ground = any(source == "ground" for source in sources)
    return min(count / desired_sources, 1.0), count, has_ground


def ground_placement_connected_ok(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    pose: np.ndarray,
    support_count: Optional[int] = None,
    support_has_ground: Optional[bool] = None,
) -> bool:
    """Reject late ground-only placements that are isolated from the stack."""
    support = support_config(inventory)
    if not support.connected_ground_enabled:
        return True
    placed_indices = [
        int(idx) for idx in state.stone_seq if int(idx) != int(stone_idx)
    ]
    if len(placed_indices) < support.connected_ground_activate_after:
        return True

    if support_has_ground is None or support_count is None:
        sources = placement_support_sources(
            inventory,
            state,
            stone_idx,
            pose,
            xy_factor=support.score_xy_factor,
            z_tolerance=support.score_z_tolerance,
            ground_z=support.ground_z,
        )
        support_has_ground = any(source == "ground" for source in sources)
        if any(source != "ground" for source in sources):
            return True
        support_count = len(sources)

    if not bool(support_has_ground):
        return True
    if int(support_count) > 1:
        return True

    max_gap = support.connected_ground_max_xy_gap
    xy_factor = support.connected_ground_xy_factor
    target = inventory.stones[int(stone_idx)]
    target_polygon = _stone_world_xy_polygon(target, pose, xy_factor)

    for placed_idx in placed_indices:
        placed = inventory.stones[int(placed_idx)]
        placed_pose = state.stone_poses.get(placed.id, None)
        if placed_pose is None:
            placed_pose = placed.pose
        placed_pose = np.asarray(placed_pose, dtype=float)
        placed_polygon = _stone_world_xy_polygon(
            placed,
            placed_pose,
            xy_factor,
        )
        if _convex_polygon_gap(target_polygon, placed_polygon) <= max_gap:
            return True
    return False


def support_pair_xy(
    inventory: "InventoryManager",
    state: "State",
    k: int,
    stone_idx: Optional[int],
    random_start: bool,
) -> np.ndarray:
    floor_fill_cfg = inventory.cfg.action.planar.get("floor_fill", {})
    support = support_config(inventory)
    pair_z_tolerance = support.pair_z_tolerance
    max_pair_distance_scale = support.pair_distance_scale
    height_order = str(floor_fill_cfg.get("support_pair_height_order", "low"))

    scored = []
    seq = list(state.stone_seq)
    for i, idx_a in enumerate(seq):
        stone_a = inventory.stones[idx_a]
        pose_a = state.stone_poses.get(stone_a.id, stone_a.pose)
        _, top_a = _stone_world_z_bounds(stone_a, pose_a)
        for idx_b in seq[i + 1 :]:
            stone_b = inventory.stones[idx_b]
            pose_b = state.stone_poses.get(stone_b.id, stone_b.pose)
            _, top_b = _stone_world_z_bounds(stone_b, pose_b)
            if abs(top_a - top_b) > pair_z_tolerance:
                continue

            xy_a = np.asarray(pose_a[:2], dtype=float)
            xy_b = np.asarray(pose_b[:2], dtype=float)
            distance = float(np.linalg.norm(xy_a - xy_b))
            max_distance = max_pair_distance_scale * float(
                np.linalg.norm(
                    0.5 * (_stone_xy_extent(stone_a) + _stone_xy_extent(stone_b))
                )
            )
            if distance > max_distance:
                continue

            anchor = 0.5 * (xy_a + xy_b)
            if not _inside_target(inventory, anchor):
                continue
            if _near_wall_boundary(inventory, anchor, stone_idx):
                continue
            pair_height = 0.5 * (top_a + top_b)
            if height_order == "high":
                height_score = pair_height
            elif height_order == "nearest":
                desired = float(
                    floor_fill_cfg.get("support_pair_desired_height", pair_height)
                )
                height_score = -abs(pair_height - desired)
            else:
                height_score = -pair_height
            pair_score = height_score - 0.1 * distance
            scored.append((pair_score, anchor))

    if not scored:
        return np.empty((0, 2), dtype=float)

    scored.sort(key=lambda item: item[0], reverse=True)
    anchors = np.asarray([anchor for _, anchor in scored], dtype=float)
    if random_start and len(anchors) > 0:
        start = np.random.randint(len(anchors))
        anchors = np.concatenate([anchors[start:], anchors[:start]], axis=0)

    k = max(int(k), 1)
    indices = np.arange(k) % len(anchors)
    return anchors[indices]


def planar_support_ok(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    planar_pose: np.ndarray,
) -> bool:
    support = support_config(inventory)
    if not support.enabled:
        return True
    if not support.pre_pose_filter:
        return True

    min_supports = support.min_sources
    activate_after = support.pre_pose_activate_after
    if len(state.stone_seq) < activate_after:
        return True

    support_count = planar_support_count(
        inventory,
        state,
        stone_idx,
        planar_pose[:2],
        xy_factor=support.pre_pose_xy_factor(fallback_to_xy_factor=False),
        z_tolerance=support.pair_z_tolerance,
    )
    if support_count >= min_supports:
        return True
    if _frontier_fill_ok(inventory, state, stone_idx, planar_pose[:2], support_count):
        return True
    if support.pre_pose_allow_empty_space:
        return _is_empty_space_xy(
            inventory,
            state,
            stone_idx,
            planar_pose[:2],
            distance_scale=support.empty_space_distance_scale,
        )
    return False


def planar_support_count(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    xy: np.ndarray,
    xy_factor: float,
    z_tolerance: float,
) -> int:
    nearby_tops = []
    target = inventory.stones[stone_idx]
    for placed_idx in state.stone_seq:
        placed = inventory.stones[placed_idx]
        placed_pose = state.stone_poses.get(placed.id, placed.pose)
        xy_limit = (
            xy_factor * 0.5 * (_stone_xy_extent(target) + _stone_xy_extent(placed))
        )
        if not _inside_xy_limit(
            np.asarray(xy, dtype=float) - placed_pose[:2], xy_limit
        ):
            continue
        _, top = _stone_world_z_bounds(placed, placed_pose)
        nearby_tops.append(top)

    if len(nearby_tops) == 0:
        return 0
    top_ref = max(nearby_tops)
    return int(sum(abs(top - top_ref) <= z_tolerance for top in nearby_tops))


def _frontier_fill_ok(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    xy: np.ndarray,
    support_count: int,
) -> bool:
    support = support_config(inventory)
    if not support.frontier_fill_enabled:
        return False

    min_count = support.frontier_fill_min_support_count
    max_count = support.frontier_fill_max_support_count
    if support_count < min_count or support_count > max_count:
        return False
    if not _inside_target(inventory, xy):
        return False
    if not support.frontier_fill_boundary_only:
        return True

    scale = support.frontier_fill_boundary_inset_radius_scale
    if scale is None:
        return _near_wall_boundary(inventory, xy, stone_idx)
    return _near_wall_boundary_with_scale(inventory, xy, stone_idx, scale)


def _is_empty_space_xy(
    inventory: "InventoryManager",
    state: "State",
    stone_idx: int,
    xy: np.ndarray,
    distance_scale: float,
) -> bool:
    if len(state.stone_seq) == 0:
        return True
    target = inventory.stones[stone_idx]
    for placed_idx in state.stone_seq:
        placed = inventory.stones[placed_idx]
        placed_pose = state.stone_poses.get(placed.id, placed.pose)
        clearance = (
            distance_scale * 0.5 * (_stone_xy_extent(target) + _stone_xy_extent(placed))
        )
        if _inside_xy_limit(np.asarray(xy, dtype=float) - placed_pose[:2], clearance):
            return False
    return True


def _inside_target(inventory: "InventoryManager", xy: np.ndarray) -> bool:
    wall = inventory.target_wall
    origin = np.asarray(wall.origin[:2], dtype=float)
    half_extent = np.array([wall.width, wall.length], dtype=float) / 2.0
    margin = float(inventory.cfg.action.planar.get("target_mask_margin", 0.0))
    return bool(
        np.all(
            (xy >= origin - half_extent - margin)
            & (xy <= origin + half_extent + margin)
        )
    )


def _near_wall_boundary(
    inventory: "InventoryManager",
    xy: np.ndarray,
    stone_idx: Optional[int],
) -> bool:
    """True when xy falls within the boundary band reserved for boundary_fill/empty_corner."""
    wall = inventory.target_wall
    origin = np.asarray(wall.origin[:2], dtype=float)
    half = np.array([wall.width, wall.length], dtype=float) / 2.0
    floor_fill_cfg = inventory.cfg.action.planar.get("floor_fill", {})
    scale = float(floor_fill_cfg.get("boundary_inset_radius_scale", 0.6))
    if stone_idx is None:
        extents = [_stone_xy_extent(stone) for stone in inventory.stones]
        extent = np.mean(extents, axis=0) if extents else np.ones(2, dtype=float)
    else:
        extent = _stone_xy_extent(inventory.stones[int(stone_idx)])
    inset = scale * 0.5 * extent
    lower = origin - half + inset
    upper = origin + half - inset
    xy_arr = np.asarray(xy[:2], dtype=float)
    return bool(np.any(xy_arr < lower) or np.any(xy_arr > upper))


def _near_wall_boundary_with_scale(
    inventory: "InventoryManager",
    xy: np.ndarray,
    stone_idx: Optional[int],
    scale: float,
) -> bool:
    wall = inventory.target_wall
    origin = np.asarray(wall.origin[:2], dtype=float)
    half = np.array([wall.width, wall.length], dtype=float) / 2.0
    if stone_idx is None:
        extents = [_stone_xy_extent(stone) for stone in inventory.stones]
        extent = np.mean(extents, axis=0) if extents else np.ones(2, dtype=float)
    else:
        extent = _stone_xy_extent(inventory.stones[int(stone_idx)])
    inset = float(scale) * 0.5 * extent
    lower = origin - half + inset
    upper = origin + half - inset
    xy_arr = np.asarray(xy[:2], dtype=float)
    return bool(np.any(xy_arr < lower) or np.any(xy_arr > upper))


def _stone_xy_extent(stone) -> np.ndarray:
    return np.asarray(stone.local_aabb_extent()[:2], dtype=float)


def _stone_world_xy_polygon(stone, pose: np.ndarray, scale: float) -> np.ndarray:
    vertices, _ = stone.get_global_hull_mesh_array()
    pose = np.asarray(pose, dtype=float)
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    points = (np.asarray(vertices, dtype=float) @ rotation.T + pose[:3])[:, :2]
    center = pose[:2]
    points = center + float(scale) * (points - center)
    try:
        return points[ConvexHull(points).vertices]
    except Exception:
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        return np.array(
            [
                [lower[0], lower[1]],
                [upper[0], lower[1]],
                [upper[0], upper[1]],
                [lower[0], upper[1]],
            ],
            dtype=float,
        )


def _convex_polygon_gap(first: np.ndarray, second: np.ndarray) -> float:
    """Return the minimum XY distance between two convex polygons."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if _point_in_convex_polygon(first[0], second) or _point_in_convex_polygon(
        second[0], first
    ):
        return 0.0

    first_edges = list(zip(first, np.roll(first, -1, axis=0)))
    second_edges = list(zip(second, np.roll(second, -1, axis=0)))
    for a, b in first_edges:
        for c, d in second_edges:
            if _segments_intersect(a, b, c, d):
                return 0.0

    distances = [
        _point_segment_distance(point, a, b)
        for point in first
        for a, b in second_edges
    ]
    distances.extend(
        _point_segment_distance(point, a, b)
        for point in second
        for a, b in first_edges
    )
    return float(min(distances))


def _point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = np.asarray(point, dtype=float) - polygon
    crosses = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(np.all(crosses >= -1e-9) or np.all(crosses <= 1e-9))


def _segments_intersect(a, b, c, d) -> bool:
    r = np.asarray(b, dtype=float) - a
    s = np.asarray(d, dtype=float) - c
    offset = np.asarray(c, dtype=float) - a
    denominator = r[0] * s[1] - r[1] * s[0]
    offset_cross_r = offset[0] * r[1] - offset[1] * r[0]
    if abs(denominator) <= 1e-9:
        if abs(offset_cross_r) > 1e-9:
            return False
        axis = int(np.argmax(np.abs(r)))
        if abs(r[axis]) <= 1e-9:
            return bool(np.linalg.norm(np.asarray(a) - np.asarray(c)) <= 1e-9)
        lo, hi = sorted(
            ((c[axis] - a[axis]) / r[axis], (d[axis] - a[axis]) / r[axis])
        )
        return hi >= -1e-9 and lo <= 1.0 + 1e-9
    t = (offset[0] * s[1] - offset[1] * s[0]) / denominator
    u = offset_cross_r / denominator
    return bool(-1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9)


def _inside_xy_limit(delta_xy: np.ndarray, limit_xy: np.ndarray) -> bool:
    limit = np.maximum(np.asarray(limit_xy, dtype=float), 1e-6)
    return bool(np.all(np.abs(np.asarray(delta_xy, dtype=float)[:2]) <= limit))
