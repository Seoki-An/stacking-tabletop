from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from agent.config_views import support_config


LOWER_FLOOR_FILL_REJECT_STACKED_KEY = "lower_floor_fill_reject_stacked"
LEGACY_GROUND_FILL_REJECT_STACKED_KEY = "ground_fill_reject_stacked"
HEIGHT_SCORE_SMOOTH_KERNEL_SIZE = 5


def lower_floor_fill_reject_stacked(floor_fill_cfg) -> bool:
    if LOWER_FLOOR_FILL_REJECT_STACKED_KEY in floor_fill_cfg:
        return bool(floor_fill_cfg.get(LOWER_FLOOR_FILL_REJECT_STACKED_KEY, False))
    return bool(floor_fill_cfg.get(LEGACY_GROUND_FILL_REJECT_STACKED_KEY, False))


def migrate_lower_floor_fill_reject_stacked(floor_fill_cfg) -> bool:
    enabled = lower_floor_fill_reject_stacked(floor_fill_cfg)
    floor_fill_cfg[LOWER_FLOOR_FILL_REJECT_STACKED_KEY] = enabled
    if LEGACY_GROUND_FILL_REJECT_STACKED_KEY in floor_fill_cfg:
        del floor_fill_cfg[LEGACY_GROUND_FILL_REJECT_STACKED_KEY]
    return enabled


def active_floor_context(inventory, state, min_count: Optional[int] = None) -> Optional[dict]:
    if min_count is not None and len(state.stone_seq) >= min_count:
        return None

    floor_fill_cfg = inventory.cfg.action.planar.get("floor_fill", {})
    support = support_config(inventory)
    support_z = support.ground_z
    bottom_tol = float(
        floor_fill_cfg.get("lower_floor_bottom_z_tolerance", support.z_tolerance)
    )
    fill_ratio = float(floor_fill_cfg.get("lower_floor_fill_ratio", 0.65))
    grid_size = max(int(floor_fill_cfg.get("lower_floor_occupancy_grid", 32)), 4)
    max_layers = int(floor_fill_cfg.get("lower_floor_max_layers", 8))

    entries = placed_floor_entries(inventory, state)
    last_top = support_z
    used: set[int] = set()
    for _ in range(max(max_layers, 1)):
        layer_items = [
            (idx, entry)
            for idx, entry in enumerate(entries)
            if idx not in used and entry["bottom"] <= support_z + bottom_tol
        ]
        layer = [entry for _, entry in layer_items]
        layer_top = max((entry["top"] for entry in layer), default=support_z)
        last_top = layer_top
        context = occupancy_context(inventory, layer, support_z, bottom_tol, grid_size)
        if context is None:
            return None
        if context["occupancy"] < fill_ratio:
            return context

        for idx, _ in layer_items:
            used.add(idx)
        support_z = layer_top
        if len(used) >= len(entries):
            return occupancy_context(inventory, [], support_z, bottom_tol, grid_size)

    return occupancy_context(inventory, [], last_top, bottom_tol, grid_size)


def active_floor_initial_height_ceiling(inventory, state) -> Optional[float]:
    entries = placed_floor_entries(inventory, state)
    if not entries:
        return None

    floor_fill_cfg = inventory.cfg.action.planar.get("floor_fill", {})
    support = support_config(inventory)
    support_z = support.ground_z
    bottom_tol = float(
        floor_fill_cfg.get("lower_floor_bottom_z_tolerance", support.z_tolerance)
    )
    fill_ratio = float(floor_fill_cfg.get("lower_floor_fill_ratio", 0.65))
    grid_size = max(int(floor_fill_cfg.get("lower_floor_occupancy_grid", 32)), 4)
    max_layers = int(floor_fill_cfg.get("lower_floor_max_layers", 8))
    max_above = float(
        floor_fill_cfg.get("lower_floor_max_initial_z_above_lowest_top", 0.65)
    )

    used: set[int] = set()
    last_top = min(entry["top"] for entry in entries)
    for _ in range(max(max_layers, 1)):
        layer_items = [
            (idx, entry)
            for idx, entry in enumerate(entries)
            if idx not in used and entry["bottom"] <= support_z + bottom_tol
        ]
        layer = [entry for _, entry in layer_items]
        if not layer:
            return last_top + max_above

        layer_top = max(entry["top"] for entry in layer)
        last_top = layer_top
        if target_xy_occupancy(inventory, layer, grid_size) < fill_ratio:
            return layer_top + max_above

        for idx, _ in layer_items:
            used.add(idx)
        support_z = layer_top
        if len(used) >= len(entries):
            return None

    return last_top + max_above


def placed_floor_entries(inventory, state) -> list[dict]:
    entries = []
    for idx in getattr(state, "stone_seq", []) or []:
        resolved = state_stone(inventory, state, idx)
        if resolved is None:
            continue
        stone, stone_id = resolved
        pose = np.asarray(state.stone_poses.get(stone_id, stone.pose), dtype=float)
        if pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
            continue
        entries.append(floor_entry(stone, pose))
    entries.sort(key=lambda item: (item["bottom"], item["top"]))
    return entries


def state_stone(inventory, state, entry):
    idx = int(entry)
    id_to_stone = {int(stone.id): stone for stone in inventory.stones}
    stone_set = getattr(state, "stone_set", None)
    if stone_set is not None and 0 <= idx < len(stone_set):
        stone_id = int(stone_set[idx])
        stone = id_to_stone.get(stone_id)
        if stone is not None:
            return stone, stone_id
        return None
    if 0 <= idx < len(inventory.stones):
        stone = inventory.stones[idx]
        return stone, int(stone.id)
    stone = id_to_stone.get(idx)
    if stone is not None:
        return stone, idx
    return None


def floor_entry(stone, pose: np.ndarray) -> dict:
    vertices = stone_world_vertices(stone, pose)
    xy = vertices[:, :2]
    xy_lower = np.min(xy, axis=0)
    xy_upper = np.max(xy, axis=0)
    z = vertices[:, 2]
    return {
        "xy": np.asarray(pose[:2], dtype=float).copy(),
        "half": np.maximum(0.5 * (xy_upper - xy_lower), 1e-6),
        "extent": np.maximum(xy_upper - xy_lower, 1e-6),
        "bottom": float(np.min(z)),
        "top": float(np.max(z)),
        "xy_aabb": (xy_lower, xy_upper),
        "xy_polygon": convex_xy_polygon(xy),
    }


def stone_world_vertices(stone, pose: np.ndarray) -> np.ndarray:
    verts_local, _ = stone.get_global_hull_mesh_array()
    rot = Rotation.from_quat(pose[3:]).as_matrix()
    return verts_local @ rot.T + pose[:3]


def convex_xy_polygon(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return xy
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return xy
    return xy[hull.vertices]


def target_xy_occupancy(inventory, entries: list[dict], grid_size: int) -> float:
    if not entries:
        return 0.0
    grid = occupancy_grid(inventory, grid_size)
    if grid is None:
        return 0.0
    lower, upper, xx, yy = grid
    occupied = entries_occupancy_mask(entries, lower, upper, xx, yy)
    return float(np.mean(occupied))


def occupancy_context(
    inventory,
    entries: list[dict],
    support_z: float,
    bottom_tol: float,
    grid_size: int,
) -> Optional[dict]:
    grid = occupancy_grid(inventory, grid_size)
    if grid is None:
        return None
    lower, upper, xx, yy = grid
    occupied = entries_occupancy_mask(entries, lower, upper, xx, yy)
    return {
        "support_z": float(support_z),
        "bottom_tol": float(bottom_tol),
        "lower": lower,
        "upper": upper,
        "xx": xx,
        "yy": yy,
        "occupied": occupied,
        "occupied_neighbor": neighbor_mask(occupied),
        "occupancy": float(np.mean(occupied)),
    }


def occupancy_grid(inventory, grid_size: int):
    wall = inventory.target_wall
    origin = np.asarray(wall.origin[:2], dtype=float)
    half = np.asarray([wall.width, wall.length], dtype=float) / 2.0
    lower = origin - half
    upper = origin + half
    if np.any(upper <= lower):
        return None

    xs = np.linspace(lower[0], upper[0], max(int(grid_size), 4))
    ys = np.linspace(lower[1], upper[1], max(int(grid_size), 4))
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return lower, upper, xx, yy


def entries_occupancy_mask(
    entries: list[dict],
    lower: np.ndarray,
    upper: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    occupied = np.zeros(xx.shape, dtype=bool)
    for entry in entries:
        occupied |= footprint_mask(entry, lower, upper, xx, yy)
    return occupied


def footprint_mask(
    entry: dict,
    lower: np.ndarray,
    upper: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    aabb_lower, aabb_upper = entry["xy_aabb"]
    aabb_lower = np.maximum(np.asarray(aabb_lower, dtype=float), lower)
    aabb_upper = np.minimum(np.asarray(aabb_upper, dtype=float), upper)
    if np.any(aabb_upper < aabb_lower):
        return np.zeros(xx.shape, dtype=bool)

    in_aabb = (
        (xx >= aabb_lower[0])
        & (xx <= aabb_upper[0])
        & (yy >= aabb_lower[1])
        & (yy <= aabb_upper[1])
    )
    polygon = np.asarray(entry.get("xy_polygon", []), dtype=float)
    if len(polygon) < 3:
        return in_aabb
    return in_aabb & points_in_convex_polygon(xx, yy, polygon)


def points_in_convex_polygon(xx: np.ndarray, yy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.ones(xx.shape, dtype=bool)
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        edge = b - a
        cross = edge[0] * (yy - a[1]) - edge[1] * (xx - a[0])
        inside &= cross >= -1e-9
    return inside


def neighbor_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = np.zeros(mask.shape, dtype=bool)
    for dx in range(3):
        for dy in range(3):
            if dx == 1 and dy == 1:
                continue
            neighbors |= padded[dx : dx + mask.shape[0], dy : dy + mask.shape[1]]
    return neighbors


def connected_component_labels(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.full(mask.shape, -1, dtype=int)
    sizes: dict[int, int] = {}
    component_id = 0
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        start_y = int(start_y)
        start_x = int(start_x)
        if labels[start_y, start_x] >= 0:
            continue
        queue = deque([(start_y, start_x)])
        labels[start_y, start_x] = component_id
        size = 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if not mask[ny, nx] or labels[ny, nx] >= 0:
                    continue
                labels[ny, nx] = component_id
                queue.append((ny, nx))
        sizes[component_id] = size
        component_id += 1
    return labels, sizes


def component_centers(labels: np.ndarray, sizes: dict[int, int]) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    for component_id in sizes:
        coords = np.argwhere(labels == component_id)
        if len(coords) == 0:
            continue
        centers[int(component_id)] = np.mean(coords, axis=0)
    return centers


def active_layer_fill_metrics(
    inventory,
    layer: dict,
    stone_idx: int,
    pose: np.ndarray,
) -> dict:
    entry = floor_entry(inventory.stones[int(stone_idx)], np.asarray(pose, dtype=float))
    above_active_layer = entry["bottom"] > (
        float(layer["support_z"]) + float(layer["bottom_tol"])
    )
    metrics = {
        "fill_score": 0.0,
        "above_active_layer": bool(above_active_layer),
        "cell_count": 0,
        "unfilled_cells": 0,
        "contact_cells": 0,
        "overlap_cells": 0,
    }
    if above_active_layer:
        return metrics

    candidate_cells = footprint_mask(
        entry,
        layer["lower"],
        layer["upper"],
        layer["xx"],
        layer["yy"],
    )
    cell_count = int(np.count_nonzero(candidate_cells))
    metrics["cell_count"] = cell_count
    if cell_count == 0:
        return metrics
    metrics["overlap_cells"] = int(np.count_nonzero(candidate_cells & layer["occupied"]))
    unfilled = candidate_cells & ~layer["occupied"]
    unfilled_cells = int(np.count_nonzero(unfilled))
    metrics["unfilled_cells"] = unfilled_cells
    contact_cells = int(np.count_nonzero(candidate_cells & layer["occupied_neighbor"]))
    metrics["contact_cells"] = contact_cells
    if unfilled_cells == 0:
        return metrics
    unfilled_ratio = float(np.count_nonzero(unfilled) / cell_count)
    adjacent_gap = unfilled & layer["occupied_neighbor"]
    adjacent_ratio = float(np.count_nonzero(adjacent_gap) / cell_count)
    metrics["fill_score"] = min(unfilled_ratio + 0.5 * adjacent_ratio, 1.0)
    return metrics


def active_layer_fill_score(inventory, layer: dict, stone_idx: int, pose: np.ndarray):
    metrics = active_layer_fill_metrics(inventory, layer, stone_idx, pose)
    return float(metrics["fill_score"]), bool(metrics["above_active_layer"])


def score_xy_debug_map(
    inventory,
    state,
    stone_idx: Optional[int],
    scene_height_map: Optional[np.ndarray] = None,
    active_floor: Optional[dict] = None,
) -> dict:
    height_map = None if scene_height_map is None else np.asarray(scene_height_map)
    shape = score_grid_shape(inventory, height_map)
    xy_grid = xy_grid_for_shape(inventory, shape)
    heights = geometry_height_grid(inventory, state, xy_grid)
    target_lower, target_upper = target_anchor_bounds(inventory, stone_idx)
    valid = (
        np.all((xy_grid >= target_lower) & (xy_grid <= target_upper), axis=-1)
        & np.isfinite(heights)
    )
    heights = masked_mean_filter(
        heights,
        valid,
        kernel_size=HEIGHT_SCORE_SMOOTH_KERNEL_SIZE,
    )
    score_map = np.full(shape, np.nan, dtype=float)
    candidate_mask = np.zeros(shape, dtype=bool)
    height_score_map = np.full(shape, np.nan, dtype=float)
    connectedness_map = np.full(shape, np.nan, dtype=float)
    open_area_map = np.full(shape, np.nan, dtype=float)
    fill_area_map = np.full(shape, np.nan, dtype=float)
    frontier_map = np.full(shape, np.nan, dtype=float)
    target_boundary_map = np.full(shape, np.nan, dtype=float)
    excavator_distance_map = np.full(shape, np.nan, dtype=float)
    if not np.any(valid):
        return {
            "x_coords": xy_grid[0, :, 0].astype(float).copy(),
            "y_coords": xy_grid[:, 0, 1].astype(float).copy(),
            "scores": score_map,
            "valid": valid.copy(),
            "candidate_mask": candidate_mask,
            "candidates": [],
        }

    cfg = inventory.cfg.action.planar.get("floor_fill", {})
    score_cfg = inventory.cfg.action.planar.get("score", {})
    w_height = float(score_cfg.get("height", -1.5))
    w_connectedness = float(score_cfg.get("connectedness", 2.0))
    w_open_area = float(score_cfg.get("open_area", 1.0))
    w_fill_area = float(score_cfg.get("fill_area", 0.0))
    w_frontier = float(score_cfg.get("frontier", 1.0))
    w_target_boundary = float(score_cfg.get("target_boundary", 0.0))
    w_excavator_distance = float(score_cfg.get("excavator_distance", 0.25))
    excavator_xy = score_excavator_xy(score_cfg)
    excavator_distance_axis = str(
        score_cfg.get("excavator_distance_axis", "xy")
    ).lower()

    layer = active_floor if active_floor is not None else active_floor_context(inventory, state)
    gap_labels = None
    gap_sizes: dict[int, int] = {}
    gap_centers: dict[int, np.ndarray] = {}
    connected_after = int(cfg.get("connected_after", 1))
    require_connected = (
        layer is not None
        and np.any(layer["occupied"])
        and len(state.stone_seq) >= connected_after
    )
    min_contact = max(int(cfg.get("u_shape_min_frontier_contact_cells", 1)), 0)
    if layer is not None and np.any(layer["occupied"]):
        gap_labels, gap_sizes = connected_component_labels(~layer["occupied"])
        gap_centers = component_centers(gap_labels, gap_sizes)

    candidate_items = []
    max_unfilled_cells = 1
    for y_idx, x_idx in np.argwhere(valid):
        xy = xy_grid[y_idx, x_idx]
        height = mean_top_height_at_anchor(inventory, heights, xy_grid, xy, stone_idx)
        frontier = 0.0
        open_area = 1.0
        unfilled_cells = 0
        contact_cells = 0
        gap_component = -1
        gap_component_size = 0
        gap_region = None

        if layer is not None and np.any(layer["occupied"]):
            candidate = anchor_occupancy_mask(inventory, layer, xy, stone_idx)
            cell_count = int(np.count_nonzero(candidate))
            if cell_count == 0:
                continue
            unfilled = candidate & ~layer["occupied"]
            unfilled_cells = int(np.count_nonzero(unfilled))
            open_area = float(unfilled_cells / cell_count)
            if open_area <= 0.0:
                continue
            if gap_labels is not None:
                component_ids, component_counts = np.unique(
                    gap_labels[unfilled],
                    return_counts=True,
                )
                valid_components = component_ids >= 0
                if np.any(valid_components):
                    component_ids = component_ids[valid_components]
                    component_counts = component_counts[valid_components]
                    best_idx = int(np.argmax(component_counts))
                    gap_component = int(component_ids[best_idx])
                    gap_component_size = int(gap_sizes.get(gap_component, 0))
                    center = gap_centers.get(gap_component)
                    if center is not None:
                        gap_region = (
                            gap_component,
                            int(y_idx >= center[0]),
                            int(x_idx >= center[1]),
                        )
            max_unfilled_cells = max(max_unfilled_cells, unfilled_cells)
            contact_cells = int(
                np.count_nonzero(candidate & dilated_mask(layer["occupied"]))
            )
            frontier = min(contact_cells / float(cell_count), 1.0)
            if require_connected and min_contact > 0 and contact_cells < min_contact:
                continue
            if bool(cfg.get("u_shape_filter", True)) and creates_u_shape_pocket(
                inventory,
                layer,
                xy,
                stone_idx,
            ):
                continue

        excavator_distance = score_excavator_distance(
            inventory,
            xy,
            excavator_xy,
            excavator_distance_axis,
        )
        boundary_lower, boundary_upper = target_anchor_bounds(
            inventory,
            stone_idx,
            z=height,
        )
        target_boundary = target_boundary_proximity(
            xy,
            boundary_lower,
            boundary_upper,
        )
        candidate_items.append(
            {
                "xy": xy.copy(),
                "height": height,
                "contact_cells": contact_cells,
                "open_area": open_area,
                "unfilled_cells": unfilled_cells,
                "gap_component": gap_component,
                "gap_component_size": gap_component_size,
                "gap_region": gap_region,
                "frontier": frontier,
                "target_boundary": target_boundary,
                "excavator_distance": excavator_distance,
                "grid_index": (int(y_idx), int(x_idx)),
            }
        )

    if not candidate_items:
        return {
            "x_coords": xy_grid[0, :, 0].astype(float).copy(),
            "y_coords": xy_grid[:, 0, 1].astype(float).copy(),
            "scores": score_map,
            "valid": valid.copy(),
            "candidate_mask": candidate_mask,
            "height": heights.copy(),
            "candidates": [],
        }
    item_heights = np.asarray([item["height"] for item in candidate_items], dtype=float)
    h_min = float(np.min(item_heights))
    h_span = max(float(np.max(item_heights) - h_min), 1e-6)
    max_contact_cells = max(max(int(item["contact_cells"]), 0) for item in candidate_items)
    candidates = []
    for item in candidate_items:
        height_term = (float(item["height"]) - h_min) / h_span
        fill_area = float(item["unfilled_cells"] / max_unfilled_cells)
        connectedness = (
            float(item["contact_cells"] / max_contact_cells)
            if max_contact_cells > 0
            else 0.0
        )
        score = (
            w_height * height_term
            + w_connectedness * connectedness
            + w_open_area * item["open_area"]
            + w_fill_area * fill_area
            + w_frontier * item["frontier"]
            + w_target_boundary * item["target_boundary"]
            + w_excavator_distance * item["excavator_distance"]
        )
        y_idx, x_idx = item["grid_index"]
        score_map[y_idx, x_idx] = float(score)
        candidate_mask[y_idx, x_idx] = True
        height_score_map[y_idx, x_idx] = float(height_term)
        connectedness_map[y_idx, x_idx] = float(connectedness)
        open_area_map[y_idx, x_idx] = float(item["open_area"])
        fill_area_map[y_idx, x_idx] = float(fill_area)
        frontier_map[y_idx, x_idx] = float(item["frontier"])
        target_boundary_map[y_idx, x_idx] = float(item["target_boundary"])
        excavator_distance_map[y_idx, x_idx] = float(item["excavator_distance"])
        candidates.append(
            {
                "xy": item["xy"],
                "score": float(score),
                "height": item["height"],
                "gap_component": int(item["gap_component"]),
                "gap_component_size": int(item["gap_component_size"]),
                "gap_region": item["gap_region"],
                "target_boundary": float(item["target_boundary"]),
            }
        )

    return {
        "x_coords": xy_grid[0, :, 0].astype(float).copy(),
        "y_coords": xy_grid[:, 0, 1].astype(float).copy(),
        "scores": score_map,
        "valid": valid.copy(),
        "candidate_mask": candidate_mask,
        "height": heights.copy(),
        "height_term": height_score_map,
        "connectedness": connectedness_map,
        "open_area": open_area_map,
        "fill_area": fill_area_map,
        "frontier": frontier_map,
        "target_boundary": target_boundary_map,
        "excavator_distance": excavator_distance_map,
        "weights": {
            "height": w_height,
            "connectedness": w_connectedness,
            "open_area": w_open_area,
            "fill_area": w_fill_area,
            "frontier": w_frontier,
            "target_boundary": w_target_boundary,
            "excavator_distance": w_excavator_distance,
        },
        "h_min": h_min,
        "h_span": h_span,
        "candidates": candidates,
    }


def score_grid_shape(inventory, height_map: Optional[np.ndarray]) -> tuple[int, int]:
    size = inventory.cfg.action.planar.get("score_map_size", None)
    if size is None:
        if height_map is None:
            return (32, 32)
        return tuple(np.asarray(height_map).shape[:2])
    if isinstance(size, int):
        width = height = max(int(size), 2)
    else:
        width = max(int(size[0]), 2)
        height = max(int(size[1]), 2)
    return height, width


def xy_grid_for_shape(inventory, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x = np.linspace(float(inventory.xlim[0]), float(inventory.xlim[1]), width)
    y = np.linspace(float(inventory.ylim[0]), float(inventory.ylim[1]), height)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack([xx, yy], axis=-1)


def target_diagonal(inventory) -> float:
    wall = inventory.target_wall
    return max(float(np.hypot(wall.width, wall.length)), 1e-6)


def target_axis_span(inventory, axis: int) -> float:
    bounds = inventory.xlim if axis == 0 else inventory.ylim
    return max(float(bounds[1] - bounds[0]), 1e-6)


def score_excavator_xy(score_cfg) -> np.ndarray:
    xy = score_cfg.get("excavator_xy", None)
    if xy is None:
        return np.zeros(2, dtype=float)
    arr = np.asarray(xy, dtype=float).reshape(-1)
    if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
        return np.zeros(2, dtype=float)
    return arr[:2].copy()


def score_excavator_distance(
    inventory,
    xy: np.ndarray,
    excavator_xy: np.ndarray,
    axis: str,
) -> float:
    if axis in ("x", "0"):
        distance = abs(float(xy[0] - excavator_xy[0]))
        scale = target_axis_span(inventory, 0)
    elif axis in ("y", "1"):
        distance = abs(float(xy[1] - excavator_xy[1]))
        scale = target_axis_span(inventory, 1)
    else:
        distance = float(np.linalg.norm(xy - excavator_xy))
        scale = target_diagonal(inventory)
    return min(distance / scale, 1.0)


def mean_top_height_at_anchor(
    inventory,
    top_heights: np.ndarray,
    xy_grid: np.ndarray,
    anchor: np.ndarray,
    stone_idx: Optional[int],
) -> float:
    footprint = anchor_grid_mask(inventory, xy_grid, anchor, stone_idx)
    values = np.asarray(top_heights[footprint], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        xy = np.asarray(anchor, dtype=float)
        distances = np.linalg.norm(xy_grid - xy, axis=-1)
        y_idx, x_idx = np.unravel_index(int(np.argmin(distances)), distances.shape)
        return float(top_heights[y_idx, x_idx])
    return float(np.mean(values))


def masked_mean_filter(
    values: np.ndarray,
    valid: np.ndarray,
    kernel_size: int = HEIGHT_SCORE_SMOOTH_KERNEL_SIZE,
) -> np.ndarray:
    """Smooth a grid without allowing masked neighbors into the mean."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values and valid must be same-shaped 2D grids")
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    included = valid & np.isfinite(values)
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    sums = convolve(
        np.where(included, values, 0.0),
        kernel,
        mode="constant",
        cval=0.0,
    )
    counts = convolve(
        included.astype(float),
        kernel,
        mode="constant",
        cval=0.0,
    )
    smoothed = values.copy()
    np.divide(sums, counts, out=smoothed, where=counts > 0.0)
    return smoothed


def geometry_height_grid(inventory, state, xy_grid: np.ndarray) -> np.ndarray:
    support = support_config(inventory)
    heights = np.full(xy_grid.shape[:2], float(support.ground_z), dtype=float)
    entries = placed_floor_entries(inventory, state)
    if not entries:
        return heights

    lower = np.array([float(inventory.xlim[0]), float(inventory.ylim[0])])
    upper = np.array([float(inventory.xlim[1]), float(inventory.ylim[1])])
    xx = xy_grid[..., 0]
    yy = xy_grid[..., 1]
    for entry in entries:
        mask = footprint_mask(entry, lower, upper, xx, yy)
        heights[mask] = np.maximum(heights[mask], float(entry["top"]))
    return heights


def inside_target_grid(inventory, xy_grid: np.ndarray, stone_idx: Optional[int]) -> np.ndarray:
    lower, upper = target_anchor_bounds(inventory, stone_idx)
    return np.all((xy_grid >= lower) & (xy_grid <= upper), axis=-1)


def target_anchor_bounds(
    inventory,
    stone_idx: Optional[int],
    z: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    wall = inventory.target_wall
    origin = np.asarray(wall.origin[:2], dtype=float)
    half_extent = np.asarray([wall.width, wall.length], dtype=float) / 2.0
    if z is not None:
        relative_z = np.clip(float(z) - float(wall.origin[2]), 0.0, wall.height)
        half_extent -= relative_z * np.tan(max(float(wall.taper), 0.0))
    margin = float(inventory.cfg.action.planar.get("target_mask_margin", 0.0))
    inset = anchor_inset_xy(inventory, stone_idx)
    lower = origin - half_extent + margin + inset
    upper = origin + half_extent - margin - inset
    return lower, upper


def target_boundary_proximity(
    xy: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    xy = np.asarray(xy, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if np.any(upper <= lower) or np.any(xy < lower) or np.any(xy > upper):
        return 0.0
    center = 0.5 * (lower + upper)
    half_extent = 0.5 * (upper - lower)
    normalized = np.abs((xy - center) / half_extent)
    return float(np.clip(np.max(normalized), 0.0, 1.0))


def anchor_grid_mask(
    inventory,
    xy_grid: np.ndarray,
    anchor: np.ndarray,
    stone_idx: Optional[int],
) -> np.ndarray:
    cfg = inventory.cfg.action.planar.get("floor_fill", {})
    scale = float(cfg.get("u_shape_footprint_scale", 0.9))
    half = 0.5 * scale * stone_xy_extent(inventory, stone_idx)
    xy = np.asarray(anchor, dtype=float)
    xx = xy_grid[..., 0]
    yy = xy_grid[..., 1]
    return (
        (xx >= xy[0] - half[0])
        & (xx <= xy[0] + half[0])
        & (yy >= xy[1] - half[1])
        & (yy <= xy[1] + half[1])
    )


def sample_scored_xy(inventory, candidates: list[dict], k: int, random_start: bool) -> np.ndarray:
    k = max(int(k), 1)
    temperature = float(inventory.cfg.action.planar.get("boltzmann_temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError("action.planar.boltzmann_temperature must be positive")
    min_probability = float(
        inventory.cfg.action.planar.get("boltzmann_min_probability", 0.0)
    )
    if min_probability < 0.0:
        raise ValueError("action.planar.boltzmann_min_probability must be >= 0")
    score_cfg = inventory.cfg.action.planar.get("score", {})
    elite_fraction = float(score_cfg.get("elite_fraction", 0.0))
    if elite_fraction < 0.0:
        raise ValueError("action.planar.score.elite_fraction must be >= 0")
    by_mode: dict[tuple[int, int], list[dict]] = {}
    heights = np.asarray([item["height"] for item in candidates], dtype=float)
    h_min = float(np.min(heights))
    h_span = max(float(np.max(heights) - h_min), 1e-6)
    origin = np.asarray(inventory.target_wall.origin[:2], dtype=float)
    for item in candidates:
        xy = np.asarray(item["xy"], dtype=float)
        h_bin = int(np.clip(np.floor(3.0 * (item["height"] - h_min) / h_span), 0, 2))
        quadrant = int(xy[0] >= origin[0]) + 2 * int(xy[1] >= origin[1])
        by_mode.setdefault((h_bin, quadrant), []).append(item)

    selected: list[dict] = []
    selected_ids: set[int] = set()
    select_y_region_representatives(
        candidates,
        selected,
        selected_ids,
        temperature,
        min_probability,
        k,
    )
    select_gap_representatives(
        candidates,
        selected,
        selected_ids,
        temperature,
        min_probability,
        k,
    )
    elite_count = min(int(np.ceil(k * min(elite_fraction, 1.0))), len(candidates))
    if elite_count > 0:
        for item in sorted(candidates, key=lambda item: item["score"], reverse=True)[
            :elite_count
        ]:
            if len(selected) >= k:
                break
            if id(item) in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(id(item))

    modes = sorted(
        by_mode.values(),
        key=lambda items: max(item["score"] for item in items),
        reverse=True,
    )
    for mode_items in modes:
        if len(selected) >= k:
            break
        item = weighted_candidate_choice(mode_items, temperature, min_probability, selected_ids)
        if item is None:
            continue
        selected.append(item)
        selected_ids.add(id(item))

    remaining = [item for item in candidates if id(item) not in selected_ids]
    while len(selected) < k and remaining:
        item = weighted_candidate_choice(remaining, temperature, min_probability, selected_ids)
        if item is None:
            break
        selected.append(item)
        selected_ids.add(id(item))
        remaining = [candidate for candidate in remaining if id(candidate) != id(item)]

    if not selected:
        return np.empty((0, 2), dtype=float)
    if not random_start:
        selected.sort(key=lambda item: item["score"], reverse=True)
    return np.asarray([item["xy"] for item in selected], dtype=float)


def select_y_region_representatives(
    candidates: list[dict],
    selected: list[dict],
    selected_ids: set[int],
    temperature: float,
    min_probability: float,
    k: int,
) -> None:
    if not candidates or len(selected) >= k:
        return
    ys = np.asarray([float(np.asarray(item["xy"], dtype=float)[1]) for item in candidates])
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    if y_max <= y_min:
        return

    by_region: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for item in candidates:
        y = float(np.asarray(item["xy"], dtype=float)[1])
        region = int(np.clip(np.floor(3.0 * (y - y_min) / (y_max - y_min)), 0, 2))
        by_region[region].append(item)

    for region in (2, 1, 0):
        if len(selected) >= k:
            break
        item = weighted_candidate_choice(
            by_region[region],
            temperature,
            min_probability,
            selected_ids,
        )
        if item is None:
            continue
        selected.append(item)
        selected_ids.add(id(item))


def select_gap_representatives(
    candidates: list[dict],
    selected: list[dict],
    selected_ids: set[int],
    temperature: float,
    min_probability: float,
    k: int,
) -> None:
    by_gap_region: dict[tuple[int, int, int], list[dict]] = {}
    for item in candidates:
        gap_region = item.get("gap_region", None)
        if gap_region is None:
            continue
        by_gap_region.setdefault(tuple(gap_region), []).append(item)
    if not by_gap_region:
        return

    max_gap_representatives = min(k, len(by_gap_region))
    gap_groups = sorted(
        by_gap_region.values(),
        key=lambda items: (
            max(int(item.get("gap_component_size", 0)) for item in items),
            max(float(item["score"]) for item in items),
        ),
        reverse=True,
    )
    for items in gap_groups[:max_gap_representatives]:
        item = weighted_candidate_choice(items, temperature, min_probability, selected_ids)
        if item is None:
            continue
        selected.append(item)
        selected_ids.add(id(item))


def weighted_candidate_choice(
    candidates: list[dict],
    temperature: float,
    min_probability: float,
    selected_ids: set[int],
) -> Optional[dict]:
    pool = [item for item in candidates if id(item) not in selected_ids]
    if not pool:
        return None
    scores = np.asarray([item["score"] for item in pool], dtype=float)
    logits = scores / temperature
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    if min_probability > 0.0:
        probs = np.maximum(probs, min_probability)
    probs = probs / np.sum(probs)
    return pool[int(np.random.choice(len(pool), p=probs))]


def creates_u_shape_pocket(inventory, layer: dict, anchor: np.ndarray, stone_idx: Optional[int]) -> bool:
    cfg = inventory.cfg.action.planar.get("floor_fill", {})
    if not bool(cfg.get("u_shape_filter", True)):
        return False

    candidate = anchor_occupancy_mask(inventory, layer, anchor, stone_idx)
    if not np.any(candidate & ~layer["occupied"]):
        return False

    if bool(cfg.get("u_shape_require_frontier_contact", True)):
        contact = candidate & dilated_mask(layer["occupied"])
        min_contact = max(int(cfg.get("u_shape_min_frontier_contact_cells", 1)), 0)
        if min_contact > 0 and int(np.count_nonzero(contact)) < min_contact:
            return True

    occupied = layer["occupied"] | candidate
    return has_u_shape_component(inventory, ~occupied, occupied, candidate)


def has_u_shape_component(
    inventory,
    empty: np.ndarray,
    occupied: np.ndarray,
    candidate: np.ndarray,
) -> bool:
    cfg = inventory.cfg.action.planar.get("floor_fill", {})
    min_empty = int(cfg.get("u_shape_min_empty_cells", 4))
    min_contact = int(cfg.get("u_shape_min_contact_cells", 6))
    max_opening_ratio = float(cfg.get("u_shape_max_opening_ratio", 0.40))

    for component in empty_components(empty):
        if len(component) < min_empty:
            continue
        component_mask = np.zeros(empty.shape, dtype=bool)
        ys, xs = zip(*component)
        component_mask[np.asarray(ys), np.asarray(xs)] = True
        if not np.any(component_mask & candidate) and not adjacent_to(
            component_mask,
            candidate,
        ):
            continue

        contact = int(np.count_nonzero(adjacent_to_mask(component_mask, occupied)))
        if contact < min_contact:
            continue

        opening = int(
            np.count_nonzero(grid_boundary_mask(component_mask.shape) & component_mask)
        )
        if opening / max(contact, 1) <= max_opening_ratio:
            return True
    return False


def anchor_occupancy_mask(
    inventory,
    layer: dict,
    anchor: np.ndarray,
    stone_idx: Optional[int],
) -> np.ndarray:
    cfg = inventory.cfg.action.planar.get("floor_fill", {})
    scale = float(cfg.get("u_shape_footprint_scale", 0.9))
    half = 0.5 * scale * stone_xy_extent(inventory, stone_idx)
    xy = np.asarray(anchor, dtype=float)
    xx = layer["xx"]
    yy = layer["yy"]
    return (
        (xx >= xy[0] - half[0])
        & (xx <= xy[0] + half[0])
        & (yy >= xy[1] - half[1])
        & (yy <= xy[1] + half[1])
    )


def empty_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            component = []
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not visited[ny, nx]
                        and mask[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(component)
    return components


def adjacent_to(mask: np.ndarray, other: np.ndarray) -> bool:
    return bool(np.any(adjacent_to_mask(mask, other)))


def adjacent_to_mask(mask: np.ndarray, other: np.ndarray) -> np.ndarray:
    return mask & dilated_mask(other)


def dilated_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (
        padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
        | padded[:-2, :-2]
        | padded[:-2, 2:]
        | padded[2:, :-2]
        | padded[2:, 2:]
    )


def grid_boundary_mask(shape: tuple[int, int]) -> np.ndarray:
    boundary = np.zeros(shape, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    return boundary


def anchor_inset_xy(inventory, stone_idx: Optional[int]) -> np.ndarray:
    cfg = inventory.cfg.action.planar.get("floor_fill", {})
    scale = float(cfg.get("boundary_inset_radius_scale", 0.6))
    return np.maximum(scale * 0.5 * stone_xy_extent(inventory, stone_idx), 0.0)


def stone_xy_extent(inventory, stone_idx: Optional[int]) -> np.ndarray:
    if stone_idx is None:
        extents = [
            np.asarray(stone.local_aabb_extent()[:2], dtype=float)
            for stone in inventory.stones
        ]
        return np.mean(extents, axis=0) if extents else np.ones(2, dtype=float)
    return np.asarray(
        inventory.stones[int(stone_idx)].local_aabb_extent()[:2],
        dtype=float,
    )
