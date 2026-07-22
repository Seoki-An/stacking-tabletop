import os
from pathlib import Path

import numpy as np


def state_score_map_debug(env, state, scene_height_map=None) -> dict:
    """Build one candidate-independent score map for the search state."""
    try:
        from agent.env.components.action.floor_fill import score_xy_debug_map

        score_map = score_xy_debug_map(
            env.inventory,
            state,
            stone_idx=None,
            scene_height_map=scene_height_map,
        )
        score_map["selected_xy"] = np.empty((0, 2), dtype=float)
        score_map["stone_idx"] = None
        score_map["stone_id"] = None
        score_map["n_candidates"] = int(
            len(score_map.get("candidates", []) or [])
        )
        score_map["scope"] = "state"
        return compact_score_map_debug(score_map)
    except Exception as exc:
        print(f"  warning: failed to build state score map: {exc}")
        return {}


def raw_scene_height_map_debug(env, state) -> np.ndarray | None:
    """Render the unfiltered stacked-stone height map before MCTS simulation."""
    try:
        return np.asarray(env.inventory.get_height_map(state), dtype=float).copy()
    except Exception as exc:
        print(f"  warning: failed to render raw scene height map: {exc}")
        return None


def compact_score_map_debug(score_map: dict) -> dict:
    keep = (
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
        "target_boundary",
        "excavator_distance",
        "selected_xy",
        "weights",
        "h_min",
        "h_span",
        "stone_idx",
        "stone_id",
        "n_candidates",
        "scope",
    )
    return {key: score_map[key] for key in keep if key in score_map}


def save_step_map_images(
    debug_dir: Path,
    step: int,
    score_map: dict,
    raw_height_map: np.ndarray | None = None,
) -> list[str]:
    try:
        out_dir = Path(debug_dir) / "maps"
        paths = [
            _save_state_map_image(out_dir, step, score_map, raw_height_map),
            _save_raw_height_map_image(out_dir, step, raw_height_map),
        ]
        return [path for path in paths if path is not None]
    except Exception as exc:
        print(f"  warning: failed to save map debug images for step {step}: {exc}")
        return []


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _colormap(plt, name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("black")
    return cmap


def _finite_2d(value) -> np.ndarray | None:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _coord_edges(coords: np.ndarray, size: int) -> np.ndarray | None:
    coords = np.asarray(coords, dtype=float).reshape(-1)
    if coords.size != int(size) or coords.size == 0 or not np.all(np.isfinite(coords)):
        return None
    if coords.size == 1:
        return np.asarray([coords[0] - 0.5, coords[0] + 0.5], dtype=float)
    mid = 0.5 * (coords[:-1] + coords[1:])
    first = coords[0] - (mid[0] - coords[0])
    last = coords[-1] + (coords[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]]).astype(float)


def _extent_from_coords(x_coords, y_coords, shape: tuple[int, int]) -> list[float] | None:
    height, width = shape
    x_edges = _coord_edges(np.asarray(x_coords, dtype=float), width)
    y_edges = _coord_edges(np.asarray(y_coords, dtype=float), height)
    if x_edges is None or y_edges is None:
        return None
    return [
        float(x_edges[0]),
        float(x_edges[-1]),
        float(y_edges[0]),
        float(y_edges[-1]),
    ]


def _image_data(data: np.ndarray, valid=None) -> np.ndarray:
    out = np.asarray(data, dtype=float).copy()
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape == out.shape:
            out[~valid] = np.nan
    out[~np.isfinite(out)] = np.nan
    if not np.any(np.isfinite(out)):
        out = np.zeros_like(out, dtype=float)
    return out


def _save_state_map_image(
    out_dir: Path,
    step: int,
    score_map: dict,
    raw_height_map: np.ndarray | None,
) -> str | None:
    items = []
    raw_height = _finite_2d(raw_height_map)
    if raw_height is not None:
        items.append(("Raw stacked-stone height", raw_height, "viridis", "upper"))
    definitions = (
        ("Root smoothed geometry height", "height", "viridis"),
        ("Root normalized height", "height_term", "viridis"),
        ("Root planar score", "scores", "magma"),
    )
    score_shape = None
    for title, key, cmap in definitions:
        data = _finite_2d(score_map.get(key))
        if data is not None:
            score_shape = data.shape if score_shape is None else score_shape
            items.append((title, data, cmap, "lower"))
    if not items:
        return None

    plt = _pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:02d}_state_maps.png"
    fig, axes = plt.subplots(
        1,
        len(items),
        figsize=(5.2 * len(items), 4.8),
        constrained_layout=True,
    )
    if len(items) == 1:
        axes = [axes]
    extent = _extent_from_coords(
        score_map.get("x_coords", []),
        score_map.get("y_coords", []),
        items[0][1].shape if score_shape is None else score_shape,
    )
    for ax, (title, data, cmap, origin) in zip(axes, items):
        plotted = _image_data(data)
        finite = plotted[np.isfinite(plotted)]
        value_range = ""
        if finite.size:
            value_range = f"\n{float(finite.min()):.3f} to {float(finite.max()):.3f}"
        image = ax.imshow(
            plotted,
            origin=origin,
            cmap=_colormap(plt, cmap),
            extent=extent,
            aspect="auto" if extent is not None else None,
        )
        ax.set_title(title + value_range)
        if extent is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _save_raw_height_map_image(
    out_dir: Path,
    step: int,
    raw_height_map: np.ndarray | None,
) -> str | None:
    height = _finite_2d(raw_height_map)
    if height is None:
        return None

    plt = _pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:02d}_height_map_raw.png"
    plt.imsave(
        path,
        _image_data(height),
        cmap=_colormap(plt, "viridis"),
        origin="upper",
    )
    return str(path)
