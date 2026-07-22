import argparse
import copy
import inspect
import os
import pathlib
import sys

import numpy as np
import open3d as o3d
from omegaconf import OmegaConf
import pyvista as pv
from scipy.spatial import ConvexHull

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import perception.dsf_fit as dsf_fit
from perception.dsf_fit.support_function import DiffSupport
from perception.reconstruction_dsf import manually_remove_points
from perception.reconstruction_mesh import pcd_to_refined_mesh, visualize_mesh_with_pcd
from perception.utils.reconstruction import (
    compute_hull_inertial_parameters,
    remove_detached_clusters,
    remove_radial_fringe_points,
)
from utils.wavefront import WavefrontExporter


def _require_coacd():
    try:
        import coacd
    except ImportError as exc:
        raise ImportError(
            "CoACD is required for convex decomposition. Install it with "
            "`pip install coacd` in this environment."
        ) from exc
    return coacd


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _clean_part_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Expected convex vertices with shape (N, 3), got {vertices.shape}"
        )
    if len(vertices) < 4:
        raise ValueError("Convex part has fewer than 4 vertices.")
    return vertices


def _clean_part_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected convex faces with shape (N, 3), got {faces.shape}")
    return faces


def decompose_mesh_coacd(
    mesh: o3d.geometry.TriangleMesh,
    threshold: float,
    max_convex_hull: int,
    preprocess_mode: str,
    preprocess_resolution: int,
    resolution: int,
    mcts_nodes: int,
    mcts_iterations: int,
    mcts_max_depth: int,
    pca: bool,
    merge: bool,
    decimate: bool,
    seed: int,
    real_metric: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    coacd = _require_coacd()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if len(vertices) < 4 or len(faces) == 0:
        raise ValueError(
            "Reconstructed mesh is empty; cannot run convex decomposition."
        )

    coacd_mesh = coacd.Mesh(vertices, faces)
    kwargs = {
        "threshold": threshold,
        "max_convex_hull": max_convex_hull,
        "preprocess_mode": preprocess_mode,
        "preprocess_resolution": preprocess_resolution,
        "resolution": resolution,
        "mcts_nodes": mcts_nodes,
        "mcts_iterations": mcts_iterations,
        "mcts_max_depth": mcts_max_depth,
        "pca": pca,
        "merge": merge,
        "decimate": decimate,
        "seed": seed,
        "real_metric": real_metric,
    }
    try:
        signature = inspect.signature(coacd.run_coacd)
        accepted = set(signature.parameters)
        kwargs = {key: val for key, val in kwargs.items() if key in accepted}
    except (TypeError, ValueError):
        pass

    parts = coacd.run_coacd(coacd_mesh, **kwargs)
    return [
        (_clean_part_vertices(part_vertices), _clean_part_faces(part_faces))
        for part_vertices, part_faces in parts
    ]


def split_points_by_convex_parts(
    points: np.ndarray,
    convex_parts: list[tuple[np.ndarray, np.ndarray]],
    inside_tolerance: float,
    min_points_per_part: int,
) -> list[np.ndarray]:
    if len(points) == 0:
        return []

    halfspaces = []
    centers = []
    for vertices, _ in convex_parts:
        hull = ConvexHull(vertices)
        halfspaces.append((hull.equations[:, :3], hull.equations[:, 3]))
        centers.append(vertices.mean(axis=0))
    centers = np.asarray(centers)

    signed_distances = np.empty((len(points), len(convex_parts)), dtype=np.float64)
    for idx, (A, b) in enumerate(halfspaces):
        signed_distances[:, idx] = np.max(points @ A.T + b[None, :], axis=1)

    inside = signed_distances <= inside_tolerance
    assignments = np.argmin(signed_distances, axis=1)
    any_inside = inside.any(axis=1)
    if np.any(any_inside):
        masked = np.where(inside, signed_distances, np.inf)
        assignments[any_inside] = np.argmin(masked[any_inside], axis=1)

    if np.any(~any_inside):
        fallback_dist = np.linalg.norm(
            points[~any_inside, None, :] - centers[None, :, :],
            axis=2,
        )
        assignments[~any_inside] = np.argmin(fallback_dist, axis=1)

    point_sets = [points[assignments == idx] for idx in range(len(convex_parts))]
    for idx, part_points in enumerate(point_sets):
        if len(part_points) >= min_points_per_part:
            continue
        fallback_dist = np.linalg.norm(points - centers[idx], axis=1)
        nearest_count = min(max(min_points_per_part, 4), len(points))
        nearest_idx = np.argsort(fallback_dist)[:nearest_count]
        point_sets[idx] = points[nearest_idx]

    return point_sets


def split_points_by_seed_points(
    points: np.ndarray,
    seed_points: np.ndarray,
    min_points_per_part: int,
) -> list[np.ndarray]:
    seed_points = np.asarray(seed_points, dtype=np.float64)
    if seed_points.ndim != 2 or seed_points.shape[1] != 3 or len(seed_points) == 0:
        raise ValueError("Manual split seeds must have shape (N, 3) with N > 0.")
    distances = np.linalg.norm(points[:, None, :] - seed_points[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    point_sets = [points[assignments == idx] for idx in range(len(seed_points))]

    for idx, part_points in enumerate(point_sets):
        if len(part_points) >= min_points_per_part:
            continue
        nearest_count = min(max(min_points_per_part, 4), len(points))
        nearest_idx = np.argsort(distances[:, idx])[:nearest_count]
        point_sets[idx] = points[nearest_idx]

    return point_sets


def get_dsf_fit_point_sets(
    points: np.ndarray,
    convex_parts: list[tuple[np.ndarray, np.ndarray]],
    fit_source: str,
    inside_tolerance: float,
    min_points_per_part: int,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    coacd_vertex_sets = [vertices for vertices, _ in convex_parts]
    if fit_source == "coacd_vertices":
        return coacd_vertex_sets, None

    split_point_sets = split_points_by_convex_parts(
        points,
        convex_parts,
        inside_tolerance=inside_tolerance,
        min_points_per_part=min_points_per_part,
    )
    if fit_source == "split_pcd":
        return split_point_sets, split_point_sets
    if fit_source == "both":
        combined = [
            np.vstack((part_points, part_vertices))
            for part_points, part_vertices in zip(split_point_sets, coacd_vertex_sets)
        ]
        return combined, split_point_sets

    raise ValueError(f"Unknown DSF fit source: {fit_source}")


def fit_dsf_parts(
    dsf_cfg,
    point_sets: list[np.ndarray],
    vertex_counts: list[int] | None = None,
) -> list[DiffSupport]:
    if vertex_counts is not None and len(vertex_counts) != len(point_sets):
        raise ValueError(
            "vertex_counts must match the number of DSF point sets: "
            f"{len(vertex_counts)} != {len(point_sets)}"
        )

    fitted_parts = []
    for idx, points in enumerate(point_sets):
        part_cfg = dsf_cfg
        vertex_count_desc = ""
        if vertex_counts is not None:
            part_cfg = copy.deepcopy(dsf_cfg)
            part_cfg.optimizer.num_vertices = vertex_counts[idx]
            vertex_count_desc = f", vertices={vertex_counts[idx]}"
        print(
            f"Fitting DSF for PCD split {idx + 1}/{len(point_sets)} "
            f"({len(points)} points{vertex_count_desc})"
        )
        dsf = dsf_fit.optimize_shape(part_cfg, points.T)
        fitted_parts.append(dsf)
    return fitted_parts


def distribute_dsf_vertex_budget(total_vertices: int, n_parts: int) -> list[int]:
    if n_parts < 1:
        return []
    min_vertices_per_part = 4
    min_total = min_vertices_per_part * n_parts
    if total_vertices < min_total:
        raise ValueError(
            f"DSF vertex budget {total_vertices} is too small for {n_parts} "
            f"convex parts. Need at least {min_total} vertices."
        )

    base = total_vertices // n_parts
    remainder = total_vertices % n_parts
    return [base + (1 if idx < remainder else 0) for idx in range(n_parts)]


def export_dsf_parts(
    output_path: pathlib.Path,
    dsf_parts: list[DiffSupport],
    mass: float,
    inertia: np.ndarray,
    mu: float,
) -> None:
    exporter = WavefrontExporter(
        output_path,
        {
            "mass": mass,
            "inertia": inertia,
            "dsf_count": len(dsf_parts),
        },
    )
    for idx, dsf in enumerate(dsf_parts):
        exporter.add_convex(
            vertices=np.array(dsf.v.T),
            sharpness=int(dsf.p),
            mu=mu,
            name=f"convex.{idx}",
        )


def choose_coacd_max_convex_hull(
    pcd_file: pathlib.Path,
    pcd: o3d.geometry.PointCloud,
    default_value: int,
) -> int:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return default_value

    selected = {"value": int(default_value)}
    plotter = pv.Plotter(title=f"CoACD max hulls: {pcd_file.name}")
    plotter.add_axes_at_origin(line_width=2)
    cloud = pv.PolyData(points)
    cloud["z"] = points[:, 2]
    plotter.add_points(
        cloud,
        scalars="z",
        point_size=4,
        render_points_as_spheres=False,
        name="pcd",
    )
    text_actor = plotter.add_text(
        _coacd_selection_text(pcd_file.name, selected["value"]),
        position=(10, 10),
        font_size=10,
        name="coacd_selection_text",
    )

    def update(delta: int):
        selected["value"] = max(1, selected["value"] + delta)
        text_actor.SetInput(_coacd_selection_text(pcd_file.name, selected["value"]))
        plotter.render()

    def on_key_press(obj, _event):
        key = obj.GetKeySym()
        if key in ("Insert", "plus", "equal", "KP_Add", "Up", "Right", "i"):
            update(1)
        elif key in ("Delete", "minus", "KP_Subtract", "Down", "Left", "d"):
            update(-1)
        elif key in ("Return", "space", "Escape", "q"):
            obj.TerminateApp()

    plotter.iren.interactor.AddObserver("KeyPressEvent", on_key_press)
    plotter.show()
    plotter.close()
    print(f"Using coacd_max_convex_hull={selected['value']} for {pcd_file}")
    return selected["value"]


def choose_reconstruct_pcd(
    pcd_file: pathlib.Path,
    pcd: o3d.geometry.PointCloud,
) -> bool:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return False

    selected = {"reconstruct": True}
    plotter = pv.Plotter(title=f"Reconstruct PCD: {pcd_file.name}")
    plotter.add_axes_at_origin(line_width=2)
    cloud = pv.PolyData(points)
    cloud["z"] = points[:, 2]
    plotter.add_points(
        cloud,
        scalars="z",
        point_size=4,
        render_points_as_spheres=False,
        name="pcd",
    )
    plotter.add_text(
        f"{pcd_file.name}\n\n" "Enter / R: reconstruct\n" "S / Delete: skip this PCD",
        position="upper_left",
        font_size=10,
    )

    def on_key_press(obj, _event):
        key = obj.GetKeySym()
        if key in ("Return", "space", "r", "R"):
            selected["reconstruct"] = True
            obj.TerminateApp()
        elif key in ("s", "S", "Delete"):
            selected["reconstruct"] = False
            obj.TerminateApp()

    plotter.iren.interactor.AddObserver("KeyPressEvent", on_key_press)
    plotter.show()
    plotter.close()

    return selected["reconstruct"]


def choose_manual_split_seeds(
    pcd_file: pathlib.Path,
    pcd: o3d.geometry.PointCloud,
    left_clicking: bool = False,
) -> np.ndarray:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    picked_points = []
    plotter = pv.Plotter(title=f"Manual DSF split seeds: {pcd_file.name}")
    plotter.add_axes_at_origin(line_width=2)
    cloud = pv.PolyData(points)
    cloud["z"] = points[:, 2]
    plotter.add_points(
        cloud,
        scalars="z",
        point_size=4,
        render_points_as_spheres=False,
        name="pcd",
    )
    plotter.add_text(
        "Manual split seeds\n"
        "Pick one seed near each desired DSF part.\n"
        "Press P over a region to add a seed.\n"
        "Close the window when done.",
        position="upper_left",
        font_size=10,
    )

    def on_pick(point):
        if point is None:
            return
        point = np.asarray(point, dtype=np.float64)
        picked_points.append(point)
        seed_idx = len(picked_points)
        plotter.add_mesh(
            pv.Sphere(radius=0.035, center=point),
            color="red",
            opacity=0.45,
            name=f"manual_split_seed_{seed_idx}",
            pickable=False,
        )
        print(f"[manual split] seed #{seed_idx}: {point.tolist()}")

    plotter.enable_point_picking(
        callback=on_pick,
        show_message=(
            "Press P over each desired decomposition region. "
            "Close the window when done."
        ),
        left_clicking=left_clicking,
        show_point=True,
        point_size=12,
        color="red",
    )
    plotter.show()

    if not picked_points:
        print("[manual split] No seeds selected.")
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(picked_points, dtype=np.float64)


def get_manual_split_seeds(
    pcd_file: pathlib.Path,
    pcd: o3d.geometry.PointCloud,
    seed_path: pathlib.Path | None,
    left_clicking: bool,
    force_pick: bool,
) -> np.ndarray:
    if seed_path is not None and seed_path.is_file() and not force_pick:
        seeds = np.load(seed_path)
        print(f"[manual split] loaded {len(seeds)} seeds from {seed_path}")
        return np.asarray(seeds, dtype=np.float64)

    seeds = choose_manual_split_seeds(
        pcd_file,
        pcd,
        left_clicking=left_clicking,
    )
    if seed_path is not None and len(seeds) > 0:
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(seed_path, seeds)
        print(f"[manual split] saved {len(seeds)} seeds to {seed_path}")
    return seeds


def _coacd_selection_text(pcd_name: str, value: int) -> str:
    return (
        f"{pcd_name}\n"
        f"coacd_max_convex_hull: {value}\n\n"
        "Insert / + / Up: increase\n"
        "Delete / - / Down: decrease\n"
        "Enter / Space / Q: accept"
    )


def _add_pcd_to_plotter(
    plotter: pv.Plotter,
    pcd: o3d.geometry.PointCloud,
    color: str = "grey",
) -> None:
    points = np.asarray(pcd.points)
    if len(points) > 0:
        cloud = pv.PolyData(points)
        plotter.add_points(
            cloud,
            color=color,
            point_size=4,
            render_points_as_spheres=False,
            name="pcd",
        )


def visualize_convex_decomposition(
    pcd: o3d.geometry.PointCloud,
    convex_parts: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    plotter = pv.Plotter(title="CoACD convex decomposition")
    plotter.add_axes_at_origin(line_width=2)
    _add_pcd_to_plotter(plotter, pcd)
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

    for idx, (vertices, faces) in enumerate(convex_parts):
        pv_faces = np.c_[np.full(len(faces), 3), faces]
        plotter.add_mesh(
            pv.PolyData(vertices, pv_faces),
            opacity=0.3,
            color=colors[idx % len(colors)],
            show_edges=True,
            name=f"convex_{idx}",
        )

    plotter.show()


def visualize_pcd_splits(point_sets: list[np.ndarray]) -> None:
    plotter = pv.Plotter(title="PCD split by CoACD convexes")
    plotter.add_axes_at_origin(line_width=2)
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    for idx, points in enumerate(point_sets):
        if len(points) == 0:
            continue
        plotter.add_points(
            pv.PolyData(points),
            color=colors[idx % len(colors)],
            point_size=5,
            render_points_as_spheres=False,
            name=f"pcd_split_{idx}",
        )
    plotter.show()


def visualize_dsf_decomposition(
    pcd: o3d.geometry.PointCloud,
    dsf_parts: list[DiffSupport],
    allow_reselect: bool = False,
) -> bool:
    selected = {"accept": True}
    plotter = pv.Plotter(title="Fitted DSF decomposition")
    plotter.add_axes_at_origin(line_width=2)
    _add_pcd_to_plotter(plotter, pcd)
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

    for idx, dsf in enumerate(dsf_parts):
        dsf.render(
            plotter,
            opacity=0.3,
            color=colors[idx % len(colors)],
            name=f"dsf_{idx}",
        )

    if allow_reselect:
        plotter.add_text(
            "Enter / Space: accept\nR: re-select convex count",
            position="upper_left",
            font_size=10,
        )

    def on_key_press(obj, _event):
        key = obj.GetKeySym()
        if key in ("Return", "space", "q", "Escape"):
            selected["accept"] = True
            obj.TerminateApp()
        elif allow_reselect and key in ("r", "R"):
            selected["accept"] = False
            obj.TerminateApp()

    plotter.iren.interactor.AddObserver("KeyPressEvent", on_key_press)
    plotter.show()
    plotter.close()
    return selected["accept"]


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--disable_filter",
        action="store_true",
        help="Skip all point cloud filtering steps and run reconstruction on the raw merged point cloud.",
    )
    parser.add_argument(
        "--disable_statistical_outlier",
        action="store_true",
        help="Skip Open3D statistical outlier removal.",
    )
    parser.add_argument(
        "--stat_nb_neighbors",
        default=20,
        type=int,
        help="Neighbor count for statistical outlier removal.",
    )
    parser.add_argument(
        "--stat_std_ratio",
        default=2.0,
        type=float,
        help="Higher values preserve more sparse surface points.",
    )
    parser.add_argument(
        "--disable_detached_cluster_filter",
        action="store_true",
        help="Skip removal of small disconnected point islands.",
    )
    parser.add_argument(
        "--detached_verbose",
        action="store_true",
        help="Print detached-cluster keep/remove diagnostics.",
    )
    parser.add_argument("--detached_voxel", default=0.01, type=float)
    parser.add_argument("--detached_eps", default=0.035, type=float)
    parser.add_argument("--detached_min_points", default=6, type=int)
    parser.add_argument("--detached_attach_distance", default=0.08, type=float)
    parser.add_argument("--detached_min_attached_size", default=80, type=int)
    parser.add_argument("--detached_keep_ratio", default=0.10, type=float)
    parser.add_argument(
        "--disable_radial_fringe_filter",
        action="store_true",
        help="Skip radial outward fringe removal.",
    )
    parser.add_argument("--fringe_k_neighbors", default=80, type=int)
    parser.add_argument("--fringe_margin", default=0.04, type=float)
    parser.add_argument("--fringe_reference_quantile", default=0.65, type=float)
    parser.add_argument(
        "--disable_cluster_filter",
        action="store_true",
        help="Skip DBSCAN clustering after voxel downsampling.",
    )
    parser.add_argument("--cluster_eps", default=0.05, type=float)
    parser.add_argument("--cluster_min_points", default=8, type=int)
    parser.add_argument(
        "--manual_remove",
        action="store_true",
        help="Open an interactive PyVista picker before reconstruction.",
    )
    parser.add_argument("--manual_remove_distance", default=0.04, type=float)
    parser.add_argument("--manual_max_remove_ratio", default=0.5, type=float)
    parser.add_argument("--manual_left_click", action="store_true")


def add_mesh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mesh_method",
        default="radial",
        choices=["radial", "alpha", "bpa", "poisson"],
        help="Surface reconstruction method before convex decomposition.",
    )
    parser.add_argument("--mesh_reconstruction_voxel", default=0.02, type=float)
    parser.add_argument("--mesh_alpha", default=0.10, type=float)
    parser.add_argument("--mesh_bpa_radius_scales", default="1.5,2.5,4.0")
    parser.add_argument("--mesh_radial_subdivisions", default=4, type=int)
    parser.add_argument("--mesh_radial_k_neighbors", default=120, type=int)
    parser.add_argument("--mesh_radial_radius_quantile", default=0.7, type=float)
    parser.add_argument("--mesh_radial_radius_padding", default=0.0, type=float)
    parser.add_argument("--mesh_radial_center_quantile", default=0.5, type=float)
    parser.add_argument("--mesh_radial_empty_angle_deg", default=25.0, type=float)
    parser.add_argument("--mesh_radial_empty_radius_quantile", default=0.75, type=float)
    parser.add_argument("--mesh_radial_support_angle_deg", default=8, type=float)
    parser.add_argument("--mesh_radial_min_support_points", default=0, type=int)
    parser.add_argument("--mesh_radial_tighten_angle_deg", default=5.0, type=float)
    parser.add_argument("--mesh_radial_tighten_quantile", default=0.9, type=float)
    parser.add_argument("--mesh_radial_tighten_blend", default=1.0, type=float)
    parser.add_argument("--mesh_radial_max_radius_quantile", default=0.99, type=float)
    parser.add_argument("--mesh_radial_radius_smooth_iterations", default=3, type=int)
    parser.add_argument("--mesh_radial_radius_smooth_weight", default=0.1, type=float)
    parser.add_argument("--mesh_normal_radius", default=0.05, type=float)
    parser.add_argument("--mesh_normal_max_nn", default=40, type=int)
    parser.add_argument("--mesh_poisson_depth", default=12, type=int)
    parser.add_argument("--mesh_density_quantile", default=0.02, type=float)
    parser.add_argument("--mesh_smooth_iterations", default=2, type=int)
    parser.add_argument("--mesh_simplify_voxel", default=0.0, type=float)
    parser.add_argument(
        "--keep_mesh_components",
        action="store_true",
        help="Keep all connected mesh components instead of only the largest one.",
    )


def add_coacd_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--coacd_threshold",
        default=0.02,
        type=float,
        help="CoACD concavity threshold. With --coacd_real_metric, this is in meters.",
    )
    parser.add_argument("--coacd_max_convex_hull", default=3, type=int)
    parser.add_argument(
        "--coacd_preprocess_mode",
        default="auto",
        choices=["auto", "on", "off"],
    )
    parser.add_argument("--coacd_resolution", default=2000, type=int)
    parser.add_argument("--coacd_preprocess_resolution", default=50, type=int)
    parser.add_argument("--coacd_mcts_nodes", default=20, type=int)
    parser.add_argument("--coacd_mcts_iterations", default=150, type=int)
    parser.add_argument("--coacd_mcts_max_depth", default=3, type=int)
    parser.add_argument("--coacd_pca", action="store_true")
    parser.add_argument("--coacd_no_merge", action="store_true")
    parser.add_argument("--coacd_decimate", action="store_true")
    parser.add_argument("--coacd_seed", default=0, type=int)
    parser.add_argument(
        "--prompt_coacd_max_convex_hull",
        action="store_true",
        help=(
            "Show each centered cleaned PCD before reconstruction and choose "
            "that stone's CoACD max convex hull count in the viewer."
        ),
    )
    parser.add_argument(
        "--coacd_real_metric",
        action="store_true",
        help="Interpret CoACD threshold in real metric units.",
    )


def clean_point_cloud(
    pcd: o3d.geometry.PointCloud,
    args: argparse.Namespace,
) -> o3d.geometry.PointCloud:
    if not args.disable_statistical_outlier:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.stat_nb_neighbors,
            std_ratio=args.stat_std_ratio,
        )
    if not args.disable_detached_cluster_filter:
        pcd = remove_detached_clusters(
            pcd,
            voxel=args.detached_voxel,
            eps=args.detached_eps,
            min_points=args.detached_min_points,
            attach_distance=args.detached_attach_distance,
            min_attached_size=args.detached_min_attached_size,
            keep_ratio=args.detached_keep_ratio,
            verbose=args.detached_verbose,
        )
    if not args.disable_radial_fringe_filter:
        pcd = remove_radial_fringe_points(
            pcd,
            k_neighbors=args.fringe_k_neighbors,
            margin=args.fringe_margin,
            reference_quantile=args.fringe_reference_quantile,
        )
    if not args.disable_cluster_filter:
        pcd = pcd.voxel_down_sample(0.005)
        labels = np.asarray(
            pcd.cluster_dbscan(
                eps=args.cluster_eps,
                min_points=args.cluster_min_points,
                print_progress=True,
            )
        )
        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            largest = labels == np.bincount(valid_labels).argmax()
            pcd = pcd.select_by_index(np.where(largest)[0])
    if args.manual_remove:
        pcd = manually_remove_points(
            pcd,
            distance=args.manual_remove_distance,
            max_remove_ratio=args.manual_max_remove_ratio,
            left_clicking=args.manual_left_click,
        )
    return pcd


def reconstruct_mesh_for_decomposition(
    pcd: o3d.geometry.PointCloud,
    args: argparse.Namespace,
) -> o3d.geometry.TriangleMesh:
    return pcd_to_refined_mesh(
        pcd,
        method=args.mesh_method,
        reconstruction_voxel=args.mesh_reconstruction_voxel,
        alpha=args.mesh_alpha,
        bpa_radius_scales=_parse_float_tuple(args.mesh_bpa_radius_scales),
        radial_subdivisions=args.mesh_radial_subdivisions,
        radial_k_neighbors=args.mesh_radial_k_neighbors,
        radial_radius_quantile=args.mesh_radial_radius_quantile,
        radial_radius_padding=args.mesh_radial_radius_padding,
        radial_center_quantile=args.mesh_radial_center_quantile,
        radial_empty_angle_deg=args.mesh_radial_empty_angle_deg,
        radial_empty_radius_quantile=args.mesh_radial_empty_radius_quantile,
        radial_support_angle_deg=args.mesh_radial_support_angle_deg,
        radial_min_support_points=args.mesh_radial_min_support_points,
        radial_tighten_angle_deg=args.mesh_radial_tighten_angle_deg,
        radial_tighten_quantile=args.mesh_radial_tighten_quantile,
        radial_tighten_blend=args.mesh_radial_tighten_blend,
        radial_max_radius_quantile=args.mesh_radial_max_radius_quantile,
        radial_radius_smooth_iterations=args.mesh_radial_radius_smooth_iterations,
        radial_radius_smooth_weight=args.mesh_radial_radius_smooth_weight,
        normal_radius=args.mesh_normal_radius,
        normal_max_nn=args.mesh_normal_max_nn,
        poisson_depth=args.mesh_poisson_depth,
        density_quantile=args.mesh_density_quantile,
        smooth_iterations=args.mesh_smooth_iterations,
        simplify_voxel=args.mesh_simplify_voxel,
        keep_largest_component=not args.keep_mesh_components,
    )


def save_or_show_mesh(
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
    pcd_file: pathlib.Path,
    args: argparse.Namespace,
) -> None:
    if not args.not_save_mesh:
        o3d.io.write_triangle_mesh(str(pcd_file.parent / args.mesh_output_name), mesh)
    if args.show_mesh:
        visualize_mesh_with_pcd(
            mesh,
            pcd,
            mesh_opacity=args.mesh_visualization_opacity,
        )


def find_pcd_paths(root_dir: pathlib.Path, input_name: str) -> list[pathlib.Path]:
    if root_dir.is_file():
        return [root_dir]
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {root_dir}")
    return sorted(root_dir.rglob(input_name))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a stone mesh from a point cloud, decompose it into "
            "multiple convex parts with CoACD, and export those parts as DSF OBJ "
            "convex objects."
        )
    )
    parser.add_argument(
        "root_dir",
        help="A PCD file or root directory to search for merged point clouds.",
    )
    parser.add_argument(
        "--input_name",
        default=None,
        help="PCD filename to search for. Defaults to merged.pcd or merged_refined.pcd.",
    )
    parser.add_argument(
        "--get_refined",
        action="store_true",
        help="Search merged_refined.pcd instead of merged.pcd when --input_name is unset.",
    )
    parser.add_argument(
        "--output_name",
        default="dsf_multiple.obj",
        help="Output OBJ containing one DSF convex object per decomposition part.",
    )
    parser.add_argument(
        "--mesh_output_name",
        default="mesh_refined.ply",
        help="Optional reconstructed mesh output name.",
    )
    parser.add_argument(
        "--not_save_mesh",
        action="store_true",
        help="Skip writing the reconstructed mesh file.",
    )
    parser.add_argument(
        "--not_save_refined",
        action="store_true",
        help="Skip saving the centered refined point cloud.",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--show_result", action="store_true")
    parser.add_argument(
        "--show_mesh",
        action="store_true",
        help="Visualize the centered PCD with the reconstructed mesh before CoACD.",
    )
    parser.add_argument(
        "--show_convex",
        action="store_true",
        help="Visualize raw CoACD convex parts before fitting DSFs.",
    )
    parser.add_argument(
        "--show_split_pcd",
        action="store_true",
        help="Visualize PCD point sets before DSF fitting.",
    )
    parser.add_argument(
        "--dsf_fit_source",
        default="split_pcd",
        choices=["coacd_vertices", "split_pcd", "both"],
        help=(
            "Points used to fit each DSF: CoACD convex vertices, split PCD "
            "points, or both."
        ),
    )
    parser.add_argument(
        "--split_inside_tolerance",
        default=0.02,
        type=float,
        help="Signed-distance tolerance for assigning PCD points to CoACD convexes.",
    )
    parser.add_argument(
        "--split_min_points_per_part",
        default=20,
        type=int,
        help="Minimum points used to fit each DSF part via nearest-point fallback.",
    )
    parser.add_argument(
        "--manual_split_seeds",
        action="store_true",
        help=(
            "Manually pick seed points on the centered PCD and fit one DSF per "
            "nearest-seed PCD region. This bypasses CoACD for DSF fitting."
        ),
    )
    parser.add_argument(
        "--manual_split_seed_path",
        default=None,
        help=(
            "Optional .npy file for manual split seeds. Existing files are loaded; "
            "new picked seeds are saved."
        ),
    )
    parser.add_argument(
        "--manual_split_force_pick",
        action="store_true",
        help="Pick manual split seeds even when --manual_split_seed_path exists.",
    )
    parser.add_argument(
        "--manual_split_left_click",
        action="store_true",
        help="Pick manual split seeds with left click instead of pressing P.",
    )
    parser.add_argument(
        "--mesh_visualization_opacity",
        default=0.35,
        type=float,
        help="Opacity of the reconstructed mesh in --show_mesh visualization.",
    )
    parser.add_argument("--mu", default=1.0, type=float)
    parser.add_argument(
        "--dsf_num_vertices",
        default=None,
        type=int,
        help=(
            "Total DSF vertex budget. For multiple convex parts, this is split "
            "across fitted DSFs. Defaults to dsf_fit/config.yml."
        ),
    )

    add_filter_arguments(parser)
    add_mesh_arguments(parser)
    add_coacd_arguments(parser)
    args = parser.parse_args()

    input_name = args.input_name
    if input_name is None:
        input_name = "merged_refined.pcd" if args.get_refined else "merged.pcd"

    pcd_paths = find_pcd_paths(pathlib.Path(args.root_dir), input_name)
    if not pcd_paths:
        print(f"No {input_name} files found under {args.root_dir}")
        return

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dsf_cfg = OmegaConf.load(os.path.join(cur_dir, "dsf_fit/config.yml"))
    if args.dsf_num_vertices is not None:
        dsf_cfg.optimizer.num_vertices = args.dsf_num_vertices

    for pcd_file in pcd_paths:
        print(f"Processing: {pcd_file}")
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        if args.visualize or args.prompt_coacd_max_convex_hull:
            if not choose_reconstruct_pcd(pcd_file, pcd):
                print(f"Skipping {pcd_file}: user selected skip.")
                continue
        if not args.disable_filter:
            pcd = clean_point_cloud(pcd, args)
        if len(pcd.points) < 4:
            print(f"Skipping {pcd_file}: not enough points after filtering.")
            continue

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        if args.visualize:
            o3d.visualization.draw_geometries(
                [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
            )

        pcd_points = np.asarray(pcd.points).copy()
        mass, inertia, center = compute_hull_inertial_parameters(pcd_points)
        if not args.not_save_refined:
            pcd.translate(-center)
            print(f"Recentered PCD by {center.round(4)} (norm {np.linalg.norm(center):.4f} m)")
        mesh = None
        wrote_mesh = False
        default_coacd_max_convex_hull = args.coacd_max_convex_hull
        manual_seed_points = None

        while True:
            coacd_max_convex_hull = default_coacd_max_convex_hull
            if args.prompt_coacd_max_convex_hull:
                coacd_max_convex_hull = choose_coacd_max_convex_hull(
                    pcd_file,
                    pcd,
                    default_value=default_coacd_max_convex_hull,
                )
                default_coacd_max_convex_hull = coacd_max_convex_hull

            if args.manual_split_seeds:
                if args.show_mesh or not args.not_save_mesh:
                    if mesh is None:
                        mesh = reconstruct_mesh_for_decomposition(pcd, args)
                    if not wrote_mesh:
                        save_or_show_mesh(mesh, pcd, pcd_file, args)
                        wrote_mesh = True

                if manual_seed_points is None:
                    seed_path = (
                        pathlib.Path(args.manual_split_seed_path)
                        if args.manual_split_seed_path is not None
                        else None
                    )
                    manual_seed_points = get_manual_split_seeds(
                        pcd_file,
                        pcd,
                        seed_path=seed_path,
                        left_clicking=args.manual_split_left_click,
                        force_pick=args.manual_split_force_pick,
                    )
                if len(manual_seed_points) == 0:
                    print(f"Skipping {pcd_file}: no manual split seeds selected.")
                    break

                fit_point_sets = split_points_by_seed_points(
                    np.asarray(pcd.points),
                    manual_seed_points,
                    min_points_per_part=args.split_min_points_per_part,
                )
                if args.show_split_pcd:
                    visualize_pcd_splits(fit_point_sets)
                vertex_counts = distribute_dsf_vertex_budget(
                    int(dsf_cfg.optimizer.num_vertices),
                    len(fit_point_sets),
                )
                print("Fitting DSFs from manual PCD seed splits.")
                print(
                    "Distributed DSF vertex budget "
                    f"{int(dsf_cfg.optimizer.num_vertices)} across "
                    f"{len(fit_point_sets)} parts: {vertex_counts}"
                )
                dsf_parts = fit_dsf_parts(dsf_cfg, fit_point_sets, vertex_counts)
                convex_count = len(fit_point_sets)

            elif coacd_max_convex_hull <= 1:
                print(
                    "Selected coacd_max_convex_hull=1; skipping CoACD and fitting "
                    "a single DSF to the full point cloud."
                )
                if args.show_mesh or not args.not_save_mesh:
                    if mesh is None:
                        mesh = reconstruct_mesh_for_decomposition(pcd, args)
                    if not wrote_mesh:
                        save_or_show_mesh(mesh, pcd, pcd_file, args)
                        wrote_mesh = True

                convex_count = 1
                dsf_parts = fit_dsf_parts(dsf_cfg, [np.asarray(pcd.points)])
            else:
                if mesh is None:
                    mesh = reconstruct_mesh_for_decomposition(pcd, args)
                if not wrote_mesh:
                    save_or_show_mesh(mesh, pcd, pcd_file, args)
                    wrote_mesh = True

                convex_parts = decompose_mesh_coacd(
                    mesh,
                    threshold=args.coacd_threshold,
                    max_convex_hull=coacd_max_convex_hull,
                    preprocess_mode=args.coacd_preprocess_mode,
                    preprocess_resolution=args.coacd_preprocess_resolution,
                    resolution=args.coacd_resolution,
                    mcts_nodes=args.coacd_mcts_nodes,
                    mcts_iterations=args.coacd_mcts_iterations,
                    mcts_max_depth=args.coacd_mcts_max_depth,
                    pca=args.coacd_pca,
                    merge=not args.coacd_no_merge,
                    decimate=args.coacd_decimate,
                    seed=args.coacd_seed,
                    real_metric=args.coacd_real_metric,
                )
                if not convex_parts:
                    print(f"Skipping {pcd_file}: CoACD produced no convex parts.")
                    break
                if args.show_convex:
                    visualize_convex_decomposition(pcd, convex_parts)
                fit_point_sets, split_point_sets = get_dsf_fit_point_sets(
                    np.asarray(pcd.points),
                    convex_parts,
                    fit_source=args.dsf_fit_source,
                    inside_tolerance=args.split_inside_tolerance,
                    min_points_per_part=args.split_min_points_per_part,
                )
                if args.show_split_pcd and split_point_sets is not None:
                    visualize_pcd_splits(split_point_sets)
                elif args.show_split_pcd:
                    print(
                        "--show_split_pcd skipped because "
                        "--dsf_fit_source=coacd_vertices does not split PCD points."
                    )
                vertex_counts = distribute_dsf_vertex_budget(
                    int(dsf_cfg.optimizer.num_vertices),
                    len(fit_point_sets),
                )
                print(f"Fitting DSFs from {args.dsf_fit_source}.")
                print(
                    "Distributed DSF vertex budget "
                    f"{int(dsf_cfg.optimizer.num_vertices)} across "
                    f"{len(fit_point_sets)} parts: {vertex_counts}"
                )
                dsf_parts = fit_dsf_parts(dsf_cfg, fit_point_sets, vertex_counts)
                convex_count = len(convex_parts)

            if args.show_result:
                accepted = visualize_dsf_decomposition(
                    pcd,
                    dsf_parts,
                    allow_reselect=(
                        args.prompt_coacd_max_convex_hull
                        and not args.manual_split_seeds
                    ),
                )
                if not accepted:
                    print("Re-selecting CoACD max convex hull after result review.")
                    continue

            export_dsf_parts(
                pcd_file.parent / args.output_name,
                dsf_parts,
                mass=mass,
                inertia=inertia,
                mu=args.mu,
            )

            if not args.not_save_refined:
                o3d.io.write_point_cloud(
                    str(pcd_file.parent / "merged_refined.pcd"), pcd
                )

            print(
                f"Wrote {convex_count} convex DSF objects to "
                f"{pcd_file.parent / args.output_name}"
            )
            break


if __name__ == "__main__":
    if sys.gettrace():
        sys.argv = [__file__, "data/stone_pcd/pcd_251024/stone01"]
    main()
