import argparse
import copy
import pathlib
import re
import sys

import numpy as np
import open3d as o3d

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from perception.reconstruction_dsf import manually_remove_points
from perception.utils.refine_pcd import multiscale_icp


def _paint_copy(pcd, color):
    painted = copy.deepcopy(pcd)
    painted.paint_uniform_color(color)
    return painted


def _depth_color_copy(pcd, low_color=None, high_color=None):
    colored = copy.deepcopy(pcd)
    points = np.asarray(colored.points)
    if len(points) == 0:
        return colored

    if low_color is None:
        low_color = np.array([0.1, 0.35, 1.0])
    else:
        low_color = np.asarray(low_color)
    if high_color is None:
        high_color = np.array([1.0, 0.45, 0.05])
    else:
        high_color = np.asarray(high_color)

    z = points[:, 2]
    z_min = z.min()
    z_range = z.max() - z_min
    if z_range < 1e-8:
        colored.paint_uniform_color([0.8, 0.8, 0.8])
        return colored

    t = (z - z_min) / z_range
    colors = low_color + t[:, None] * (high_color - low_color)
    colored.colors = o3d.utility.Vector3dVector(colors)
    return colored


def _visualization_copy(pcd, color, normal_radius):
    visualized = _paint_copy(pcd, color)
    if normal_radius > 0:
        visualized.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
    return visualized


def _picking_copy(pcd, normal_radius):
    picked = _depth_color_copy(pcd)
    if normal_radius > 0:
        picked.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
    return picked


def _move_view_copy(pcd, low_color, high_color, normal_radius):
    viewed = _depth_color_copy(pcd, low_color, high_color)
    if normal_radius > 0:
        viewed.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
    return viewed


def _estimate_normals(pcd, normal_radius):
    if normal_radius > 0 and len(pcd.points) > 0:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )
    return pcd


def _load_pcd(path):
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"PCD is empty or unreadable: {path}")
    return pcd


def _pick_points(pcd, title, show_normals):
    print("")
    print(title)
    print("  Shift + left click: pick a point")
    print("  Shift + right click: undo last pick")
    print("  Q or close window: finish picking")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.point_show_normal = show_normals
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def _manual_initial_transform(source, target, args):
    source_view = _picking_copy(source, args.normal_radius)
    target_view = _picking_copy(target, args.normal_radius)

    print("")
    print("[initial] Pick matching points in the same order.")
    print("[initial] First window is moving source, second window is fixed target.")
    source_ids = _pick_points(source_view, "Pick source points", args.show_normals)
    target_ids = _pick_points(
        target_view, "Pick matching target points", args.show_normals
    )

    if len(source_ids) != len(target_ids):
        raise ValueError(
            f"Picked point counts differ: source={len(source_ids)}, "
            f"target={len(target_ids)}"
        )
    if len(source_ids) < 3:
        raise ValueError("Pick at least 3 matching points for initial alignment.")

    source_picks = source.select_by_index(source_ids)
    target_picks = target.select_by_index(target_ids)
    correspondences = np.asarray([[i, i] for i in range(len(source_ids))])
    return o3d.pipelines.registration.TransformationEstimationPointToPoint(
        False
    ).compute_transformation(
        source_picks,
        target_picks,
        o3d.utility.Vector2iVector(correspondences),
    )


def _rotation_transform(axis, angle, center):
    transform = np.eye(4)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.asarray(axis) * angle
    )

    to_origin = np.eye(4)
    to_origin[:3, 3] = -center
    back = np.eye(4)
    back[:3, 3] = center
    return back @ transform @ to_origin


def _translation_transform(delta):
    transform = np.eye(4)
    transform[:3, 3] = delta
    return transform


def _gui_initial_transform(source, target, args):
    fixed = _move_view_copy(
        target,
        low_color=[0.05, 0.25, 0.95],
        high_color=[0.35, 0.95, 1.0],
        normal_radius=args.normal_radius,
    )
    moving = _move_view_copy(
        source,
        low_color=[1.0, 0.5, 0.0],
        high_color=[1.0, 0.05, 0.05],
        normal_radius=args.normal_radius,
    )
    source_center = source.get_center()
    target_center = target.get_center()
    initial_centering = _translation_transform(target_center - source_center)
    moving.transform(initial_centering)

    state = {
        "transform": initial_centering.copy(),
        "move_step": 0.02,
        "rotation_step": np.deg2rad(5.0),
    }

    print("")
    print("[initial] Move the orange/red source cloud onto the blue/cyan target.")
    print("[initial] Close the window when the pose is good enough for ICP.")
    print("  Translate: A/D=x, S/W=y, F/R=z")
    print("  Rotate:    J/L=yaw(z), I/K=pitch(y), U/O=roll(x)")
    print("  Step:      [ smaller, ] larger")
    print("  Reset:     X")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Move source to target", width=1280, height=720)
    vis.add_geometry(fixed)
    vis.add_geometry(moving)
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.point_show_normal = args.show_normals

    def apply_increment(transform):
        moving.transform(transform)
        state["transform"] = transform @ state["transform"]
        vis.update_geometry(moving)
        vis.update_renderer()
        return False

    def rotate(axis, sign):
        def callback(_vis):
            center = moving.get_center()
            transform = _rotation_transform(axis, sign * state["rotation_step"], center)
            return apply_increment(transform)

        return callback

    def smaller_step(_vis):
        state["move_step"] *= 0.5
        state["rotation_step"] *= 0.5
        print(
            f"[initial] step: move={state['move_step']:.4f}m, "
            f"rotate={np.rad2deg(state['rotation_step']):.2f}deg"
        )
        return False

    def larger_step(_vis):
        state["move_step"] *= 2.0
        state["rotation_step"] *= 2.0
        print(
            f"[initial] step: move={state['move_step']:.4f}m, "
            f"rotate={np.rad2deg(state['rotation_step']):.2f}deg"
        )
        return False

    def reset(_vis):
        inverse = np.linalg.inv(state["transform"])
        moving.transform(inverse)
        moving.transform(initial_centering)
        state["transform"] = initial_centering.copy()
        vis.update_geometry(moving)
        vis.update_renderer()
        return False

    def move(axis, sign):
        def callback(_vis):
            delta = np.zeros(3)
            delta[axis] = sign * state["move_step"]
            return apply_increment(_translation_transform(delta))

        return callback

    key_callbacks = {
        "A": move(0, -1.0),
        "D": move(0, 1.0),
        "S": move(1, -1.0),
        "W": move(1, 1.0),
        "F": move(2, -1.0),
        "R": move(2, 1.0),
        "J": rotate([0, 0, 1], 1.0),
        "L": rotate([0, 0, 1], -1.0),
        "I": rotate([0, 1, 0], 1.0),
        "K": rotate([0, 1, 0], -1.0),
        "U": rotate([1, 0, 0], 1.0),
        "O": rotate([1, 0, 0], -1.0),
    }
    for key, callback in key_callbacks.items():
        vis.register_key_callback(ord(key), callback)
        vis.register_key_callback(ord(key.lower()), callback)
    vis.register_key_callback(ord("["), smaller_step)
    vis.register_key_callback(ord("]"), larger_step)
    vis.register_key_callback(ord("X"), reset)
    vis.register_key_callback(ord("x"), reset)
    vis.run()
    vis.destroy_window()

    return state["transform"]


def _find_stone_pairs(root_dir):
    pairs = {}
    pattern = re.compile(r"(.+)_([0-9]+)$")

    for scan_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        match = pattern.match(scan_dir.name)
        if match is None:
            continue

        pcd_path = scan_dir / "merged.pcd"
        if not pcd_path.is_file():
            continue

        stone_name, group_id = match.groups()
        pairs.setdefault(stone_name, []).append((int(group_id), pcd_path))

    return {
        stone_name: sorted(paths)
        for stone_name, paths in pairs.items()
        if len(paths) >= 2
    }


def _stone_matches(stone_name, requested):
    if requested is None:
        return True
    if stone_name == requested:
        return True
    return stone_name == f"stone{requested}"


def _merge_pair(target_path, source_path, args):
    print(f"[load] target: {target_path}")
    print(f"[load] source: {source_path}")
    target = _load_pcd(target_path)
    source = _load_pcd(source_path)

    if args.voxel > 0:
        target_for_icp = target.voxel_down_sample(args.voxel)
        source_for_icp = source.voxel_down_sample(args.voxel)
    else:
        target_for_icp = target
        source_for_icp = source

    initial_transform = (
        np.eye(4)
        if args.identity_initial_pose
        else (
            _manual_initial_transform(source_for_icp, target_for_icp, args)
            if args.pick_initial_points
            else _gui_initial_transform(source_for_icp, target_for_icp, args)
        )
    )

    aligned_initial = copy.deepcopy(source)
    aligned_initial.transform(initial_transform)
    if args.show_initial:
        o3d.visualization.draw_geometries(
            [
                _visualization_copy(target, [0.2, 0.45, 1.0], args.normal_radius),
                _visualization_copy(
                    aligned_initial, [1.0, 0.55, 0.05], args.normal_radius
                ),
            ],
            window_name="Initial alignment",
            point_show_normal=args.show_normals,
        )

    transform, history = multiscale_icp(
        source,
        target,
        init_trans=initial_transform,
        voxel_sizes=[0.08, 0.04, 0.02, 0.01],
        max_iters=[80, 50, 30, 20],
        method="point_to_plane",
    )
    _, fitness, rmse, n_corr = history[-1]
    print(f"[icp] fitness={fitness:.3f}, rmse={rmse:.4f}, corr={n_corr}")

    aligned_source = copy.deepcopy(source)
    aligned_source.transform(transform)

    merged = copy.deepcopy(target)
    merged += aligned_source
    if args.output_voxel > 0:
        merged = merged.voxel_down_sample(args.output_voxel)

    merged = _estimate_normals(merged, args.normal_radius)

    if args.manual_remove:
        merged = manually_remove_points(
            merged,
            distance=args.manual_remove_distance,
            max_remove_ratio=args.max_remove_ratio,
            left_clicking=args.left_click,
        )
        merged = _estimate_normals(merged, args.normal_radius)

    if not args.no_show_result:
        o3d.visualization.draw_geometries(
            [
                _visualization_copy(target, [0.2, 0.45, 1.0], args.normal_radius),
                _visualization_copy(
                    aligned_source, [1.0, 0.55, 0.05], args.normal_radius
                ),
            ],
            window_name="Aligned groups",
            point_show_normal=args.show_normals,
        )
        merged_view = _visualization_copy(merged, [0.8, 0.8, 0.8], args.normal_radius)
        o3d.visualization.draw_geometries(
            [merged_view],
            window_name="Merged PCD",
            point_show_normal=args.show_normals,
        )

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Interactively ICP-merge two merged.pcd groups for each stone."
    )
    parser.add_argument(
        "root_dir", help="Root directory containing stoneXX_1/stoneXX_2."
    )
    parser.add_argument(
        "--stone",
        default=None,
        help="Only process one stone name, for example stone34.",
    )
    parser.add_argument(
        "--voxel",
        default=0.01,
        type=float,
        help="Voxel size for picking/ICP preview clouds.",
    )
    parser.add_argument(
        "--output_voxel",
        default=0.005,
        type=float,
        help="Voxel size for saved merged cloud. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--identity_initial_pose",
        action="store_true",
        help="Skip manual initialization and start ICP from identity.",
    )
    parser.add_argument(
        "--pick_initial_points",
        action="store_true",
        help="Use point-pair picking instead of keyboard source-cloud movement.",
    )
    parser.add_argument(
        "--manual_remove",
        action="store_true",
        help="Open the manual point-removal picker before saving.",
    )
    parser.add_argument(
        "--manual_remove_distance",
        default=0.10,
        type=float,
        help="Radius removed around each manually picked point.",
    )
    parser.add_argument(
        "--max_remove_ratio",
        default=0.5,
        type=float,
        help="Skip manual deletion if selected regions exceed this cloud fraction.",
    )
    parser.add_argument(
        "--left_click",
        action="store_true",
        help="Pick manual removal centers with left click instead of pressing P.",
    )
    parser.add_argument(
        "--show_initial",
        action="store_true",
        help="Show the manually initialized alignment before ICP.",
    )
    parser.add_argument(
        "--no_show_result",
        action="store_true",
        help="Do not show aligned groups and final merged cloud.",
    )
    parser.add_argument(
        "--show_result",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--show_normals",
        action="store_true",
        help="Show estimated point normals in visualization windows.",
    )
    parser.add_argument(
        "--normal_radius",
        default=0.05,
        type=float,
        help="Search radius for visualization normal estimation.",
    )
    args = parser.parse_args()

    root_dir = pathlib.Path(args.root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    pairs = _find_stone_pairs(root_dir)
    if args.stone is not None:
        pairs = {
            name: paths
            for name, paths in pairs.items()
            if _stone_matches(name, args.stone)
        }
    if not pairs:
        raise ValueError(
            f"No stone pairs with two merged.pcd files found in {root_dir}"
        )

    for stone_name, paths in pairs.items():
        if len(paths) > 2:
            print(
                f"[warning] {stone_name}: found {len(paths)} groups; using first two."
            )

        (_, target_path), (_, source_path) = paths[:2]
        print("")
        print(f"Processing {stone_name}")
        merged = _merge_pair(target_path, source_path, args)

        output_dir = root_dir / f"{stone_name}_merged"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "merged.pcd"
        o3d.io.write_point_cloud(str(output_path), merged)
        print(f"[write] {output_path} ({len(merged.points)} points)")


if __name__ == "__main__":
    main()
