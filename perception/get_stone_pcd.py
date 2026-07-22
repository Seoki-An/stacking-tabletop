import os
import sys
import argparse
import pathlib
from tqdm import tqdm
import copy
import random
import time

import pandas as pd
import numpy as np
import open3d as o3d

from model import get_excavator_model

from perception.utils.refine_pcd import (
    multiscale_icp,
    refine_pose_graph,
    remove_points_from_points,
    select_points_near_points,
    box_crop_largest_cluster,
)
from perception.merge_filtered_groups import (
    build_incremental_groups,
    merge_groups,
    save_groups,
    save_skipped,
)


def _rotation_angle_deg(rotation):
    cos_theta = (np.trace(rotation) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _paint_copy(pcd, color):
    painted = copy.deepcopy(pcd)
    painted.paint_uniform_color(color)
    return painted


def _crop_nonnegative_y(pcd):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd
    keep_indices = np.where(points[:, 1] >= 0.0)[0]
    return pcd.select_by_index(keep_indices)


def _draw_geometries_with_timeout(geometries, window_name, timeout_sec):
    if timeout_sec <= 0:
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    for geometry in geometries:
        vis.add_geometry(geometry)

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(0.01)
    vis.destroy_window()


def review_filtered_frames(filtered_frames, preview_voxel=0.01):
    decisions = [None] * len(filtered_frames)
    index = 0

    print(
        "[review] Manual merge review enabled. "
        "Press Y to accept, N to reject, B/U to go back, "
        "or close the window to accept."
    )
    while index < len(filtered_frames):
        frame_id, pcd = filtered_frames[index]
        action = {"value": "accept"}

        def accept(vis):
            action["value"] = "accept"
            vis.close()
            return False

        def reject(vis):
            action["value"] = "reject"
            vis.close()
            return False

        def go_back(vis):
            action["value"] = "back"
            vis.close()
            return False

        accepted_preview = o3d.geometry.PointCloud()
        for previous_decision, (_, previous_pcd) in zip(
            decisions[:index],
            filtered_frames[:index],
        ):
            if previous_decision is True:
                accepted_preview += previous_pcd

        merged_view = accepted_preview
        candidate_view = pcd
        if preview_voxel > 0:
            if len(merged_view.points) > 0:
                merged_view = merged_view.voxel_down_sample(preview_voxel)
            candidate_view = candidate_view.voxel_down_sample(preview_voxel)

        geometries = []
        if len(merged_view.points) > 0:
            geometries.append(_paint_copy(merged_view, [0.55, 0.55, 0.55]))
        geometries.append(_paint_copy(candidate_view, [1.0, 0.65, 0.05]))
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        )

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            window_name=(
                f"Review {index + 1}/{len(filtered_frames)} "
                f"filtered_{frame_id}.pcd: Y accept / N reject / B back"
            ),
            width=1280,
            height=720,
        )
        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.register_key_callback(ord("Y"), accept)
        vis.register_key_callback(ord("y"), accept)
        vis.register_key_callback(ord("N"), reject)
        vis.register_key_callback(ord("n"), reject)
        vis.register_key_callback(ord("B"), go_back)
        vis.register_key_callback(ord("b"), go_back)
        vis.register_key_callback(ord("U"), go_back)
        vis.register_key_callback(ord("u"), go_back)
        vis.run()
        vis.destroy_window()

        if action["value"] == "back":
            if index == 0:
                print("[review] already at first frame")
            else:
                decisions[index] = None
                index -= 1
                decisions[index] = None
                print(f"[review] rollback to frame={filtered_frames[index][0]}")
            continue

        if action["value"] == "reject":
            decisions[index] = False
            print(f"[review] frame={frame_id}: reject")
        else:
            decisions[index] = True
            print(f"[review] frame={frame_id}: accept")
        index += 1

    accepted = [
        frame
        for decision, frame in zip(decisions, filtered_frames)
        if decision is True
    ]
    rejected = [
        frame
        for decision, frame in zip(decisions, filtered_frames)
        if decision is False
    ]

    print(
        f"[review] accepted={len(accepted)}, rejected={len(rejected)}, "
        f"total={len(filtered_frames)}"
    )
    return accepted, rejected


def get_stone_pcd():
    parser = argparse.ArgumentParser(
        description="Merge .pcd files in the assigned root directory and its subdirectories."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory to start search (ex: STONE_PCD)",
        default=".data/stone1",
    )
    parser.add_argument(
        "--d_thresh",
        help="Distance threshold for removing gripper points from target point cloud",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--sample_ratio",
        help="Ratio of pcd files to sample for processing",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Whether to visualize the result"
    )
    parser.add_argument(
        "--show_result",
        action="store_true",
        help="Whether to visualize the final merged point cloud",
    )
    parser.add_argument(
        "--show_result_timeout",
        default=5.0,
        type=float,
        help=(
            "Seconds to keep each --show_result window open. "
            "Set <= 0 for blocking Open3D windows."
        ),
    )
    parser.add_argument(
        "--no_pose_graph",
        action="store_true",
        help="Skip the pose-graph refinement step before merging.",
    )
    parser.add_argument(
        "--pose_graph_verbose",
        action="store_true",
        help="Print accepted pose-graph edges and connectivity diagnostics.",
    )
    parser.add_argument(
        "--icp_target_radius",
        default=0.6,
        type=float,
        help="Radius for selecting lidar points near the predicted gripper before ICP.",
    )
    parser.add_argument(
        "--icp_refine_radius",
        default=0.15,
        type=float,
        help="Radius for selecting lidar points near the ICP-corrected gripper.",
    )
    parser.add_argument(
        "--max_icp_translation",
        default=0.8,
        type=float,
        help="Reject frames whose gripper ICP correction translates farther than this.",
    )
    parser.add_argument(
        "--max_icp_rotation_deg",
        default=25.0,
        type=float,
        help="Reject frames whose gripper ICP correction rotates farther than this.",
    )
    parser.add_argument(
        "--registration_verbose",
        action="store_true",
        help="Print rejected gripper-to-lidar ICP frames and correction magnitudes.",
    )
    parser.add_argument(
        "--registration_threshold",
        default=0.4,
        type=float,
        help="Minimum ICP fitness to accept a frame for merging.",
    )
    parser.add_argument(
        "--direct_merge",
        dest="direct_merge",
        action="store_true",
        help="Directly merge filtered frames. This is the default.",
    )
    parser.add_argument(
        "--group_merge",
        dest="direct_merge",
        action="store_false",
        help="Use incremental ICP grouping before final merge.",
    )
    parser.add_argument(
        "--review_merge",
        action="store_true",
        help="Manually accept/reject filtered frames before merging. Default is disabled.",
    )
    parser.add_argument(
        "--review_preview_voxel",
        default=0.01,
        type=float,
        help="Voxel size for manual-review preview clouds. Set <= 0 to disable downsampling.",
    )
    parser.add_argument(
        "--merge_group_voxel",
        default=0.01,
        type=float,
        help="Voxel size used when updating incremental ICP groups.",
    )
    parser.add_argument(
        "--merge_final_voxel",
        default=0.005,
        type=float,
        help="Voxel size for final grouped merged cloud.",
    )
    parser.add_argument(
        "--merge_frame_fitness_threshold",
        default=0.30,
        type=float,
        help="Minimum ICP fitness for adding one filtered frame to a group.",
    )
    parser.add_argument(
        "--merge_frame_min_correspondences",
        default=50,
        type=int,
        help="Minimum ICP correspondences for adding one filtered frame to a group.",
    )
    parser.add_argument(
        "--merge_frame_max_translation",
        default=0.30,
        type=float,
        help="Maximum frame-to-group ICP translation correction.",
    )
    parser.add_argument(
        "--merge_frame_max_rotation_deg",
        default=25.0,
        type=float,
        help="Maximum frame-to-group ICP rotation correction.",
    )
    parser.add_argument(
        "--merge_new_group_min_fitness",
        default=0.05,
        type=float,
        help="Minimum best-match ICP fitness required to start a new group.",
    )
    parser.add_argument(
        "--merge_new_group_min_correspondences",
        default=30,
        type=int,
        help="Minimum best-match correspondences required to start a new group.",
    )
    parser.add_argument(
        "--merge_group_fitness_threshold",
        default=0.25,
        type=float,
        help="Minimum ICP fitness for merging one discovered group into final cloud.",
    )
    parser.add_argument(
        "--merge_group_min_correspondences",
        default=100,
        type=int,
        help="Minimum ICP correspondences for merging one discovered group into final cloud.",
    )
    parser.add_argument(
        "--merge_group_max_translation",
        default=0.50,
        type=float,
        help="Maximum group-to-final ICP translation correction.",
    )
    parser.add_argument(
        "--merge_group_max_rotation_deg",
        default=35.0,
        type=float,
        help="Maximum group-to-final ICP rotation correction.",
    )
    parser.add_argument(
        "--merge_groups_output_dir",
        default="icp_groups",
        help="Directory under each scan directory for discovered ICP groups.",
    )
    parser.add_argument(
        "--merge_verbose",
        action="store_true",
        help="Print incremental merge ICP decisions.",
    )
    parser.set_defaults(direct_merge=True)

    args = parser.parse_args()
    visualize = args.visualize
    show_result = args.show_result
    distance_threshold = args.d_thresh
    sample_ratio = args.sample_ratio
    use_pose_graph = not args.no_pose_graph

    root_dir = pathlib.Path(args.root_dir)
    if not root_dir.is_dir():
        print(f"Error: '{args.root_dir}' is not available.")
        return

    lidar_name = "lidar2_link"

    # Setting configuration of the gripper
    model, link_meshes = get_excavator_model()

    csv_joint_map = {
        "joint_0": "cs_cor",
        "joint_1": "cs_boom",
        "joint_2": "cs_arm",
        "joint_3": "cs_bucket",
        "joint_4": "cs_tilt",
        "joint_5": "cs_rotate",
    }

    # Setting Rays
    num_az = 400
    num_el = 200
    num_rays = num_az * num_el

    phi, theta = np.meshgrid(
        np.linspace(-np.pi / 4, np.pi / 4, num_az),
        np.linspace(np.pi / 3, 2 * np.pi / 3, num_el),
    )
    phi = phi.reshape(-1)
    theta = theta.reshape(-1)
    rays = np.zeros((num_rays, 6), dtype=np.float32)
    rays[:, 3:] = np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )

    origin_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )

    # Cropping intervals
    width = np.array([-2.0, 2.0])
    length = np.array([-1.0, 2.0])
    height = np.array([-2.0, 2.0])

    # Load the polynomial coefficients for opening angle estimation
    coeff = np.load("assets/opening_angle_fit.npy")
    poly = np.poly1d(coeff)

    csv_paths = sorted(root_dir.rglob("*.csv"))
    total_pcd_groups = len(csv_paths)
    merged_pcd_groups = 0
    skipped_pcd_groups = 0
    print(f"Found {total_pcd_groups} PCD groups to merge.")

    for group_idx, csv_path in enumerate(csv_paths, start=1):
        rearranged_pcd_list = []
        gripper_pcd_list = []

        csv_dir = csv_path.parent
        print(f"Processing [{group_idx}/{total_pcd_groups}]: {csv_dir}")

        # Build {timestamp -> .pcd path} once per CSV, scoped to its directory.
        pcd_map = {}
        for p in csv_dir.rglob("*.pcd"):
            try:
                pcd_map[int(p.stem)] = p
            except ValueError:
                continue

        csv_file = pd.read_csv(csv_path, header=0, skipinitialspace=True)
        joint_angles = {}
        for key in csv_joint_map.keys():
            joint_angles[key] = []

        pcd_times = []

        min_idx = 0
        min_rotate = float("inf")
        for i, pcd_time in enumerate(csv_file["time"]):
            if pcd_time in pcd_map and random.random() < sample_ratio:
                for key in joint_angles.keys():
                    joint_angles[key].append(csv_file[key][i])
                pcd_times.append(pcd_time)
                if abs(csv_file["joint_5"][i]) < min_rotate:
                    min_rotate = abs(csv_file["joint_5"][i])
                    min_idx = len(pcd_times) - 1

        if not pcd_times:
            print(f"[WARNING] no matching PCD timestamps found for {csv_path}")
            skipped_pcd_groups += 1
            continue

        q = [joint_angles[key][min_idx] for key in csv_joint_map.keys()]
        q = np.concatenate([q, np.zeros(2)])
        model.SetState(q)
        end_frame = np.eye(4)
        end_frame[:3, :3] = model.GetLink("grip_body").geoms[0].GetRotation()
        end_frame[:3, -1] = model.GetLink("grip_body").geoms[0].GetPosition()

        lidar_frame = np.eye(4)
        lidar_frame[:3, :3] = model.GetLink(lidar_name).GetRotation()
        lidar_frame[:3, -1] = model.GetLink(lidar_name).GetPosition()

        rel_frame = (
            np.linalg.inv(lidar_frame) @ end_frame
        )  # ee frame seen from lidar frame

        pcd_file = pcd_map[pcd_times[min_idx]]
        target_pcd = o3d.io.read_point_cloud(pcd_file)
        target_pcd = box_crop_largest_cluster(
            target_pcd, [0, 6], [-2, 2], [-2, 2], voxel=0.0, cluster=False
        )
        target_pcd = target_pcd.transform(np.linalg.inv(rel_frame))
        target_pcd = box_crop_largest_cluster(
            target_pcd, width, length, height, voxel=0.0, cluster=False
        )
        target_pcd_np = np.asarray(target_pcd.points)
        x_extent = target_pcd_np[:, 0].max() - target_pcd_np[:, 0].min()
        opening_angle = poly(x_extent)

        for i, pcd_time in tqdm(enumerate(pcd_times), total=len(pcd_times)):
            pcd_file = pcd_map[pcd_time]
            q = [joint_angles[key][i] for key in csv_joint_map.keys()]
            q = np.concatenate([q, opening_angle * np.ones(2)])

            ##############################################
            ############### Filter raw PCD ###############
            ##############################################

            model.SetState(q)
            end_frame = np.eye(4)
            end_frame[:3, :3] = model.GetLink("grip_body").geoms[0].GetRotation()
            end_frame[:3, -1] = model.GetLink("grip_body").geoms[0].GetPosition()

            lidar_frame = np.eye(4)
            lidar_frame[:3, :3] = model.GetLink(lidar_name).GetRotation()
            lidar_frame[:3, -1] = model.GetLink(lidar_name).GetPosition()

            rel_frame = (
                np.linalg.inv(lidar_frame) @ end_frame
            )  # ee frame seen from lidar frame

            target_pcd = o3d.io.read_point_cloud(pcd_file)

            # Align pcd to end-effector frame
            target_pcd = box_crop_largest_cluster(
                target_pcd, [0, 6], [-2, 2], [-2, 2], voxel=0.0, cluster=False
            )
            target_pcd = target_pcd.transform(np.linalg.inv(rel_frame))
            target_pcd = box_crop_largest_cluster(
                target_pcd, width, length, height, voxel=0.0, cluster=False
            )
            if len(target_pcd.points) < 50:
                continue

            # Align back to lidar frame
            target_pcd = target_pcd.transform(rel_frame)

            ##############################################
            #### Sample gripper pcd using ray casting ####
            ##############################################

            o3d_device = o3d.core.Device("CPU:0")
            scene = o3d.t.geometry.RaycastingScene()
            transformed_meshes = []
            gripper_pcd_full = o3d.geometry.PointCloud()

            for name, mesh in link_meshes.items():
                mesh = copy.deepcopy(mesh)
                link_frame = np.eye(4)
                link_frame[:3, :3] = model.GetLink(name).geoms[0].GetRotation()
                link_frame[:3, -1] = model.GetLink(name).geoms[0].GetPosition()
                mesh = mesh.transform(link_frame)
                transformed_meshes.append(mesh)
                if name in [
                    "cs_tilt",
                    "cs_rotate",
                    "grip_body",
                    "grip_left",
                    "grip_right",
                ]:
                    gripper_pcd_full += mesh.sample_points_uniformly(
                        number_of_points=2000
                    )
                    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                    mesh_tensor = mesh_tensor.to(o3d_device)
                    scene.add_triangles(mesh_tensor)

            rays_t = rays.copy()
            rays_t[:, :3] = lidar_frame[:3, -1]
            rays_t[:, 3:] = np.einsum("ij,nj -> ni", lidar_frame[:3, :3], rays_t[:, 3:])
            rays_tensor = o3d.core.Tensor(rays_t)
            raycast_result = scene.cast_rays(rays_tensor)
            t_hit = raycast_result["t_hit"].numpy()
            normals = raycast_result["primitive_normals"].numpy()
            mask = np.isfinite(t_hit)
            t_hit = t_hit[mask]
            normals = normals[mask]
            rays_valid = rays_t[mask]

            gripper_points = rays_valid[:, :3] + t_hit[:, None] * rays_valid[:, 3:]
            gripper_pcd = o3d.geometry.PointCloud()
            gripper_pcd.points = o3d.utility.Vector3dVector(gripper_points)
            gripper_pcd.transform(np.linalg.inv(lidar_frame))
            gripper_pcd_full.transform(np.linalg.inv(lidar_frame))

            ##############################################

            target_gripper_pcd = select_points_near_points(
                gripper_pcd, target_pcd, args.icp_target_radius
            )
            if len(target_gripper_pcd.points) < 100:
                if args.registration_verbose:
                    print(
                        f"[registration] skip frame={i}, time={pcd_time}: "
                        f"near_gripper_points={len(target_gripper_pcd.points)}"
                    )
                continue

            icp_transformation, history = multiscale_icp(
                gripper_pcd,
                target_gripper_pcd,
                init_trans=np.eye(4),
                voxel_sizes=[0.2, 0.1, 0.05, 0.02],
                max_iters=[100, 50, 30, 14],
            )
            _, fitness, inlier_rmse, n_correspondence = history[-1]

            if fitness < args.registration_threshold or n_correspondence < 50:
                if args.registration_verbose:
                    print(
                        f"[registration] skip frame={i}, time={pcd_time}: "
                        f"fitness={fitness:.3f}, corr={n_correspondence}, "
                        f"near_gripper_points={len(target_gripper_pcd.points)}"
                    )
                continue
            else:
                gripper_pcd_coarse = copy.deepcopy(gripper_pcd)
                gripper_pcd_coarse.transform(icp_transformation)
                target_gripper_pcd = select_points_near_points(
                    gripper_pcd_coarse, target_pcd, args.icp_refine_radius
                )
                if len(target_gripper_pcd.points) < 50:
                    if args.registration_verbose:
                        print(
                            f"[registration] skip frame={i}, time={pcd_time}: "
                            f"refine_near_gripper_points={len(target_gripper_pcd.points)}"
                        )
                    continue

                icp_transformation, history = multiscale_icp(
                    gripper_pcd,
                    target_gripper_pcd,
                    init_trans=icp_transformation,
                    voxel_sizes=[0.02, 0.01],
                    max_iters=[30, 14],
                )
                _, fitness, inlier_rmse, n_correspondence = history[-1]

            translation_correction = np.linalg.norm(icp_transformation[:3, 3])
            rotation_correction = _rotation_angle_deg(icp_transformation[:3, :3])
            if (
                translation_correction > args.max_icp_translation
                or rotation_correction > args.max_icp_rotation_deg
            ):
                if args.registration_verbose:
                    print(
                        f"[registration] skip frame={i}, time={pcd_time}: "
                        f"large correction trans={translation_correction:.3f}m, "
                        f"rot={rotation_correction:.1f}deg, "
                        f"fitness={fitness:.3f}, corr={n_correspondence}"
                    )
                continue

            gripper_pose_from_lidar = icp_transformation @ rel_frame

            gripper_pcd_transformed = copy.deepcopy(gripper_pcd)
            gripper_pcd_transformed.transform(icp_transformation)
            gripper_pcd_full.transform(icp_transformation)
            gripper_pcd_list.append(
                gripper_pcd_full.transform(np.linalg.inv(gripper_pose_from_lidar))
            )

            filtered_pcd = target_pcd
            if len(filtered_pcd.points) < 50:
                continue

            filtered_pcd_transformed = copy.deepcopy(filtered_pcd)
            filtered_pcd_transformed.transform(np.linalg.inv(gripper_pose_from_lidar))

            rearranged_pcd_list.append(filtered_pcd_transformed)

            if visualize:
                print(f"Joint angles: {q.transpose()}")
                gripper_pcd.paint_uniform_color([0.2, 1.0, 0.2])
                gripper_pcd_transformed.paint_uniform_color([1.0, 0.2, 0.2])
                filtered_pcd.paint_uniform_color([0.2, 0.2, 1.0])
                # target_pcd.paint_uniform_color([0.8, 0.8, 0.2])

                lidar_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=1.0, origin=[0, 0, 0]
                )
                lidar_coord.transform(lidar_frame)

                target_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )
                o3d.visualization.draw_geometries(
                    # [gripper_pcd, gripper_pcd_transformed, filtered_pcd, target_pcd]
                    # [gripper_pcd_transformed, target_pcd, filtered_pcd]
                    [filtered_pcd, gripper_pcd_transformed]
                    + transformed_meshes
                    + [lidar_coord]
                )

        if use_pose_graph and len(rearranged_pcd_list) >= 2:
            print(f"Pose-graph refinement on {len(rearranged_pcd_list)} frames...")
            try:
                refinements = refine_pose_graph(
                    rearranged_pcd_list,
                    verbose=args.pose_graph_verbose,
                )
                for i, T in enumerate(refinements):
                    rearranged_pcd_list[i].transform(T)
                    gripper_pcd_list[i].transform(T)
            except Exception as e:
                print(f"[WARNING] pose-graph refinement failed, skipping: {e}")

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd_with_gripper = o3d.geometry.PointCloud()
        filtered_frames = []
        for i, pcds in enumerate(zip(rearranged_pcd_list, gripper_pcd_list)):
            target_pcd, gripper_pcd = pcds
            try:
                merged_pcd_with_gripper += target_pcd
                filtered_pcd = remove_points_from_points(
                    gripper_pcd,
                    target_pcd,
                    distance_threshold,
                    cluster=True,
                )
                filtered_pcd = _crop_nonnegative_y(filtered_pcd)
                filtered_frames.append((i, filtered_pcd))
            except Exception as e:
                print(f"[WARNING] clustering failed in removing gripper PCD: {e}")

        rejected_frames = []
        if args.review_merge and filtered_frames:
            filtered_frames, rejected_frames = review_filtered_frames(
                filtered_frames,
                preview_voxel=args.review_preview_voxel,
            )

        filtered_pcd_list = [pcd for _, pcd in filtered_frames]
        if args.direct_merge:
            for _, filtered_pcd in filtered_frames:
                merged_pcd += filtered_pcd

        if not args.direct_merge and filtered_pcd_list:
            print(
                f"Incremental ICP grouping/merge on {len(filtered_pcd_list)} "
                "filtered frames..."
            )
            entries = [
                (frame_id, pathlib.Path(f"filtered_{frame_id}.pcd"), pcd)
                for frame_id, pcd in filtered_frames
            ]
            groups, skipped = build_incremental_groups(
                entries,
                group_voxel=args.merge_group_voxel,
                min_fitness=args.merge_frame_fitness_threshold,
                min_correspondences=args.merge_frame_min_correspondences,
                max_translation=args.merge_frame_max_translation,
                max_rotation_deg=args.merge_frame_max_rotation_deg,
                new_group_min_fitness=args.merge_new_group_min_fitness,
                new_group_min_correspondences=args.merge_new_group_min_correspondences,
                verbose=args.merge_verbose,
            )
            print(
                "[grouping] result: "
                + ", ".join(
                    f"group {g.group_id}: {len(g.frame_ids)} frames" for g in groups
                )
            )
            save_groups(csv_dir, groups, args.merge_groups_output_dir)
            save_skipped(csv_dir, skipped, f"{args.merge_groups_output_dir}_skipped")
            merged_pcd, merged_group_ids = merge_groups(
                groups,
                group_voxel=args.merge_group_voxel,
                final_voxel=args.merge_final_voxel,
                min_fitness=args.merge_group_fitness_threshold,
                min_correspondences=args.merge_group_min_correspondences,
                max_translation=args.merge_group_max_translation,
                max_rotation_deg=args.merge_group_max_rotation_deg,
                verbose=args.merge_verbose,
            )
            print(f"[grouping] merged groups: {merged_group_ids}")

        if len(merged_pcd.points) > 0:
            labels = np.array(
                merged_pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=False)
            )
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0:
                largest = labels == np.bincount(labels[labels >= 0]).argmax()
                merged_pcd = merged_pcd.select_by_index(np.where(largest)[0])

        if visualize:
            o3d.visualization.draw_geometries([merged_pcd_with_gripper, origin_coord])
            o3d.visualization.draw_geometries([merged_pcd, origin_coord])
        elif show_result:
            _draw_geometries_with_timeout(
                [merged_pcd_with_gripper, origin_coord],
                "Merged PCD with gripper",
                args.show_result_timeout,
            )
            _draw_geometries_with_timeout(
                [merged_pcd, origin_coord],
                "Merged PCD",
                args.show_result_timeout,
            )
        o3d.io.write_point_cloud(os.path.join(str(csv_dir), "merged.pcd"), merged_pcd)
        os.makedirs(os.path.join(str(csv_dir), f"filtered"), exist_ok=True)
        for frame_id, pcd in filtered_frames:
            o3d.io.write_point_cloud(
                os.path.join(str(csv_dir), f"filtered/filtered_{frame_id}.pcd"), pcd
            )
        if rejected_frames:
            os.makedirs(
                os.path.join(str(csv_dir), f"filtered_rejected"),
                exist_ok=True,
            )
            for frame_id, pcd in rejected_frames:
                o3d.io.write_point_cloud(
                    os.path.join(
                        str(csv_dir),
                        f"filtered_rejected/filtered_{frame_id}.pcd",
                    ),
                    pcd,
                )
        merged_pcd_groups += 1
        print(
            f"[progress] merged PCD groups: "
            f"{merged_pcd_groups}/{total_pcd_groups} "
            f"(skipped={skipped_pcd_groups})"
        )

    print(
        f"Finished merging PCD groups: "
        f"merged={merged_pcd_groups}, skipped={skipped_pcd_groups}, "
        f"total={total_pcd_groups}"
    )


if __name__ == "__main__":

    if sys.gettrace():
        sys.argv = [__file__, "data/stone_pcd/pcd_251024/stone01"]

    get_stone_pcd()
