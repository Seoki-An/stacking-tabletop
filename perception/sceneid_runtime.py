"""Runtime SceneID helpers for live execution.

This module keeps live execution entry points out of the desktop log-processing
scripts while reusing the existing SceneID/ICP utilities. Scene scans are
captured on the NUC, but live scene identification now runs on the desktop.
"""

import argparse
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from agent.env.components.contexts import get_sceneid
from model import get_stone_model
from scripts.desktop.sceneid_from_logs import (
    PoseVector,
    apply_manual_initial_poses,
    apply_ground_height_to_plan_poses,
    apply_recursive_initial_poses,
    apply_sceneid_args_to_config,
    estimate_ground_height_ransac,
    fix_locked_solution_poses,
    mark_user_fixed_step_sources,
    mark_ground_height_sources,
    pose_map_from_solution,
    read_or_merge_scene_pcd,
    refine_plan_initial_poses_with_icp,
    filter_sceneid_entries_by_point_support,
    remove_ground_points_by_plane_model,
    run_manual_fix_gui,
    run_manual_initialization_gui,
    save_manual_init_pose_file,
    set_sceneid_ground_height,
    step_index_from_scene_dir,
    write_results,
)
from scripts.desktop.sceneid_from_plan import (
    add_planned_bodies,
    crop_scene_pcd_near_expected_poses,
    downsample_and_select_points,
    remove_ground_plane_ransac,
    select_visual_meshes,
    visualize_result,
)


def make_sceneid_runtime_args() -> argparse.Namespace:
    return argparse.Namespace(
        plan_dir=None,
        asset_dir=None,
        output_prefix="sceneid_scene_poses",
        merged_name="scene_scan_merged_for_sceneid.ply",
        remerge=False,
        raw=False,
        merge_voxel_size=0.0,
        no_recursive_init=False,
        no_fix_missing_prior_steps=False,
        fix_initial_steps=[],
        manual_init_by_step={},
        manual_init_fixed=False,
        manual_init_gui=False,
        manual_init_gui_output=None,
        manual_fix_gui=False,
        manual_init_gui_translation_step=0.01,
        manual_init_gui_rotation_step_deg=5.0,
        manual_init_gui_point_size=2.0,
        manual_init_gui_normal_radius=0.25,
        manual_init_gui_normal_max_nn=30,
        manual_init_gui_voxel_size=0.02,
        no_offset=False,
        voxel_size=0.02,
        max_points=20000,
        crop_margin=0.5,
        crop_max_margin=3.0,
        no_crop=False,
        no_ground_removal=False,
        ground_distance_threshold=0.08,
        ground_ransac_n=3,
        ground_num_iterations=1000,
        ground_normal_z_min=0.7,
        ground_min_inlier_ratio=0.02,
        no_ground_height_init=False,
        ground_height_distance_threshold=0.04,
        ground_height_max_abs=1.0,
        no_ground_height_crop=False,
        ground_height_crop_margin=0.75,
        ground_height_crop_max_margin=3.0,
        # Live execution trusts recursive previous-step poses and manual GUI
        # initialization directly. ICP remains available in the offline
        # sceneid_from_logs/sceneid_from_plan tools, but live/test scene scans
        # should not spend time refining many prior stones.
        no_icp_init=True,
        # Keep diffsimpy SceneID disabled until the native solver is ready for
        # live use.
        skip_sceneid_solve=True,
        icp_plan_only=True,
        icp_prior_poses=False,
        icp_axes="xyz",
        icp_angle_step_deg=90.0,
        icp_crop_margin=0.35,
        icp_source_voxel_size=0.01,
        icp_mesh_points=20000,
        icp_voxel_sizes=[0.10, 0.05, 0.02, 0.01],
        icp_max_iters=[50, 30, 14, 7],
        icp_correspondence_distance_scale=1.0,
        icp_min_target_points=50,
        icp_min_correspondences=20,
        icp_min_fitness=0.0,
        icp_prior_min_correspondences=80,
        icp_prior_min_fitness=0.05,
        icp_max_translation=0.6,
        icp_rmse_weight=0.25,
        sceneid_min_stone_points=5000,
        sceneid_stone_point_margin=0.15,
        no_icp_empty_target_prior_fallback=False,
        no_icp_low_correspondence_prior_fallback=False,
        max_iter=20,
        log_interval=1,
        n_threads=0,
        target_structure_offset=None,
        graph_max_iter=100,
        trust_region_eps=0.1,
        delta_init=0.125,
        k_pcd=1.0,
        pcd_huber_delta=0.02,
        pcd_max_gap=0.15,
        k_gap_c=80.0,
        k_comp=0.0,
        visualize=False,
        visualize_solver_pcd=False,
    )


def _target_offset_array(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        offset = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if offset.shape[0] < 2 or not np.all(np.isfinite(offset[:2])):
        return None
    if np.linalg.norm(offset[:2]) <= 1e-9:
        return None
    return offset[:2].copy()


def _pose_xy_error(
    poses_by_stone: dict[int, np.ndarray],
    reference_by_stone: dict[int, np.ndarray],
    xy_shift: np.ndarray | None = None,
) -> tuple[float, int]:
    errors = []
    for stone_id, pose in poses_by_stone.items():
        reference = reference_by_stone.get(int(stone_id))
        if reference is None:
            continue
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        if pose.shape[0] < 2 or reference.shape[0] < 2:
            continue
        xy = pose[:2].copy()
        if xy_shift is not None:
            xy += xy_shift
        if np.all(np.isfinite(xy)) and np.all(np.isfinite(reference[:2])):
            errors.append(float(np.linalg.norm(xy - reference[:2])))
    if not errors:
        return np.inf, 0
    return float(np.median(errors)), len(errors)


def correct_target_offset_frame_if_needed(
    poses_by_stone: dict[int, np.ndarray],
    reference_by_stone: dict[int, np.ndarray],
    target_structure_offset,
) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    """Correct SceneID poses when the output is target-local instead of base-frame.

    A frame slip by exactly ``target_structure_offset`` shows up as the same
    translation error for every stone. Correct it before preview/apply/logging.
    """
    offset = _target_offset_array(target_structure_offset)
    if offset is None or not poses_by_stone:
        return poses_by_stone, None

    raw_error, n_compared = _pose_xy_error(poses_by_stone, reference_by_stone)
    if n_compared == 0 or not np.isfinite(raw_error):
        return poses_by_stone, None

    offset_norm = float(np.linalg.norm(offset))
    candidates = []
    for shift in (offset, -offset):
        shifted_error, _ = _pose_xy_error(
            poses_by_stone,
            reference_by_stone,
            xy_shift=shift,
        )
        votes = 0
        for stone_id, pose in poses_by_stone.items():
            reference = reference_by_stone.get(int(stone_id))
            if reference is None:
                continue
            pose = np.asarray(pose, dtype=np.float64).reshape(-1)
            reference = np.asarray(reference, dtype=np.float64).reshape(-1)
            if pose.shape[0] < 2 or reference.shape[0] < 2:
                continue
            if not (
                np.all(np.isfinite(pose[:2])) and np.all(np.isfinite(reference[:2]))
            ):
                continue
            raw_stone_error = float(np.linalg.norm(pose[:2] - reference[:2]))
            shifted_stone_error = float(
                np.linalg.norm(pose[:2] + shift - reference[:2])
            )
            if raw_stone_error - shifted_stone_error > max(
                0.20, 0.35 * offset_norm
            ) and shifted_stone_error < max(0.35, 0.35 * offset_norm):
                votes += 1
        candidates.append((shift, shifted_error, votes))
    best_shift, best_error, votes = min(candidates, key=lambda item: item[1])

    min_votes = max(1, int(np.ceil(0.5 * n_compared)))
    improves_by_offset = raw_error - best_error > max(0.20, 0.35 * offset_norm)
    close_after_shift = best_error < max(0.35, 0.35 * offset_norm)
    if not (votes >= min_votes and improves_by_offset and close_after_shift):
        return poses_by_stone, None

    corrected = {}
    for stone_id, pose in poses_by_stone.items():
        pose = np.asarray(pose, dtype=np.float64).copy()
        pose[:2] += best_shift
        corrected[int(stone_id)] = pose
    print(
        "  corrected SceneID output frame by target_structure_offset "
        f"{best_shift.tolist()} "
        f"(median xy error {raw_error:.3f}m -> {best_error:.3f}m, "
        f"votes={votes}/{n_compared})"
    )
    return corrected, best_shift


def scene_initial_entries_from_pose_map(
    poses_by_stone: dict[int, np.ndarray] | list[tuple[int, np.ndarray]],
) -> list[tuple[int, np.ndarray]]:
    if isinstance(poses_by_stone, dict):
        items = poses_by_stone.items()
    else:
        items = poses_by_stone

    entries = []
    seen = set()
    for stone_id, pose in items:
        try:
            sid = int(stone_id)
        except (TypeError, ValueError):
            continue
        if sid in seen:
            continue
        arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 7 or not np.all(np.isfinite(arr[:7])):
            continue
        entries.append((sid, arr[:7].copy()))
        seen.add(sid)
    return entries


def live_sceneid_action_prefix(
    action_sequence: list | None,
    initial_entries: list[tuple[int, np.ndarray]],
    step_index: int,
) -> tuple[list, int]:
    if action_sequence:
        prefix = list(action_sequence[:step_index])
        if len(prefix) < step_index:
            step_index = len(prefix)
        return prefix, step_index

    prefix = [{"stone_id": int(stone_id)} for stone_id, _ in initial_entries]
    return prefix, len(prefix)


def manual_gui_scene_pcd(
    scene_pcd: o3d.geometry.PointCloud,
    args: argparse.Namespace,
) -> o3d.geometry.PointCloud:
    voxel = float(
        getattr(
            args,
            "manual_init_gui_voxel_size",
            getattr(args, "voxel_size", 0.0),
        )
        or 0.0
    )
    if voxel > 0.0 and not scene_pcd.is_empty():
        return scene_pcd.voxel_down_sample(voxel)
    return o3d.geometry.PointCloud(scene_pcd)


def identify_scene_dir_from_initial_poses(
    scene_dir: Path,
    args: argparse.Namespace,
    initial_poses_by_stone: dict[int, np.ndarray] | list[tuple[int, np.ndarray]],
    prior_poses_by_stone: dict[int, np.ndarray] | None = None,
    calibrated_ground_height: float | None = None,
    calibrated_ground_plane_model: np.ndarray | None = None,
    asset_dir: str | Path | None = None,
    action_sequence: list | None = None,
) -> tuple[bool, dict[int, np.ndarray], float | None, np.ndarray | None]:
    print(f"\nScene scan: {scene_dir}")
    initial_entries = scene_initial_entries_from_pose_map(initial_poses_by_stone)
    if not initial_entries:
        raise ValueError("no /scene_poseid_init poses were provided")

    prior_poses_by_stone = prior_poses_by_stone or {}
    step_index = step_index_from_scene_dir(scene_dir)
    scene_pcd, scene_pcd_path = read_or_merge_scene_pcd(
        scene_dir,
        args.remerge,
        args.raw,
        args.merge_voxel_size,
        args.merged_name,
    )
    print(
        f"  runtime init: step={step_index}, "
        f"placed stones={len(initial_entries)}, scene points={len(scene_pcd.points)}"
    )

    asset_dir = asset_dir or args.asset_dir or "assets/stone"
    dsf_meshes, stone_configs, stone_pcds, ply_meshes = get_stone_model(asset_dir)
    visual_meshes = select_visual_meshes(dsf_meshes, ply_meshes)

    planned_entries_plan = [(sid, pose.copy()) for sid, pose in initial_entries]
    plan_pose_sources = {sid: "plan_state_runtime" for sid, _ in planned_entries_plan}
    if not args.no_ground_height_init:
        if calibrated_ground_height is None:
            reference_xy = np.mean(
                np.vstack([pose[:2] for _, pose in planned_entries_plan]), axis=0
            )
            ground_height_pcd = scene_pcd
            if not args.no_ground_height_crop:
                try:
                    ground_height_pcd = crop_scene_pcd_near_expected_poses(
                        scene_pcd,
                        scene_pcd_path or scene_dir,
                        planned_entries_plan,
                        visual_meshes,
                        args.ground_height_crop_margin,
                        args.ground_height_crop_max_margin,
                    )
                except Exception as exc:
                    print(
                        "  ground height crop skipped: "
                        f"{exc}; using full scene PCD for calibration"
                    )
            ground_estimate = estimate_ground_height_ransac(
                ground_height_pcd,
                reference_xy,
                args.ground_height_distance_threshold,
                args.ground_ransac_n,
                args.ground_num_iterations,
                args.ground_normal_z_min,
                args.ground_min_inlier_ratio,
                args.ground_height_max_abs,
            )
            if ground_estimate is not None:
                calibrated_ground_height, calibrated_ground_plane_model = (
                    ground_estimate
                )
        else:
            print(
                f"  ground height calibration: reusing z={calibrated_ground_height:.4f}"
            )
    planned_entries_plan = apply_ground_height_to_plan_poses(
        planned_entries_plan,
        None if args.no_ground_height_init else calibrated_ground_height,
    )

    initial_stone_ids = {sid for sid, _ in planned_entries_plan}
    prior_poses_for_step = {
        int(stone_id): pose.copy()
        for stone_id, pose in prior_poses_by_stone.items()
        if int(stone_id) in initial_stone_ids
    }
    prior_poses_for_step, prior_frame_shift = correct_target_offset_frame_if_needed(
        prior_poses_for_step,
        {stone_id: pose.copy() for stone_id, pose in planned_entries_plan},
        args.target_structure_offset,
    )
    if prior_frame_shift is not None:
        print(
            "  corrected recursive prior frame by target_structure_offset "
            f"{prior_frame_shift.tolist()}"
        )
    planned_entries, initial_pose_sources = apply_recursive_initial_poses(
        planned_entries_plan,
        prior_poses_for_step,
        enabled=not args.no_recursive_init,
        base_sources=plan_pose_sources,
    )
    mark_ground_height_sources(
        initial_pose_sources,
        planned_entries,
        None if args.no_ground_height_init else calibrated_ground_height,
    )
    action_prefix, manual_step_index = live_sceneid_action_prefix(
        action_sequence,
        initial_entries,
        step_index,
    )
    planned_entries, manual_applied = apply_manual_initial_poses(
        planned_entries,
        initial_pose_sources,
        action_prefix,
        manual_step_index,
        dict(getattr(args, "manual_init_by_step", {}) or {}),
        bool(getattr(args, "manual_init_fixed", False)),
    )
    if manual_applied:
        print(
            "  manual init poses: "
            + ", ".join(
                f"step {step} -> stone {stone_id}"
                for step, stone_id in manual_applied
            )
            + (
                " (fixed)"
                if getattr(args, "manual_init_fixed", False)
                else " (ICP/sceneid refinable)"
            )
        )
    if getattr(args, "manual_init_gui", False):
        manual_scene_pcd = manual_gui_scene_pcd(scene_pcd, args)
        try:
            planned_entries, manual_gui_poses, _ = run_manual_initialization_gui(
                manual_scene_pcd,
                planned_entries,
                initial_pose_sources,
                action_prefix,
                manual_step_index,
                visual_meshes,
                args,
                scene_dir,
            )
        finally:
            del manual_scene_pcd
        manual_output = getattr(args, "manual_init_gui_output", None)
        if manual_output is not None and manual_gui_poses:
            saved_manual_poses = dict(getattr(args, "manual_init_by_step", {}) or {})
            saved_manual_poses.update(manual_gui_poses)
            save_manual_init_pose_file(Path(manual_output), saved_manual_poses)
        updated_manual_poses = dict(getattr(args, "manual_init_by_step", {}) or {})
        updated_manual_poses.update(manual_gui_poses)
        args.manual_init_by_step = updated_manual_poses
    if getattr(args, "manual_fix_gui", False):
        manual_scene_pcd = manual_gui_scene_pcd(scene_pcd, args)
        try:
            args.fix_initial_steps = run_manual_fix_gui(
                manual_scene_pcd,
                planned_entries,
                initial_pose_sources,
                action_prefix,
                manual_step_index,
                visual_meshes,
                args,
                scene_dir,
                list(getattr(args, "fix_initial_steps", []) or []),
            )
        finally:
            del manual_scene_pcd
    user_fixed_steps = mark_user_fixed_step_sources(
        initial_pose_sources,
        action_prefix,
        manual_step_index,
        list(getattr(args, "fix_initial_steps", []) or []),
    )
    if user_fixed_steps:
        print(
            "  fixing user-specified initialized poses: "
            + ", ".join(
                f"step {step} -> stone {stone_id}"
                for step, stone_id in user_fixed_steps
            )
        )
    frame_reference_poses_by_stone = {
        stone_id: pose.copy() for stone_id, pose in planned_entries
    }

    n_prior = sum(1 for source in initial_pose_sources.values() if source == "prior")
    n_state = sum(
        1 for source in initial_pose_sources.values() if source.startswith("plan_state")
    )
    if n_prior:
        print(
            "  recursive init: using "
            f"{n_prior} prior poses and {n_state} desktop state.pkl poses"
        )
    else:
        print(
            "  recursive init: using desktop state.pkl poses for all "
            f"{len(planned_entries)} stones"
        )

    if not args.no_crop:
        scene_pcd = crop_scene_pcd_near_expected_poses(
            scene_pcd,
            scene_pcd_path or scene_dir,
            planned_entries,
            visual_meshes,
            args.crop_margin,
            args.crop_max_margin,
        )
    visualization_pcd = o3d.geometry.PointCloud(scene_pcd)
    if not args.no_ground_removal:
        filtered_by_model = (
            None
            if args.no_ground_height_init
            else remove_ground_points_by_plane_model(
                scene_pcd,
                calibrated_ground_plane_model,
                args.ground_distance_threshold,
                args.ground_normal_z_min,
            )
        )
        if filtered_by_model is not None:
            scene_pcd = filtered_by_model
        else:
            scene_pcd = remove_ground_plane_ransac(
                scene_pcd,
                args.ground_distance_threshold,
                args.ground_ransac_n,
                args.ground_num_iterations,
                args.ground_normal_z_min,
                args.ground_min_inlier_ratio,
            )
        visualization_pcd = o3d.geometry.PointCloud(scene_pcd)

    (
        planned_entries,
        sceneid_stone_point_counts,
        skipped_low_scene_points,
    ) = filter_sceneid_entries_by_point_support(
        scene_pcd,
        planned_entries,
        initial_pose_sources,
        visual_meshes,
        args,
    )
    if not planned_entries:
        print(
            "  SceneID skipped: no initialized stones have enough nearby scene points"
        )
        write_results(
            scene_dir,
            args.output_prefix,
            Path("<desktop-runtime-state>"),
            Path("/scene_poseid_init"),
            step_index,
            scene_pcd_path,
            {},
            {},
            initial_pose_sources,
            {},
            None if args.no_ground_height_init else calibrated_ground_height,
            None if args.no_ground_height_init else calibrated_ground_plane_model,
            args.skip_sceneid_solve,
            0.0,
            sceneid_stone_point_counts,
            skipped_low_scene_points,
        )
        return (
            False,
            {},
            calibrated_ground_height,
            calibrated_ground_plane_model,
        )

    planned_entries = refine_plan_initial_poses_with_icp(
        scene_pcd,
        planned_entries,
        initial_pose_sources,
        stone_pcds,
        visual_meshes,
        args,
        prior_poses_for_step,
    )

    scene_pcd, points = downsample_and_select_points(
        scene_pcd,
        scene_pcd_path or scene_dir,
        args.voxel_size,
        args.max_points,
    )
    if args.visualize_solver_pcd:
        visualization_pcd = scene_pcd
    print(f"  scene points passed to sceneid: {points.shape[1]}")

    if args.skip_sceneid_solve:
        sceneid_to_stone = {}
        initial_poses = {}
        solution_poses = {}
        optimal_poses_by_stone = {}
        for body_id, (stone_id, pose) in enumerate(planned_entries, start=1):
            pose = pose.copy()
            sceneid_to_stone[body_id] = stone_id
            initial_poses[body_id] = pose
            solution_poses[body_id] = PoseVector(pose)
            optimal_poses_by_stone[stone_id] = pose
        elapsed = 0.0
        print("  skipped diffsimpy sceneid solve; using initialized poses as output")
    else:
        context, config = get_sceneid(
            ground_height=(
                None if args.no_ground_height_init else calibrated_ground_height
            )
        )
        apply_sceneid_args_to_config(config, args)
        set_sceneid_ground_height(
            config,
            None if args.no_ground_height_init else calibrated_ground_height,
        )
        sceneid_to_stone, initial_poses = add_planned_bodies(
            context, planned_entries, stone_configs, stone_pcds
        )

        started = time.time()
        solution = context.solve(points)
        elapsed = time.time() - started
        solution_poses = solution.optimal_poses
        optimal_poses_by_stone = pose_map_from_solution(solution, sceneid_to_stone)

    solution_poses, optimal_poses_by_stone = fix_locked_solution_poses(
        sceneid_to_stone,
        initial_poses,
        initial_pose_sources,
        solution_poses,
        optimal_poses_by_stone,
        args,
    )
    optimal_poses_by_stone, frame_shift = correct_target_offset_frame_if_needed(
        optimal_poses_by_stone,
        frame_reference_poses_by_stone,
        args.target_structure_offset,
    )
    if frame_shift is not None:
        for body_id, stone_id in sceneid_to_stone.items():
            pose = optimal_poses_by_stone.get(int(stone_id))
            if pose is not None:
                solution_poses[int(body_id)] = PoseVector(pose)

    print(f"  solve elapsed: {elapsed:.3f}s")
    for stone_id in sorted(optimal_poses_by_stone):
        print(f"    stone {stone_id}: {optimal_poses_by_stone[stone_id]}")

    write_results(
        scene_dir,
        args.output_prefix,
        Path("<desktop-runtime-state>"),
        Path("/scene_poseid_init"),
        step_index,
        scene_pcd_path,
        sceneid_to_stone,
        initial_poses,
        initial_pose_sources,
        optimal_poses_by_stone,
        None if args.no_ground_height_init else calibrated_ground_height,
        None if args.no_ground_height_init else calibrated_ground_plane_model,
        args.skip_sceneid_solve,
        elapsed,
        sceneid_stone_point_counts,
        skipped_low_scene_points,
    )

    if args.visualize:
        print(f"  visualization scene points: {len(visualization_pcd.points)}")
        visualize_result(
            visualization_pcd,
            visual_meshes,
            sceneid_to_stone,
            initial_poses,
            solution_poses,
            None if args.no_ground_height_init else calibrated_ground_plane_model,
        )
    return (
        True,
        optimal_poses_by_stone,
        calibrated_ground_height,
        calibrated_ground_plane_model,
    )
