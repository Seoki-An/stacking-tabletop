import os
import numpy as np
from typing import List, Tuple
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

from diffsimpy import diffsim, planner
from model import MANIPULATOR_PATH, GRIPPER_PATH

Q_HOME = np.array([-np.pi / 3, np.pi / 4, -np.pi / 3, -np.pi / 2 + 0.2, 0.0, 0.0])
Q_SCAN = np.array([0.0, np.pi / 4, -np.pi / 3, -np.pi / 2 + 0.2, 0.0, 0.0])
Q_SCAN_SCENE = np.array([0.0, 20.0 / 180 * np.pi, -20.0 / 180 * np.pi, 0.0, 0.0, 0.0])

Q_SCAN_INHAND = Q_SCAN
Q_SCAN_INHAND[0] = -(115.0 / 90.0) * np.pi / 2


def regrasp_position_candidates(base_xy: np.ndarray) -> List[np.ndarray]:
    base_xy = np.asarray(base_xy, dtype=float)
    offsets = (
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
    )
    return [base_xy + np.asarray(offset, dtype=float) for offset in offsets]


def motion_failure_stage(result) -> str:
    stage = getattr(result, "failure_stage", "")
    detail = getattr(result, "failure_detail", "")
    if stage:
        return str(stage)
    if detail:
        return str(detail)
    return ""


def motion_failure_detail(result) -> str:
    detail = getattr(result, "failure_detail", "")
    return str(detail) if detail else ""


def motion_failure_summary(result) -> str:
    stage = str(getattr(result, "failure_stage", "") or "")
    detail = str(getattr(result, "failure_detail", "") or "")
    parts = []
    if stage:
        parts.append(f"failure_stage={stage!r}")
    if detail:
        parts.append(f"failure_detail={detail!r}")
    return " ".join(parts) if parts else "failure_stage='unknown'"


def motion_result_summary(result) -> str:
    stage = str(getattr(result, "failure_stage", "") or "")
    detail = str(getattr(result, "failure_detail", "") or "")
    # Render stage/detail raw (not repr) and on their own lines so the
    # multi-line failure_detail built by the planner stays readable instead of
    # collapsing into one giant escaped string.
    return (
        f"feasible={bool(getattr(result, 'is_feasible', False))} "
        f"selected_sequence_length={len(getattr(result, 'q_path_sequence', []) or [])} "
        f"candidate_sequences={len(getattr(result, 'q_path_sequences', []) or [])}\n"
        f"  failure_stage = {stage}\n"
        f"  failure_detail = {detail}"
    )


def _motion_failure_text(result) -> str:
    return "|".join(
        str(value)
        for value in (
            getattr(result, "failure_stage", ""),
            getattr(result, "failure_detail", ""),
        )
        if value
    )


def motion_failure_has_place_part(result) -> bool:
    return "place" in _motion_failure_text(result)


def motion_failure_can_retry_regrasp_xy(result) -> bool:
    text = _motion_failure_text(result)
    if not text:
        return False
    if "place" in text or "pick" in text:
        return False
    # The B grasp is solved at the (fixed) place pose B; regrasp_xy only moves
    # the intermediate scene C. A B-grasp-generation failure is therefore
    # invariant to regrasp_xy, so retrying another regrasp_xy re-solves the
    # identical B grasp and fails identically -- skip it.
    if "regrasp_b_grasp_generation" in text:
        return False
    return "regrasp" in text


def _candidate_indices(result) -> List[int]:
    sequences = list(getattr(result, "q_path_sequences", []) or [])
    if not sequences:
        return []

    feasible = list(getattr(result, "is_feasible_sequence", []) or [])
    if not feasible:
        return list(range(len(sequences)))

    return [
        idx
        for idx in range(len(sequences))
        if idx < len(feasible) and bool(feasible[idx])
    ]


def num_regrasp_candidates(result) -> int:
    if len(result.q_path_sequence) == 0 or result.is_feasible is False:
        return 0

    indices = _candidate_indices(result)
    if indices:
        return len(indices)
    return 1


def select_regrasp_candidate(result, candidate_idx: int):
    indices = _candidate_indices(result)
    if not indices:
        return result
    if candidate_idx < 0 or candidate_idx >= len(indices):
        raise IndexError(
            f"Candidate index {candidate_idx} is out of range for "
            f"{len(indices)} regrasp candidates"
        )

    src_idx = indices[candidate_idx]
    q_sequences = list(result.q_path_sequences)
    target_sequences = list(getattr(result, "target_path_sequences", []) or [])
    grasp_sequences = list(getattr(result, "grasp_sequences", []) or [])
    settled_sequences = list(getattr(result, "settled_target_path_sequences", []) or [])
    scores = list(getattr(result, "scores", []) or [])
    failure_stages = list(getattr(result, "failure_stage_sequence", []) or [])

    q_path_sequence = list(q_sequences[src_idx])
    target_path_sequence = (
        list(target_sequences[src_idx]) if src_idx < len(target_sequences) else []
    )
    settled_target_path_sequence = (
        list(settled_sequences[src_idx])
        if src_idx < len(settled_sequences)
        else list(getattr(result, "settled_target_path_sequence", []) or [])
    )
    grasp_sequence = (
        list(grasp_sequences[src_idx]) if src_idx < len(grasp_sequences) else []
    )
    score = float(scores[src_idx]) if src_idx < len(scores) else None
    failure_stage = (
        str(failure_stages[src_idx])
        if src_idx < len(failure_stages)
        else getattr(result, "failure_stage", "")
    )

    return SimpleNamespace(
        is_feasible=True,
        q_path_sequence=q_path_sequence,
        target_path_sequence=target_path_sequence,
        settled_target_path_sequence=settled_target_path_sequence,
        grasp_sequence=grasp_sequence,
        q_path_sequences=[q_path_sequence],
        target_path_sequences=[target_path_sequence],
        settled_target_path_sequences=[settled_target_path_sequence],
        grasp_sequences=[grasp_sequence],
        is_feasible_sequence=[True],
        scores=[] if score is None else [score],
        failure_stage=failure_stage,
        failure_detail=getattr(result, "failure_detail", ""),
        failure_stage_sequence=[failure_stage],
    )


def regrasp_candidate_score(result, candidate_idx: int):
    indices = _candidate_indices(result)
    if not indices:
        scores = list(getattr(result, "scores", []) or [])
        return float(scores[0]) if scores else None

    scores = list(getattr(result, "scores", []) or [])
    src_idx = indices[candidate_idx]
    if src_idx >= len(scores):
        return None
    return float(scores[src_idx])


def normalize_joint_branches(paths, q_home):
    """Normalize 2π-range joints across all path segments so they start in the
    same branch as q_home and are continuous throughout.

    When the optimizer takes the shorter arc for joints with large ranges (j0,
    j5 range [-10, 10]), it may represent q_home as +2π ≈ 6.28 instead of 0.
    This causes a raw 6.28-rad jump at step boundaries in the visualization
    even though the physical joint position is identical.

    Applies to ALL 4 segments concatenated so the pick junction (seg[0]/seg[1]
    ↔ seg[2]/seg[3]) is also consistent.
    """
    all_configs = []
    lengths = []
    for p in paths:
        arr = [np.asarray(c, dtype=float) for c in p]
        all_configs.extend(arr)
        lengths.append(len(arr))

    if not all_configs:
        return paths

    n_joints = len(q_home)

    # Step 1: normalize first config to be in the same branch as q_home.
    normalized = [c.copy() for c in all_configs]
    for j in range(n_joints):
        diff = float(normalized[0][j] - q_home[j])
        k = int(round(diff / (2 * np.pi)))
        if k != 0:
            normalized[0][j] -= 2 * np.pi * k

    # Step 2: ensure continuity — no 2π jumps between consecutive configs.
    for t in range(1, len(normalized)):
        for j in range(n_joints):
            diff = float(normalized[t][j] - normalized[t - 1][j])
            k = int(round(diff / (2 * np.pi)))
            if k != 0:
                normalized[t][j] -= 2 * np.pi * k

    # Split back into original segment lengths.
    result_paths = []
    idx = 0
    for l in lengths:
        result_paths.append(normalized[idx : idx + l])
        idx += l
    return result_paths


def split_regrasp_place_paths(result, n_move: int, n_grasp: int, q_home=None):
    if len(result.q_path_sequence) == 2:
        idx = 0
    elif len(result.q_path_sequence) == 4:
        idx = 2
    elif len(result.q_path_sequence) == 6:
        idx = 4
    elif len(result.q_path_sequence) == 8:
        idx = 6
    elif len(result.q_path_sequence) == 10:
        idx = 8
    else:
        raise ValueError(
            f"Unexpected path sequence length: {len(result.q_path_sequence)}"
        )
    path1 = result.q_path_sequence[idx][:n_move]
    path2 = result.q_path_sequence[idx][n_move:]
    path3 = result.q_path_sequence[idx + 1][:n_grasp]
    path4 = result.q_path_sequence[idx + 1][n_grasp:]
    if q_home is not None:
        path1, path2, path3, path4 = normalize_joint_branches(
            [path1, path2, path3, path4], q_home
        )
    return path1, path2, path3, path4


def _interpolate_joint_segment(q_from: np.ndarray, q_to: np.ndarray, n: int):
    q_from = np.asarray(q_from, dtype=np.float64)
    q_to = np.asarray(q_to, dtype=np.float64)
    if n <= 1:
        return [q_from.copy()]

    delta = (q_to - q_from + np.pi) % (2 * np.pi) - np.pi
    return [q_from + delta * (i / (n - 1)) for i in range(n)]


def _reposition_path(q_start: np.ndarray, q_mid: np.ndarray, q_end: np.ndarray):
    first = _interpolate_joint_segment(q_start, q_mid, 10)
    second = _interpolate_joint_segment(q_mid, q_end, 10)
    return first[:-1] + second


def _resample_joint_path(path, n_samples: int):
    if len(path) == 0 or n_samples <= 0:
        return []
    if len(path) == n_samples:
        return list(path)
    if n_samples == 1:
        return [np.asarray(path[-1], dtype=np.float64).copy()]

    idxs = np.linspace(0, len(path) - 1, n_samples)
    out = []
    for idx in idxs:
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        if lo == hi:
            out.append(np.asarray(path[lo], dtype=np.float64).copy())
            continue
        t = idx - lo
        q_lo = np.asarray(path[lo], dtype=np.float64)
        q_hi = np.asarray(path[hi], dtype=np.float64)
        delta = (q_hi - q_lo + np.pi) % (2 * np.pi) - np.pi
        out.append(q_lo + t * delta)
    return out


def _replace_move_prefix_with_reposition(
    q_path_sequence,
    target_path_sequence,
    q_start: np.ndarray,
    q_mid: np.ndarray,
    q_end: np.ndarray,
    n_move: int,
):
    reposition = _reposition_path(q_start, q_mid, q_end)
    if len(q_path_sequence) == 0 or len(reposition) <= 1:
        return q_path_sequence, target_path_sequence

    q_path_sequence = [list(path) for path in q_path_sequence]
    target_path_sequence = [list(path) for path in target_path_sequence]
    native_move = q_path_sequence[0][:n_move]
    native_rest = q_path_sequence[0][n_move:]
    move_with_reposition = reposition[:-1] + native_move
    q_path_sequence[0] = (
        _resample_joint_path(move_with_reposition, n_move) + native_rest
    )
    return q_path_sequence, target_path_sequence


def _apply_reposition_to_result(
    result,
    q_start: np.ndarray,
    q_mid: np.ndarray,
    q_end: np.ndarray,
    n_move: int,
):
    result.q_path_sequence, result.target_path_sequence = (
        _replace_move_prefix_with_reposition(
            result.q_path_sequence,
            result.target_path_sequence,
            q_start,
            q_mid,
            q_end,
            n_move,
        )
    )
    q_path_sequences = list(result.q_path_sequences)
    target_path_sequences = list(result.target_path_sequences)
    result.q_path_sequences = []
    result.target_path_sequences = []
    for q_seq, target_seq in zip(q_path_sequences, target_path_sequences):
        q_seq, target_seq = _replace_move_prefix_with_reposition(
            q_seq,
            target_seq,
            q_start,
            q_mid,
            q_end,
            n_move,
        )
        result.q_path_sequences.append(q_seq)
        result.target_path_sequences.append(target_seq)


def get_planner(
    pick_plane_height: float = 0.0,
    place_plane_height: float | None = None,
    n_threads: int = 20,
) -> Tuple[planner.Context, planner.Config]:

    config = planner.Config()

    config.graspgen.cost.antipodal_1 = 5.0
    config.graspgen.cost.antipodal_2 = 2.0
    config.graspgen.cost.align = 1.0
    config.graspgen.cost.enclosure = 7.0
    config.graspgen.cost.dist = 3.0
    config.graspgen.cost.contact = 3.0
    config.graspgen.cost.sqnorm = 5.0
    config.graspgen.cost.teeth_fit = 2.0
    config.graspgen.cost.teeth_align = 1.0

    config.graspgen.score.sqnorm = 3.0
    config.graspgen.score.align = 1.0
    config.graspgen.score.enclosure = 7.0
    config.graspgen.score.dist = 3.0
    config.graspgen.score.teeth_gap = 2.0
    config.graspgen.score.teeth_fit = 1.0
    config.graspgen.score.teeth_align = 1.0

    config.graspgen.force.wrench = 1e-1
    config.graspgen.force.comp = 10.0
    config.graspgen.force.cone = 10.0
    config.graspgen.force.rho = 1e-4
    config.graspgen.force.moment_align = 10.0
    config.graspgen.mu = 1.0

    config.graspgen.separate_margin = 0.01
    config.graspgen.plane_separate_margin = -0.02
    config.graspgen.contact_margin = 0.05
    config.graspgen.num_samples = 120
    config.graspgen.n_threads = n_threads
    config.graspgen.surface_grasp_init = True
    config.graspgen.surface_grasp_num_dirs = 80
    config.graspgen.surface_grasp_num_base_yaw = 12
    config.graspgen.surface_grasp_base_yaw_step = np.pi / 6
    config.graspgen.surface_grasp_offset_scale = 0.5
    config.graspgen.gripper_max_width = 1.6
    config.graspgen.scene_clearance_margin = 0.05
    config.graspgen.yz_cos_threshold = np.cos(5 * np.pi / 12)

    config.graspgen.alm.iter = 20
    config.graspgen.alm.tol_ineq = 0.015
    config.graspgen.alm.tol_eq = 1e-3
    config.graspgen.alm.beta_ineq_increase_rate = 10.0
    config.graspgen.alm.beta_eq_increase_rate = 10.0
    config.graspgen.alm.beta_ineq_init = 50.0
    config.graspgen.alm.beta_eq_init = 50.0

    config.tr.tol_eps = 1e-12
    config.tr.delta_init = 1.0
    config.tr.max_iter = 200

    config.gripper.height = 1.5
    config.gripper.virtual_closing_torque = 0.5
    config.gripper.virtual_approach_force = 5.0
    config.gripper.sim_damping = 2.0
    config.gripper.sim_step = 500
    config.gripper.sim_dt = 0.01
    config.gripper.set_sim_grasp_offset_tol(np.array([0.5, 1.0, 0.6]))
    config.gripper.set_opening_angle_range(np.array([-np.pi * 80 / 180, 0.0]))

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config.gripper.file_name = os.path.join(cur_dir, GRIPPER_PATH, "stone_grab.urdf")
    config.manipulator.file_name = os.path.join(
        cur_dir, MANIPULATOR_PATH, "vdk23_cx_ik.urdf"
    )
    config.manipulator.end_effector_name = "grip_body"

    config.motion_planning.w_smooth = 50.0
    config.motion_planning.w_collision = 1000.0
    config.motion_planning.w_joint_limit = 1000.0
    config.motion_planning.w_boundary = 20.0
    # Tightened from 10 -> 30 so the relaxed grasped place endpoint hugs the
    # (collision-free) designed place pose: a few-mm EE drift was amplified by
    # the EE->stone lever arm into ~1 cm of stone penetration at placement,
    # tripping target_collision_tol even when the designed place gap is ~0.
    config.motion_planning.grasped_boundary_pos_scale = 10.0
    config.motion_planning.grasped_boundary_rot_scale = 0.5
    config.motion_planning.eps_gap = 2e-2
    # Huber-clamp the collision penetration depth at eps_gap so a deep
    # penetration (e.g. the grasped stone briefly several cm into a neighbour)
    # gives a bounded, constant outward push instead of an explosive
    # w_collision*(eps_gap-gap) gradient that the weak smoothness term cannot
    # resist -> a velocity jump at that waypoint. Set == eps_gap: margin
    # avoidance (gap in [0,eps_gap]) stays fully quadratic; only penetration
    # (gap<0) is capped at the surface force.
    config.motion_planning.collision_penetration_clamp = 2e-2
    config.motion_planning.eps_joint = 1e-3
    config.motion_planning.tol_gap = -0.005
    config.motion_planning.max_iter = 400
    # Looser FISTA convergence for the grasped pick-place motion solves: ~88% of
    # grasped FISTA runs hit max_iter at 1e-6 yet are already feasible, so 1e-4
    # stops once the path is settled -> ~4.4x faster grasped solves, feasibility
    # preserved, +0.01 mean per-step jump (denoised, plan_260614_3).
    config.motion_planning.grasped_motion_tol = 1e-4
    config.motion_planning.alpha = 1e-2
    config.motion_planning.regrasp_enabled = True

    config.motion_planning.free_alm_enabled = True
    config.motion_planning.free_alm_beta_init = 50.0
    config.motion_planning.free_alm_beta_increase = 5.0

    config.motion_planning.grasped_alm_enabled = True
    config.motion_planning.grasped_alm_beta_init = 50.0
    config.motion_planning.grasped_alm_beta_increase = 5.0

    # Keep the native in-hand put-down + regrasp fallback available; the actual
    # toggle stays inhand_replan_mode ("direct" calls the fallback-free overload,
    # "regrasp" calls the 9-arg overload that uses this fallback).
    config.motion_planning.inhand_regrasp_enabled = True
    config.motion_planning.scene_approach_min_up_comp = np.sin(np.pi / 6)
    config.motion_planning.target_collision_tol = 5e-3

    config.motion_planning.tol_ik = 1e-3
    config.motion_planning.max_iter_ik = 2000
    config.motion_planning.grasp_score_threshold = 4.0

    context = planner.Context(config)

    if place_plane_height is None:
        place_plane_height = pick_plane_height

    if abs(float(pick_plane_height) - float(place_plane_height)) < 1e-9:
        if abs(float(pick_plane_height)) < 1e-9:
            context.add_plane(diffsim.PlaneGeometryConfig())
        elif hasattr(context, "add_plane_at_height"):
            context.add_plane_at_height(float(pick_plane_height))
        else:
            raise RuntimeError(
                "planner.Context needs add_plane_at_height for nonzero plane height"
            )
    elif hasattr(context, "add_pick_plane_at_height") and hasattr(
        context, "add_place_plane_at_height"
    ):
        context.add_pick_plane_at_height(float(pick_plane_height))
        context.add_place_plane_at_height(float(place_plane_height))
    else:
        raise RuntimeError(
            "planner.Context needs split plane-height bindings for different "
            "pick/place plane heights"
        )

    return context, config


def get_grasp_init(
    offset: np.ndarray,
    euler_angles: np.ndarray,
    euler_order: str = "ZYX",
) -> planner.Grasp:

    grasp_init = planner.Grasp()
    grasp_init.pose.setPosition(offset)

    quat = Rotation.from_euler(euler_order, euler_angles).as_quat()
    grasp_init.pose.setOrientation(quat)

    grasp_init.opening_angle = -0.01

    return grasp_init


def solve_single_grasp_and_ik(
    context: planner, target: diffsim.BodyConfig, grasp_init: planner.Grasp
) -> Tuple[np.ndarray, planner.GraspSolution]:
    grasp_sol = context.solve_single_grasp(target, grasp_init, True)

    xyz = grasp_sol.grasp.pose.position()

    q1 = np.arctan2(xyz[1], xyz[0])
    z_axis = Rotation.from_quat(grasp_sol.grasp.pose.orientation()).as_matrix()[:, -1]
    q6 = np.arctan2(z_axis[1], z_axis[0]) - q1 + np.pi

    q_init = np.array([q1, 1.0, -1.5, -1.0, 0.0, q6])
    q_sol = context.inverse_kinematics(
        q_init, grasp_sol.grasp.pose, max_iter=100, tol=1e-3
    )

    return q_sol, grasp_sol


def solve_dualscene_grasp_and_ik(
    context: planner,
    target_pick: diffsim.BodyConfig,
    target_place: diffsim.BodyConfig,
    grasp_init: planner.Grasp,
) -> Tuple[np.ndarray, planner.GraspSolution]:
    grasp_sol = context.solve_dualscene_grasp(
        "pick", "place", target_pick, target_place, grasp_init
    )

    xyz = grasp_sol.grasp.pose.position()

    q1 = np.arctan2(xyz[1], xyz[0])
    z_axis = Rotation.from_quat(grasp_sol.grasp.pose.orientation()).as_matrix()[:, -1]
    q6 = np.arctan2(z_axis[1], z_axis[0]) - q1 + np.pi

    q_init = np.array([q1, 1.0, -1.5, -1.0, 0.0, q6])
    ik_sol = context.inverse_kinematics(
        q_init, grasp_sol.grasp.pose, max_iter=100, tol=1e-3
    )

    return ik_sol, grasp_sol


def free_motion_planning(
    context: planner,
    target: diffsim.BodyConfig,
    q_init: np.ndarray,
    n_move: int = 100,
    n_grasp: int = 50,
    scene_name: str = "pick",
) -> List[np.ndarray]:

    grasp_init = planner.Grasp()
    grasp_init.pose.setPosition(np.array([0, 0, 2.0]) + target.pose.position())

    angle = np.arctan2(target.pose.position()[1], target.pose.position()[0])
    grasp_init.pose.setOrientation(
        Rotation.from_euler("ZYX", np.array([angle - np.pi, 0.0, -np.pi / 2])).as_quat()
    )
    grasp_init.opening_angle = -0.01

    ik_sol, grasp_sol = solve_single_grasp_and_ik(context, target, grasp_init)
    q_fin = ik_sol.q
    print("Solving grasp done")

    pose_mid = diffsim.Pose(grasp_sol.grasp.pose.vectorized())
    pose_mid.setPosition(pose_mid.position() + np.array([0, 0, 1.0]))

    ik_sol_mid = context.inverse_kinematics(q_fin, pose_mid, max_iter=100, tol=1e-4)
    q_mid = ik_sol_mid.q

    pick_angle = -0.20
    planning_sol_fin = context.free_motion_planning_with_multigoals(
        q_init,
        [q_mid, q_fin],
        scene_name,
        target,
        pick_angle,
        [n_move, n_grasp],
        1000,
    )

    path = planning_sol_fin.path
    is_feasible = planning_sol_fin.is_feasible

    path = [np.concatenate([q_t, pick_angle * np.ones(2)]) for q_t in path]
    for _ in range(10):
        path.append(path[-1])

    n_grasp = 30
    for t in range(n_grasp):
        angle = (t / n_grasp) * (grasp_sol.grasp.opening_angle - 0.05) + (
            1 - t / n_grasp
        ) * pick_angle
        path.append(np.concatenate([q_fin, angle * np.ones(2)]))

    return path, is_feasible


def regrasp_planning(
    context: planner,
    pick_target: diffsim.BodyConfig,
    place_target: diffsim.BodyConfig,
    q_start: np.ndarray,
    regrasp_xy_pos: np.ndarray,
    n_move: int = 100,
    n_grasp: int = 50,
    max_num_solutions: int = 1,
    q_mid: np.ndarray = None,
    q_end: np.ndarray = None,
) -> planner.RegraspSolution:
    if q_mid is None:
        q_mid = q_start
    if q_end is None:
        q_end = q_mid

    result = context.regrasp_planning(
        pick_target,
        place_target,
        q_end,
        regrasp_xy_pos,
        n_move,
        n_grasp,
        max_num_solutions,
    )
    if result.is_feasible and (
        not np.allclose(q_start, q_mid, atol=1e-6)
        or not np.allclose(q_mid, q_end, atol=1e-6)
    ):
        _apply_reposition_to_result(result, q_start, q_mid, q_end, n_move)
    return result


def solve_inhand_grasp_planning(
    context: planner,
    q_home: np.ndarray,
    inhand_pose: diffsim.Pose,
    place_target: diffsim.BodyConfig,
    opening_angle: float,
    n_move: int = 100,
    n_grasp: int = 50,
    enable_near_ik: bool = False,
    regrasp_xy_pos: np.ndarray = None,
    max_num_solutions: int = 1,
    inhand_replan_mode: str = "regrasp",
) -> planner.RegraspSolution:
    mode = str(inhand_replan_mode).strip().lower()
    if mode not in {"direct", "regrasp"}:
        raise ValueError("inhand_replan_mode must be either 'direct' or 'regrasp'")

    if mode == "direct" or regrasp_xy_pos is None:
        result = context.solve_inhand_grasp_planning(
            inhand_pose,
            place_target,
            q_home,
            opening_angle,
            n_move,
            n_grasp,
            enable_near_ik,
        )
    else:
        try:
            result = context.solve_inhand_grasp_planning(
                inhand_pose,
                place_target,
                q_home,
                opening_angle,
                n_move,
                n_grasp,
                enable_near_ik,
                regrasp_xy_pos,
                max_num_solutions,
            )
        except TypeError:
            result = context.solve_inhand_grasp_planning(
                inhand_pose,
                place_target,
                q_home,
                opening_angle,
                n_move,
                n_grasp,
                enable_near_ik,
            )
    return result


if __name__ == "__main__":

    print("planning-test")
