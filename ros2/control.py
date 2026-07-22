import time
import numpy as np
from contextlib import nullcontext

from typing import List

import rclpy
from .joint_node import CtrlJointPublisher, CtrlJointSubscriber, project_angle

BOOM_OFFSET_ON = False
BOOM_OFFSET = -1.5 * np.pi / 180.0

GRASP_CLOSE_PUBLISH_COUNT = 20000
OPEN_AFTER_PLACE_PUBLISH_COUNT = 1000
GRASP_CONTROL_SETTLE_TIME = 5.0
POSITION_CONTROL_MAX_PUBLISH_STEP = 0.02  # rad per publish loop
POSITION_CONTROL_MAX_SWING_PUBLISH_STEP = 0.01  # rad per publish loop
POSITION_CONTROL_INITIAL_STATE_TIMEOUT = 0.5
POSITION_CONTROL_SWING_LOG_THRESHOLD = 0.25


def _wrap_swing_rotate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if q.size >= 1:
        q[0] = project_angle(q[0])
    if q.size >= 6:
        q[5] = project_angle(q[5])
    return q


def _joint_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.asarray(a, dtype=np.float64).copy() - np.asarray(b, dtype=np.float64)
    if delta.size >= 1:
        delta[0] = project_angle(delta[0])
    if delta.size >= 6:
        delta[5] = project_angle(delta[5])
    return delta


def _initial_command_position(
    q_desired: np.ndarray,
    joint_node_sub: CtrlJointSubscriber,
    spin_lock=None,
    wait_timeout: float = POSITION_CONTROL_INITIAL_STATE_TIMEOUT,
) -> np.ndarray:
    deadline = time.time() + max(0.0, float(wait_timeout))
    while True:
        with spin_lock if spin_lock is not None else nullcontext():
            rclpy.spin_once(joint_node_sub, timeout_sec=0.01)

        if getattr(joint_node_sub, "last_msg", None) is not None:
            q_current = np.asarray(joint_node_sub.pos, dtype=np.float64).reshape(-1)
            if q_current.size >= q_desired.size and np.all(
                np.isfinite(q_current[: q_desired.size])
            ):
                return q_current[: q_desired.size].copy()

        if time.time() >= deadline:
            return q_desired.copy()


def _interpolated_command(
    q_command: np.ndarray,
    q_desired: np.ndarray,
    max_step: float = POSITION_CONTROL_MAX_PUBLISH_STEP,
    max_swing_step: float = POSITION_CONTROL_MAX_SWING_PUBLISH_STEP,
) -> np.ndarray:
    delta = _joint_delta(q_desired, q_command)
    max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    swing_delta = abs(float(delta[0])) if delta.size >= 1 else 0.0
    if max_delta <= max_step and swing_delta <= max_swing_step:
        return q_desired.copy()

    scales = []
    if max_delta > max_step:
        scales.append(max_step / max_delta)
    if swing_delta > max_swing_step:
        scales.append(max_swing_step / swing_delta)
    scale = min(scales) if scales else 1.0
    return _wrap_swing_rotate(q_command + delta * scale)


def _publish_grab_signal(
    q_desired: np.ndarray,
    joint_node_pub: CtrlJointPublisher,
    grab: str,
    publish_count: int,
) -> None:
    for _ in range(publish_count):
        joint_node_pub.publish(pos=q_desired, grab=grab)


def position_control(
    q_desired: np.ndarray,
    v_desired: np.ndarray,
    joint_node_pub: CtrlJointPublisher,
    joint_node_sub: CtrlJointSubscriber,
    error_tol: float = 0.03,
    time_limit: float = 5.0,
    grab: str = "stay",
    state_cb=None,
    spin_lock=None,
):
    q_desired = _wrap_swing_rotate(np.asarray(q_desired, dtype=np.float64))
    v_desired = np.asarray(v_desired, dtype=np.float64).copy()
    pos_error = 1000000.0
    pos_error_vec = np.zeros_like(q_desired)
    print(f"Publish desired joint angles {q_desired.tolist()}")
    q_command = _initial_command_position(q_desired, joint_node_sub, spin_lock)
    initial_swing_delta = (
        abs(float(_joint_delta(q_desired, q_command)[0]))
        if q_desired.size >= 1
        else 0.0
    )
    if initial_swing_delta > POSITION_CONTROL_SWING_LOG_THRESHOLD:
        print(
            "[INFO] position_control interpolating large swing command: "
            f"delta={initial_swing_delta:.3f} rad, "
            f"max_step={POSITION_CONTROL_MAX_SWING_PUBLISH_STEP:.3f} rad/publish"
        )

    start_time = time.time()
    elapsed_time = 0.0
    while pos_error > error_tol and elapsed_time < time_limit:
        q_command = _interpolated_command(q_command, q_desired)
        joint_node_pub.publish(q_command.copy(), v_desired, grab)
        with spin_lock if spin_lock is not None else nullcontext():
            rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
        if joint_node_sub.get_flag():
            pos_error_vec = _joint_delta(joint_node_sub.pos.copy(), q_desired)
            pos_error = np.linalg.norm(pos_error_vec)
            if state_cb is not None:
                state_cb(joint_node_sub.pos.copy())
        joint_node_sub.reset_get_flag()
        elapsed_time = time.time() - start_time

    if elapsed_time >= time_limit:
        print(
            f"[INFO] position_control reach time limit - pos error: {pos_error_vec.tolist()}"
        )


def sequential_grasp_control(
    path_move: List[np.ndarray],
    path_grasp: List[np.ndarray],
    path_lift: List[np.ndarray],
    path_home: List[np.ndarray],
    joint_node_pub: CtrlJointPublisher,
    joint_node_sub: CtrlJointSubscriber,
    mode: str = "pick",  # "pick" or "place"
    confirm_cb=None,
    pre_grasp_confirm_cb=None,
    state_cb=None,
    phase_cb=None,
    spin_lock=None,
    move_error_tol: float = 0.05,
    move_convergence_time_limit: float = 40.0,
) -> None:

    if mode == "pick":
        grab1, grab2 = "open", "close"
    elif mode == "place":
        grab1, grab2 = "close", "open"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    def _fire_phase(name):
        if phase_cb is not None:
            phase_cb(name)

    _fire_phase("move")
    for q_t in path_move:
        q_d = project_angle(q_t[:6])
        v_d = np.zeros_like(q_d)
        position_control(
            q_d,
            v_d,
            joint_node_pub,
            joint_node_sub,
            error_tol=move_error_tol,
            time_limit=1.0,
            grab=grab1,
            state_cb=state_cb,
            spin_lock=spin_lock,
        )
    position_control(
        q_d,
        v_d,
        joint_node_pub,
        joint_node_sub,
        error_tol=0.017,
        time_limit=move_convergence_time_limit,
        grab=grab1,
        state_cb=state_cb,
        spin_lock=spin_lock,
    )
    approach_offset = np.zeros_like(q_d)
    if pre_grasp_confirm_cb is not None:
        confirm_result = pre_grasp_confirm_cb(q_d.copy(), grab1)
        if isinstance(confirm_result, str):
            if confirm_result == "abort":
                return
        elif confirm_result is not None:
            adjusted_q = project_angle(np.asarray(confirm_result, dtype=np.float64)[:6])
            approach_offset = project_angle(adjusted_q - q_d)
            q_d = adjusted_q

    _fire_phase("approach")
    for q_t in path_grasp:
        q_d = project_angle(q_t[:6] + approach_offset)
        v_d = np.zeros_like(q_d)
        position_control(
            q_d,
            v_d,
            joint_node_pub,
            joint_node_sub,
            error_tol=0.1,
            grab=grab1,
            state_cb=state_cb,
            spin_lock=spin_lock,
        )
    if BOOM_OFFSET_ON and mode == "pick":
        q_d[1] += BOOM_OFFSET

    position_control(
        q_d,
        v_d,
        joint_node_pub,
        joint_node_sub,
        error_tol=0.017,
        time_limit=10.0,
        grab=grab1,
        state_cb=state_cb,
        spin_lock=spin_lock,
    )
    # stone grab control
    if confirm_cb is None:
        input("Press Enter to execute grasp control...")
        confirm_result = None
    else:
        confirm_result = confirm_cb(q_d.copy(), grab1)
        if isinstance(confirm_result, str):
            if confirm_result == "abort":
                return
            if confirm_result == "skip_grasp_publish":
                q_d = project_angle(q_d[:6])
        elif confirm_result is not None:
            q_d = project_angle(np.asarray(confirm_result, dtype=np.float64)[:6])
    _fire_phase("grasping")
    if not (isinstance(confirm_result, str) and confirm_result == "skip_grasp_publish"):
        if mode == "pick":
            grasp_publish_count = GRASP_CLOSE_PUBLISH_COUNT
        else:
            grasp_publish_count = OPEN_AFTER_PLACE_PUBLISH_COUNT
        _publish_grab_signal(q_d, joint_node_pub, grab2, grasp_publish_count)
        time.sleep(GRASP_CONTROL_SETTLE_TIME)

    # lift up and back to the home configuration
    _fire_phase("lift")
    for q_t in path_lift:
        q_d = project_angle(q_t[:6])
        v_d = np.zeros_like(q_d)
        position_control(
            q_d,
            v_d,
            joint_node_pub,
            joint_node_sub,
            error_tol=0.1,
            grab=grab2,
            state_cb=state_cb,
            spin_lock=spin_lock,
        )
    position_control(
        q_d,
        v_d,
        joint_node_pub,
        joint_node_sub,
        error_tol=0.017,
        time_limit=5.0,
        grab=grab2,
        state_cb=state_cb,
        spin_lock=spin_lock,
    )
    _fire_phase("retreat")
    for q_t in path_home:
        q_d = project_angle(q_t[:6])
        v_d = np.zeros_like(q_d)
        position_control(
            q_d,
            v_d,
            joint_node_pub,
            joint_node_sub,
            error_tol=0.2,
            grab=grab2,
            state_cb=state_cb,
            spin_lock=spin_lock,
        )
    _fire_phase("done")
