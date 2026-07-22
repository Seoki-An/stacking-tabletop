import os
import sys
import time
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Tuple
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.clock import Clock
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
DIFFSIM_PY_BUILD = ROOT_DIR.parent / "diffsim" / "interop" / "python" / "build"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if DIFFSIM_PY_BUILD.exists() and str(DIFFSIM_PY_BUILD) not in sys.path:
    sys.path.insert(0, str(DIFFSIM_PY_BUILD))

from model import get_excavator_model

from ros2.joint_node import (
    CtrlJointPublisher,
    CtrlJointSubscriber,
    OpeningAnglePublisher,
)
from ros2.poseid_node import SCENE_SCAN_Q_TAIL, PoseIdentifier
from ros2.pose_node import PosePublisher, PoseArraySubscriber
from ros2.phase_node import PhaseSubscriber
from ros2.grasp_status_node import GraspStatusPublisher
from ros2.field_recovery_status_node import FieldRecoveryStatusPublisher
from ros2.log_dir_node import LogDirSubscriber
from ros2.scene_scan_node import (
    DiagnosticPcdRequestSubscriber,
    SceneICPResultPublisher,
    SceneScanDonePublisher,
)
from ros2.sceneid_node import SceneIdentifierPublisher
from ros2.control import position_control, project_angle
from tp_msgs.msg import Phase

from planning import Q_HOME, Q_SCAN_INHAND
from utils.log_paths import NUC_LOG_SUFFIX, with_log_machine_suffix

N_ROTS = 10
DURATION_TIME = 2.0  # (s)
FIELD_SCAN_DWELL = 5.0  # (s)
SCENE_SCAN_DWELL = 5.0  # (s)
FIELD_SCAN_END_TAIL = Q_HOME[1:]

SAVE_DATA = False
# Lidar1 and lidar2 see the in-hand stone/gripper.  Lidar3 points must be
# excluded from the in-hand scan because the arm-mounted frame does not match
# the gripper-frame scan assumption below.
INHAND_PCD_TOPICS = ("/iv_points21",)
SCENE_PCD_TOPICS = ("/iv_points22", "/iv_points21", "/iv_points20")
SCENE_PCD_TOPIC = "/scene_pcd"
SCENE_PCD_PUBLISH_VOXEL = 0.005
LIDAR_LINK_BY_TOPIC = {
    "/iv_points22": "lidar1_link",
    "/iv_points21": "lidar2_link",
    "/iv_points20": "lidar3_link",
}
# Per-rotation PCD cap.  We want a small set of PCDs at different joint
# angles for ICP, not many near-duplicates at the same dwell pose.
MAX_PCDS_PER_ROT = 2
# After a PCD is accepted, wait at most this long for a fresh joint
# feedback message so the (pcd, q) pair is temporally consistent.
JOINT_FRESHNESS_WINDOW = 0.2  # (s)

VISUALIZATION_ON = os.environ.get("POSE_IDENTIFICATION_VISUALIZATION", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

BUCKET_ANGLE_CHANGE_ON = True
BUCKET_SCAN_ANGLE_OFFSET = -np.pi / 6
INHAND_SCAN_SETTLE_TIME = 2.0  # seconds
INHAND_SCAN_SETTLE_TOL = 0.01
INHAND_SCAN_SETTLE_TIMEOUT = 10.0

FIELD_SCAN_SWING_VARIATION = np.pi / 12
FIELD_SCAN_SWING_NUM = 3

SCENE_SCAN_SWING_VARIATION = np.pi / 4
SCENE_SCAN_SWING_NUM = 5

INIT_POSE_WAIT_TIMEOUT = 10.0  # seconds
DIAGNOSTIC_PCD_WAIT_TIMEOUT = 15.0  # seconds
DIAGNOSTIC_PCD_DRAIN_TIME = 1.0  # seconds


def safe_log_token(value: str, fallback: str) -> str:
    token = "".join(
        ch if ch.isalnum() or ch in {"_", "-", "."} else "_"
        for ch in str(value).strip()
    )
    return token or fallback


def safe_topic_name(topic: str) -> str:
    return topic.strip("/").replace("/", "_") or "pcd"


def publish_points_as_pcd(
    scene_pcd_pub: SceneIdentifierPublisher | None,
    points: np.ndarray,
    frame_id: str,
) -> None:
    if scene_pcd_pub is None:
        return
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    scene_pcd_pub.publish(pcd, frame_id=frame_id)


def publish_inhand_scan_pcds(
    samples: List[Tuple[int, str, np.ndarray, np.ndarray]],
    gripper_pcds: List[np.ndarray],
    scene_pcd_pub: SceneIdentifierPublisher | None,
) -> None:
    if scene_pcd_pub is None:
        return
    for idx, (sample, pcd) in enumerate(zip(samples, gripper_pcds)):
        _, topic, _, _ = sample
        publish_points_as_pcd(
            scene_pcd_pub,
            pcd,
            f"inhand_gripper_{safe_topic_name(topic)}_{idx}",
        )
        time.sleep(0.01)


def scan_inhand_pcd(
    q: np.ndarray,
    q_rot_list: List[float],
    duration_time: float,
    joint_node_pub: CtrlJointPublisher,
    joint_node_sub: CtrlJointSubscriber,
    pose_identifier: PoseIdentifier,
) -> List[Tuple[int, str, np.ndarray, np.ndarray]]:
    data = []
    for q_rot in q_rot_list:
        q[5] = q_rot
        position_control(q, v, joint_node_pub, joint_node_sub)
        time.sleep(0.5)
        # The arm is now dwelling at the rotation target.  Discard any
        # stale flags from before the dwell (PCDs captured mid-motion or
        # joint feedback from the previous waypoint) so the (pcd, q) pair
        # we capture below actually corresponds to this rotation.
        for topic in INHAND_PCD_TOPICS:
            pose_identifier.reset_get_pcd_flag(topic)
        joint_node_sub.reset_get_flag()
        start_time = time.time()
        collected_per_topic = {topic: 0 for topic in INHAND_PCD_TOPICS}
        while any(
            count < MAX_PCDS_PER_ROT for count in collected_per_topic.values()
        ) and (time.time() - start_time < duration_time):
            rclpy.spin_once(joint_node_sub, timeout_sec=0.0)
            rclpy.spin_once(pose_identifier, timeout_sec=0.05)
            ready_topics = [
                topic
                for topic, count in collected_per_topic.items()
                if count < MAX_PCDS_PER_ROT and pose_identifier.get_pcd_flag(topic)
            ]
            if not ready_topics:
                continue

            # Wait briefly for a joint feedback message that arrives
            # AFTER the PCD so the joint pose paired with it is current.
            # The arm is dwelling, so the value should already be stable;
            # this just guards against the case where joint feedback is
            # slightly stale (e.g. the previous waypoint's tail).
            joint_node_sub.reset_get_flag()
            wait_start = time.time()
            while (
                not joint_node_sub.get_flag()
                and time.time() - wait_start < JOINT_FRESHNESS_WINDOW
            ):
                rclpy.spin_once(joint_node_sub, timeout_sec=0.05)

            ts = Clock().now().nanoseconds
            q_pcd = joint_node_sub.pos.copy()
            for topic in ready_topics:
                data.append(
                    (
                        ts,
                        topic,
                        pose_identifier.points_per_topic[topic].copy(),
                        q_pcd,
                    )
                )
                pose_identifier.reset_get_pcd_flag(topic)
                collected_per_topic[topic] += 1
            joint_node_sub.reset_get_flag()

    return data


def to_gripper_frame(
    pcd: np.ndarray, q: np.ndarray, excavator_model, lidar_link: str
) -> np.ndarray:
    """Transform a lidar PCD into the gripper (grip_body geom) frame.

    Mirrors the per-PCD transform used downstream when collecting the
    multi-rotation scan so the gripper-frame crop used by
    PoseIdentifier.detect_grasp_failure matches what
    identify_inhand_pose_multiple would see.
    """
    excavator_model.SetState(np.concatenate([q, np.zeros(2)]))
    lidar_rot = excavator_model.GetLink(lidar_link).GetRotation()
    lidar_pos = excavator_model.GetLink(lidar_link).GetPosition()
    pcd = (pcd @ lidar_rot.T) + lidar_pos
    gripper_geom = excavator_model.GetLink("grip_body").geoms[0]
    gripper_rot = gripper_geom.GetRotation()
    gripper_pos = gripper_geom.GetPosition()
    return (pcd - gripper_pos) @ gripper_rot


def data_to_gripper_frame(
    topic: str, pcd: np.ndarray, q: np.ndarray, excavator_model
) -> np.ndarray:
    return to_gripper_frame(
        pcd,
        q,
        excavator_model,
        LIDAR_LINK_BY_TOPIC[topic],
    )


def scan_field_pcd(
    q: np.ndarray, duration_time: float, joint_node_pub: CtrlJointPublisher
):
    data = []
    start_time = time.time()
    while time.time() - start_time < duration_time:
        position_control(q, v, joint_node_pub)
        time.sleep(0.5)
        rclpy.spin_once(pose_identifier, timeout_sec=0.1)
        if pose_identifier.get_pcd_flag():
            ts = Clock().now().nanoseconds
            data.append((ts, pose_identifier.points.copy()))
            pose_identifier.reset_get_pcd_flag()

    return data


def settle_inhand_scan_pose(
    q_scan: np.ndarray,
    v: np.ndarray,
    joint_node_pub: CtrlJointPublisher,
    joint_node_sub: CtrlJointSubscriber,
    pose_identifier: PoseIdentifier,
):
    print("Moving to Q_SCAN_INHAND and waiting for convergence...")
    position_control(
        q_scan,
        v,
        joint_node_pub,
        joint_node_sub,
        error_tol=0.03,
        time_limit=10.0,
    )
    position_control(
        q_scan,
        v,
        joint_node_pub,
        joint_node_sub,
        error_tol=INHAND_SCAN_SETTLE_TOL,
        time_limit=INHAND_SCAN_SETTLE_TIMEOUT,
    )

    settle_start = time.time()
    while time.time() - settle_start < INHAND_SCAN_SETTLE_TIME:
        joint_node_pub.publish(q_scan, v)
        rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
        rclpy.spin_once(pose_identifier, timeout_sec=0.01)

    for topic in INHAND_PCD_TOPICS:
        pose_identifier.reset_get_pcd_flag(topic)
    joint_node_sub.reset_get_flag()
    print("Q_SCAN_INHAND convergence wait done")


def drain_initial_pose_messages(duration: float = 0.3):
    before = {
        "inhand": pose_identifier.get_inhand_pose_seq(),
        "field": pose_identifier.get_field_pose_seq(),
    }
    end_time = time.time() + duration
    while time.time() < end_time:
        rclpy.spin_once(pose_identifier, timeout_sec=0.0)
    after = {
        "inhand": pose_identifier.get_inhand_pose_seq(),
        "field": pose_identifier.get_field_pose_seq(),
    }
    pose_identifier.reset_get_pose_flag()
    return before, after


def wait_for_field_initial_pose(drain_state=None, label: str = "field scan") -> bool:
    if drain_state is None:
        field_before = pose_identifier.get_field_pose_seq()
    else:
        field_before = drain_state[0]["field"]
    start = time.time()

    while not pose_identifier.get_field_pose_flag():
        if (
            pose_identifier.field_pose_init is not None
            and pose_identifier.get_field_pose_seq() > field_before
        ):
            print(
                f"{label}: using /field_poseid_init received during " "pre-wait drain."
            )
            return True
        if time.time() - start > INIT_POSE_WAIT_TIMEOUT:
            print(
                f"{label}: timed out waiting for /field_poseid_init after "
                f"{INIT_POSE_WAIT_TIMEOUT:.1f}s; skipping this request."
            )
            return False
        rclpy.spin_once(pose_identifier, timeout_sec=0.1)
    return True


def wait_for_inhand_initial_poses(drain_state=None) -> bool:
    if drain_state is None:
        inhand_before = pose_identifier.get_inhand_pose_seq()
        field_before = pose_identifier.get_field_pose_seq()
    else:
        inhand_before = drain_state[0]["inhand"]
        field_before = drain_state[0]["field"]
    start = time.time()

    while (
        not pose_identifier.get_inhand_pose_flag()
        or not pose_identifier.get_field_pose_flag()
        or pose_identifier.inhand_id != pose_identifier.field_id
    ):
        cached_pair_is_current = (
            pose_identifier.inhand_pose_init is not None
            and pose_identifier.field_pose_init is not None
            and pose_identifier.get_inhand_pose_seq() > inhand_before
            and pose_identifier.get_field_pose_seq() > field_before
            and pose_identifier.inhand_id == pose_identifier.field_id
        )
        if cached_pair_is_current:
            print("In-hand scan: using init poses received during pre-wait drain.")
            return True
        if time.time() - start > INIT_POSE_WAIT_TIMEOUT:
            print(
                "In-hand scan: timed out waiting for matching "
                f"/inhand_poseid_init and /field_poseid_init after "
                f"{INIT_POSE_WAIT_TIMEOUT:.1f}s; skipping this request."
            )
            return False
        rclpy.spin_once(pose_identifier, timeout_sec=0.1)
    return True


def scene_initial_pose_dict() -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    for stone_id, (pos, quat) in scene_pose_init_sub.pose_id.items():
        try:
            sid = int(stone_id)
        except (TypeError, ValueError):
            continue
        pose = np.concatenate(
            [
                np.asarray(pos, dtype=np.float64).reshape(3),
                np.asarray(quat, dtype=np.float64).reshape(4),
            ]
        )
        if np.all(np.isfinite(pose)):
            poses[sid] = pose
    return poses


def wait_for_scene_initial_poses(timeout: float = INIT_POSE_WAIT_TIMEOUT):
    deadline = time.time() + timeout
    latest: dict[int, np.ndarray] = {}
    last_update = None
    scene_pose_init_sub.reset_get_flag()

    while time.time() < deadline:
        rclpy.spin_once(scene_pose_init_sub, timeout_sec=0.1)
        if scene_pose_init_sub.get_flag():
            poses = scene_initial_pose_dict()
            scene_pose_init_sub.reset_get_flag()
            if poses:
                latest = poses
                last_update = time.time()
        if latest and last_update is not None and time.time() - last_update > 0.3:
            print(
                "Scene scan: received /scene_poseid_init poses for stones "
                f"{sorted(latest)}"
            )
            return latest

    if latest:
        print(
            "Scene scan: using latest /scene_poseid_init poses for stones "
            f"{sorted(latest)}"
        )
        return latest
    print(
        f"Scene scan: timed out waiting for /scene_poseid_init after "
        f"{timeout:.1f}s."
    )
    return {}


def scan_scene_pcd(
    target_xy: np.ndarray,
    dwell_time: float,
    joint_node_pub: CtrlJointPublisher,
    joint_node_sub: CtrlJointSubscriber,
    pose_identifier: PoseIdentifier,
    excavator_model,
    log_dir: str,
    scene_pcd_pub: SceneIdentifierPublisher | None = None,
):
    os.makedirs(log_dir, exist_ok=True)
    scan_stats = {
        "num_sweeps": 0,
        "num_fresh_joint_timeouts": 0,
        "topic_timeouts": {},
        "topic_points": {},
        "merged_points": 0,
    }

    def save_array(name: str, value):
        np.save(os.path.join(log_dir, name), value)

    def save_pcd(name: str, pcd: o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(os.path.join(log_dir, name), pcd)

    def save_failure(reason: str):
        with open(os.path.join(log_dir, "scene_scan_status.txt"), "w") as f:
            f.write(f"failed: {reason}\n")
            for key, value in scan_stats.items():
                f.write(f"{key}: {value}\n")
        print(f"Scene scan failed: {reason}; stats={scan_stats}")
        return False, reason

    def safe_topic_name(topic: str) -> str:
        return topic.strip("/").replace("/", "_") or "pcd"

    def visualize_merged_scene_pcd(pcd: o3d.geometry.PointCloud):
        if not VISUALIZATION_ON:
            return

        debug_pcd = o3d.geometry.PointCloud(pcd)
        points = np.asarray(debug_pcd.points, dtype=np.float64)
        if points.size == 0:
            return

        z = points[:, 2]
        z_min = float(np.min(z))
        z_max = float(np.max(z))
        z_span = z_max - z_min
        if z_span > 1e-9:
            t = (z - z_min) / z_span
        else:
            t = np.zeros_like(z)
        colors = np.column_stack(
            [
                t,
                0.15 + 0.70 * (1.0 - np.abs(2.0 * t - 1.0)),
                1.0 - t,
            ]
        )
        debug_pcd.colors = o3d.utility.Vector3dVector(colors)

        print(
            "Visualizing merged scene PCD: "
            f"{len(points)} points, z=[{z_min:.4f}, {z_max:.4f}]"
        )
        o3d.visualization.draw_geometries(
            [debug_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)],
            window_name="merged scene PCD",
        )

    def current_swing_angle(timeout: float = 0.5) -> float:
        joint_node_sub.reset_get_flag()
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(joint_node_sub, timeout_sec=0.02)
            if joint_node_sub.get_flag():
                q = np.asarray(joint_node_sub.pos, dtype=np.float64).reshape(-1)
                joint_node_sub.reset_get_flag()
                if q.size >= 1 and np.isfinite(q[0]):
                    return float(project_angle(q[0]))

        q = np.asarray(
            getattr(joint_node_sub, "pos", Q_HOME), dtype=np.float64
        ).reshape(-1)
        if getattr(joint_node_sub, "last_msg", None) is not None:
            if q.size >= 1 and np.isfinite(q[0]):
                return float(project_angle(q[0]))

        print(
            "Scene scan safety staging could not read current swing; "
            f"falling back to Q_HOME swing {float(Q_HOME[0]):.6f} rad."
        )
        return float(project_angle(Q_HOME[0]))

    def move_to_scene_scan_safety_posture():
        q_swing = current_swing_angle()
        q_home_same_swing = np.asarray(Q_HOME, dtype=np.float64).copy()
        q_home_same_swing[0] = q_swing
        q_scene_tail_same_swing = np.concatenate([[q_swing], SCENE_SCAN_Q_TAIL])

        save_array("scene_scan_q_home_same_swing.npy", q_home_same_swing)
        save_array("scene_scan_q_tail_same_swing.npy", q_scene_tail_same_swing)
        print(
            "Scene scan safety staging: moving to q_home tail at current swing "
            f"{q_swing:.6f} rad before applying scene-scan tail."
        )
        position_control(
            q_home_same_swing,
            v,
            joint_node_pub,
            joint_node_sub,
            error_tol=0.10,
            time_limit=10.0,
        )
        print(
            "Scene scan safety staging: moving to scene-scan tail while keeping "
            f"swing {q_swing:.6f} rad."
        )
        position_control(
            q_scene_tail_same_swing,
            v,
            joint_node_pub,
            joint_node_sub,
            error_tol=0.05,
            time_limit=10.0,
        )

    move_to_scene_scan_safety_posture()

    q0 = float(np.arctan2(target_xy[1], target_xy[0]))
    q0_offsets = (
        np.linspace(
            -SCENE_SCAN_SWING_VARIATION,
            SCENE_SCAN_SWING_VARIATION,
            SCENE_SCAN_SWING_NUM,
        )
        if SCENE_SCAN_SWING_NUM > 1
        else [0.0]
    )
    save_array("scene_scan_xy.npy", target_xy)

    scene_pcd = o3d.geometry.PointCloud()
    q_actual = None
    for swing_idx, offset in enumerate(q0_offsets):
        scan_stats["num_sweeps"] += 1
        q_scan = np.concatenate([[q0 + offset], SCENE_SCAN_Q_TAIL])
        save_array(f"scene_scan_q_scan_swing{swing_idx}.npy", q_scan)
        print(
            f"Scene scan (swing {swing_idx + 1}/{len(q0_offsets)}): moving to "
            f"q={q_scan.tolist()} for target xy {target_xy.tolist()}."
        )
        position_control(
            q_scan,
            v,
            joint_node_pub,
            joint_node_sub,
            error_tol=0.05,
            time_limit=15.0,
        )
        time.sleep(dwell_time)

        pose_identifier.reset_get_pcd_flag()
        joint_node_sub.reset_get_flag()
        start = time.time()
        while time.time() - start < 0.3:
            rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
            rclpy.spin_once(pose_identifier, timeout_sec=0.1)

        q_actual = None
        wait_start = time.time()
        while q_actual is None:
            if joint_node_sub.get_flag():
                q_actual = joint_node_sub.pos.copy()
                break
            if time.time() - wait_start > 5.0:
                q_actual = joint_node_sub.pos.copy()
                scan_stats["num_fresh_joint_timeouts"] += 1
                print(
                    "Scene scan warning: timed out waiting for fresh joint "
                    "feedback; using the last received joint state."
                )
                break
            rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
            joint_node_pub.publish(q_scan, v)

        for topic in SCENE_PCD_TOPICS:
            wait_start = time.time()
            while not pose_identifier.get_pcd_flag(topic):
                if time.time() - wait_start > max(5.0, dwell_time + 2.0):
                    scan_stats["topic_timeouts"][topic] = (
                        scan_stats["topic_timeouts"].get(topic, 0) + 1
                    )
                    print(
                        "Scene scan warning: timed out waiting for "
                        f"point cloud topic {topic}."
                    )
                    break
                rclpy.spin_once(pose_identifier, timeout_sec=0.1)
                joint_node_pub.publish(q_scan, v)

        save_array(f"scene_scan_q_actual_swing{swing_idx}.npy", q_actual)
        excavator_model.SetState(np.concatenate([q_actual, np.zeros(2)]))

        for topic in SCENE_PCD_TOPICS:
            if not pose_identifier.get_pcd_flag(topic):
                continue
            pts = pose_identifier.points_per_topic[topic].copy()
            scan_stats["topic_points"][topic] = scan_stats["topic_points"].get(
                topic, 0
            ) + int(len(pts))
            save_array(f"scene_raw_{safe_topic_name(topic)}_swing{swing_idx}.npy", pts)
            if pts.size == 0:
                continue
            link_name = LIDAR_LINK_BY_TOPIC[topic]
            lidar_frame = np.eye(4)
            lidar_frame[:3, :3] = excavator_model.GetLink(link_name).GetRotation()
            lidar_frame[:3, -1] = excavator_model.GetLink(link_name).GetPosition()
            save_array(
                f"scene_lidar_frame_{safe_topic_name(topic)}_swing{swing_idx}.npy",
                lidar_frame,
            )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.transform(lidar_frame)
            base_frame_name = f"scene_base_{safe_topic_name(topic)}_swing{swing_idx}"
            save_pcd(f"{base_frame_name}.ply", pcd)
            if scene_pcd_pub is not None:
                scene_pcd_pub.publish(pcd, frame_id=base_frame_name)
            scene_pcd += pcd

    scan_stats["merged_points"] = int(len(scene_pcd.points))
    if scene_pcd.is_empty():
        save_pcd("scene_scan_merged.ply", scene_pcd)
        return save_failure("merged_scene_pcd_empty")

    save_pcd("scene_scan_merged.ply", scene_pcd)
    visualize_merged_scene_pcd(scene_pcd)
    with open(os.path.join(log_dir, "scene_scan_status.txt"), "w") as f:
        f.write("succeeded\n")
        for key, value in scan_stats.items():
            f.write(f"{key}: {value}\n")
    q_after_scan = q_actual.copy()
    q_after_scan[1:] = FIELD_SCAN_END_TAIL
    position_control(
        q_after_scan,
        v,
        joint_node_pub,
        joint_node_sub,
        error_tol=0.05,
        time_limit=10.0,
    )
    print(f"Scene scan done. stats={scan_stats}")
    return True, ""


def scan_diagnostic_pcd(
    label: str,
    joint_node_sub: CtrlJointSubscriber,
    pose_identifier: PoseIdentifier,
    excavator_model,
    log_dir: str,
    scene_pcd_pub: SceneIdentifierPublisher | None = None,
):
    os.makedirs(log_dir, exist_ok=True)
    label = safe_log_token(label, "subgoal")
    topics = SCENE_PCD_TOPICS
    scan_stats = {
        "label": label,
        "topic_timeouts": {},
        "topic_points": {},
        "merged_points": 0,
        "fresh_joint_timeout": False,
    }

    def save_array(name: str, value):
        np.save(os.path.join(log_dir, name), value)

    def save_pcd(name: str, pcd: o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(os.path.join(log_dir, name), pcd)

    def save_status(status: str, reason: str = ""):
        with open(os.path.join(log_dir, "diagnostic_pcd_status.txt"), "w") as f:
            f.write(status + "\n")
            if reason:
                f.write(f"reason: {reason}\n")
            for key, value in scan_stats.items():
                f.write(f"{key}: {value}\n")

    def safe_topic_name(topic: str) -> str:
        return topic.strip("/").replace("/", "_") or "pcd"

    for topic in topics:
        pose_identifier.reset_get_pcd_flag(topic)
    joint_node_sub.reset_get_flag()

    start = time.time()
    q_actual = None
    while time.time() - start < DIAGNOSTIC_PCD_WAIT_TIMEOUT:
        rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
        rclpy.spin_once(pose_identifier, timeout_sec=0.05)
        if q_actual is None and joint_node_sub.get_flag():
            q_actual = joint_node_sub.pos.copy()
        if q_actual is not None and all(
            pose_identifier.get_pcd_flag(topic) for topic in topics
        ):
            break

    drain_end = time.time() + DIAGNOSTIC_PCD_DRAIN_TIME
    while time.time() < drain_end:
        rclpy.spin_once(joint_node_sub, timeout_sec=0.0)
        rclpy.spin_once(pose_identifier, timeout_sec=0.05)
        if joint_node_sub.get_flag():
            q_actual = joint_node_sub.pos.copy()

    if q_actual is None:
        q_actual = joint_node_sub.pos.copy()
        scan_stats["fresh_joint_timeout"] = True
        print(
            "Diagnostic PCD warning: timed out waiting for fresh joint feedback; "
            "using the last received joint state."
        )
    save_array("diagnostic_q_actual.npy", q_actual)

    scene_pcd = o3d.geometry.PointCloud()
    excavator_model.SetState(np.concatenate([q_actual, np.zeros(2)]))
    for topic in topics:
        if not pose_identifier.get_pcd_flag(topic):
            scan_stats["topic_timeouts"][topic] = 1
            print(f"Diagnostic PCD warning: no point cloud received from {topic}.")
            continue
        pts = pose_identifier.points_per_topic[topic].copy()
        scan_stats["topic_points"][topic] = int(len(pts))
        topic_name = safe_topic_name(topic)
        save_array(f"diagnostic_raw_{topic_name}.npy", pts)
        if pts.size == 0:
            continue

        link_name = LIDAR_LINK_BY_TOPIC[topic]
        lidar_frame = np.eye(4)
        lidar_frame[:3, :3] = excavator_model.GetLink(link_name).GetRotation()
        lidar_frame[:3, -1] = excavator_model.GetLink(link_name).GetPosition()
        save_array(f"diagnostic_lidar_frame_{topic_name}.npy", lidar_frame)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.transform(lidar_frame)
        frame_name = f"diagnostic_{label}_{topic_name}"
        save_pcd(f"{frame_name}.ply", pcd)
        if scene_pcd_pub is not None:
            scene_pcd_pub.publish(pcd, frame_id=frame_name)
        scene_pcd += pcd

    scan_stats["merged_points"] = int(len(scene_pcd.points))
    save_pcd("diagnostic_pcd_merged.ply", scene_pcd)
    if scene_pcd.is_empty():
        save_status("failed", "merged_diagnostic_pcd_empty")
        print(f"Diagnostic PCD failed: merged cloud is empty; stats={scan_stats}")
        return False, "merged_diagnostic_pcd_empty"

    save_status("succeeded")
    print(f"Diagnostic PCD done. stats={scan_stats}")
    return True, ""


if __name__ == "__main__":

    n_rots = N_ROTS
    duration_time = DURATION_TIME  # (s)
    drot = 2 * np.pi / n_rots
    q_scan = Q_SCAN_INHAND.copy()
    v = np.zeros(6)
    q_rot_list = []
    for i in range(n_rots):
        q_rot = q_scan[5] + drot * i
        if (
            abs(project_angle(q_rot)) < np.pi / 4
            or abs(project_angle(q_rot)) > 3 * np.pi / 4
        ):
            q_rot_list.append(q_rot)

    cfg = OmegaConf.load("agent/configs/config.yml")
    excavator_model, _ = get_excavator_model()

    #######################################################################

    rclpy.init()
    joint_node_pub = CtrlJointPublisher()
    joint_node_sub = CtrlJointSubscriber()
    phase_node_sub = PhaseSubscriber()
    pose_node_pub = PosePublisher(
        name="inhand_poseid_opt_publisher", topic="/inhand_poseid_opt"
    )
    opening_angle_opt_pub = OpeningAnglePublisher(
        name="inhand_opening_angle_opt_publisher",
        topic="/inhand_opening_angle_opt",
    )
    field_recovery_pub = PosePublisher(
        name="field_poseid_recovery_publisher", topic="/field_poseid_recovery"
    )
    scene_pose_init_sub = PoseArraySubscriber(
        name="scene_poseid_init_subscriber", topic="/scene_poseid_init"
    )
    scene_scan_done_pub = SceneScanDonePublisher(
        name="scene_scan_done_publisher", topic="/scene_scan_done"
    )
    diagnostic_pcd_request_sub = DiagnosticPcdRequestSubscriber(
        name="diagnostic_pcd_request_subscriber_poseid",
        topic="/diagnostic_pcd_request",
    )
    diagnostic_pcd_done_pub = SceneScanDonePublisher(
        name="diagnostic_pcd_done_publisher",
        topic="/diagnostic_pcd_done",
    )
    scene_icp_result_pub = SceneICPResultPublisher(
        name="scene_icp_result_publisher", topic="/scene_icp_result"
    )
    scene_pcd_pub = SceneIdentifierPublisher(
        name="scene_pcd_publisher",
        topic=SCENE_PCD_TOPIC,
        voxel=SCENE_PCD_PUBLISH_VOXEL,
    )
    grasp_status_pub = GraspStatusPublisher(
        name="grasp_status_publisher", topic="/grasp_status"
    )
    field_recovery_status_pub = FieldRecoveryStatusPublisher(
        name="field_recovery_status_publisher", topic="/field_recovery_status"
    )
    pose_identifier = PoseIdentifier(
        pcd_topics=list(dict.fromkeys([*INHAND_PCD_TOPICS, *SCENE_PCD_TOPICS])),
        inhand_pose_topic="/inhand_poseid_init",
        field_pose_topic="/field_poseid_init",
        stone_model_dir=cfg.environment.data.load_dir,
    )
    log_dir_sub = LogDirSubscriber()

    def lidar_links_for_active_pcd_topics():
        links = []
        missing = []
        for topic in pose_identifier.pcd_topics:
            link = LIDAR_LINK_BY_TOPIC.get(topic)
            if link is None:
                missing.append(topic)
                continue
            links.append(link)
        if missing:
            raise KeyError(f"Missing LiDAR link mapping for PCD topic(s): {missing}")
        return links

    def publish_field_recovery(target_id, step_log_dir):
        field_log_dir = os.path.join(step_log_dir, "field_scan")
        for _ in range(10):
            field_recovery_status_pub.publish("started", target_id)
        active_lidar_links = lidar_links_for_active_pcd_topics()
        print(
            "Field re-scan using PCD topics and LiDAR links: "
            f"{list(zip(pose_identifier.pcd_topics, active_lidar_links))}"
        )
        recovered = pose_identifier.recover_field_pose(
            target_id=target_id,
            dwell_time=FIELD_SCAN_DWELL,
            joint_node_pub=joint_node_pub,
            joint_node_sub=joint_node_sub,
            excavator_model=excavator_model,
            lidar_links=active_lidar_links,
            log_dir=field_log_dir,
            scene_pcd_pub=scene_pcd_pub,
            swing_variation=FIELD_SCAN_SWING_VARIATION,
            swing_num=FIELD_SCAN_SWING_NUM,
        )

        if recovered is None:
            if pose_identifier.field_pose_init is None:
                print(
                    "Field re-scan failed and no field_pose_init was "
                    "received from the desktop. Skipping recovery publish."
                )
                for _ in range(10):
                    field_recovery_status_pub.publish(
                        "failed", target_id, "no_field_pose_init"
                    )
                return False
            print(
                "Field re-scan failed; falling back to the global pose "
                "guess from the desktop so it can decide what to do next."
            )
            pos_opt, quat_opt = pose_identifier.field_pose_init
            for _ in range(10):
                field_recovery_status_pub.publish(
                    "fallback", target_id, "using_field_pose_init"
                )
        else:
            pos_opt, quat_opt = recovered
            for _ in range(10):
                field_recovery_status_pub.publish("succeeded", target_id)

        for _ in range(1000):
            field_recovery_pub.publish(pos_opt, quat_opt, target_id)

        q_after_scan = joint_node_sub.pos.copy()
        q_after_scan[1:] = FIELD_SCAN_END_TAIL
        print(
            "Field re-scan done; keeping adaptive swing and fixing other joints: "
            f"{q_after_scan.tolist()}"
        )
        position_control(
            q_after_scan,
            v,
            joint_node_pub,
            joint_node_sub,
            error_tol=0.05,
            time_limit=10.0,
        )
        return True

    def drain_phase_messages(duration: float = 0.5):
        end_time = time.time() + duration
        while time.time() < end_time:
            rclpy.spin_once(phase_node_sub, timeout_sec=0.0)
        phase_node_sub.reset_get_flag()

    def wait_for_log_dir(timeout: float = 10.0) -> str | None:
        start = time.time()
        while True:
            if log_dir_sub.get_flag() and log_dir_sub.path:
                step_log_dir_ = log_dir_sub.path
                log_dir_sub.reset_get_flag()
                return step_log_dir_
            if time.time() - start > timeout:
                if log_dir_sub.path:
                    print(
                        "Timed out waiting for a fresh /log_dir; using latest "
                        f"cached path: {log_dir_sub.path}"
                    )
                    return log_dir_sub.path
                print(
                    f"Timed out waiting for /log_dir after {timeout:.1f}s; "
                    "skipping this request."
                )
                return None
            rclpy.spin_once(log_dir_sub, timeout_sec=0.1)

    def publish_scene_scan_done(step_log_dir: str) -> None:
        for _ in range(10):
            scene_scan_done_pub.publish(step_log_dir)
            time.sleep(0.02)
        print(f"Scene scan PCD transfer complete for {step_log_dir}.")

    def publish_diagnostic_pcd_done(log_dir: str) -> None:
        for _ in range(10):
            diagnostic_pcd_done_pub.publish(log_dir)
            time.sleep(0.02)
        print(f"Diagnostic PCD transfer complete for {log_dir}.")

    def drain_duplicate_diagnostic_requests(log_dir: str, label: str) -> int:
        drained = 0
        deadline = time.time() + 0.3
        while time.time() < deadline:
            rclpy.spin_once(diagnostic_pcd_request_sub, timeout_sec=0.02)
            if not diagnostic_pcd_request_sub.get_flag():
                continue
            if (
                diagnostic_pcd_request_sub.log_dir == log_dir
                and diagnostic_pcd_request_sub.label == label
            ):
                drained += 1
                diagnostic_pcd_request_sub.reset_get_flag()
                continue
            break
        if drained:
            print(
                "Drained duplicate diagnostic PCD request(s): "
                f"{drained} for label={label}, log_dir={log_dir}"
            )
        return drained

    done = False
    while not done:
        phase_node_sub.reset_get_flag()
        while True:
            rclpy.spin_once(diagnostic_pcd_request_sub, timeout_sec=0.0)
            if diagnostic_pcd_request_sub.get_flag():
                break
            rclpy.spin_once(phase_node_sub, timeout_sec=0.1)
            if phase_node_sub.get_flag() and phase_node_sub.phase in (
                Phase.INHANDSCAN,
                Phase.FIELDSCAN,
                Phase.SCENESCAN,
            ):
                break

        if diagnostic_pcd_request_sub.get_flag():
            diagnostic_request_log_dir = diagnostic_pcd_request_sub.log_dir
            diagnostic_log_dir = with_log_machine_suffix(
                diagnostic_request_log_dir, NUC_LOG_SUFFIX
            )
            diagnostic_label = diagnostic_pcd_request_sub.label
            diagnostic_pcd_request_sub.reset_get_flag()
            drain_duplicate_diagnostic_requests(
                diagnostic_request_log_dir,
                diagnostic_label,
            )
            print(
                "Received diagnostic PCD request: "
                f"label={diagnostic_label}, requested_log_dir={diagnostic_request_log_dir}, "
                f"nuc_log_dir={diagnostic_log_dir}"
            )
            scan_ok, scan_reason = scan_diagnostic_pcd(
                diagnostic_label,
                joint_node_sub,
                pose_identifier,
                excavator_model,
                diagnostic_log_dir,
                scene_pcd_pub,
            )
            if not scan_ok:
                scene_icp_result_pub.publish(
                    diagnostic_request_log_dir,
                    "failed",
                    f"diagnostic_pcd:{scan_reason}",
                )
            publish_diagnostic_pcd_done(diagnostic_request_log_dir)
            drain_phase_messages()
            continue

        requested_phase = phase_node_sub.phase
        phase_node_sub.reset_get_flag()
        if requested_phase == Phase.FIELDSCAN:
            print("Received FIELDSCAN request.")
        elif requested_phase == Phase.SCENESCAN:
            print("Received SCENESCAN request.")
        else:
            print("Received INHANDSCAN request.")

        requested_step_log_dir = wait_for_log_dir()
        if requested_step_log_dir is None:
            drain_phase_messages()
            continue
        step_log_dir = with_log_machine_suffix(requested_step_log_dir, NUC_LOG_SUFFIX)
        print(f"Log directory for this step: {step_log_dir}")
        if step_log_dir != requested_step_log_dir:
            print(f"Desktop-requested log directory: {requested_step_log_dir}")

        # The desktop publishes init poses in short bursts, and RELIABLE queues
        # can retain old /inhand_poseid_init or /field_poseid_init messages.
        # Drain them here so the following wait receives the current request.
        # SCENESCAN sends its target xy in the same burst as the phase, so
        # draining here would consume the current target.
        drain_state = None
        if requested_phase != Phase.SCENESCAN:
            drain_state = drain_initial_pose_messages()

        print("Waiting initial pose...")

        pose_identifier.reset_get_pose_flag()
        if requested_phase == Phase.FIELDSCAN:
            if not wait_for_field_initial_pose(drain_state, label="Field re-scan"):
                drain_phase_messages()
                continue
            pose_identifier.reset_get_pose_flag()
            print(f"Initial field/recovery pose: {pose_identifier.field_pose_init}")
            target_id = pose_identifier.field_id
            publish_field_recovery(target_id, step_log_dir)
            drain_phase_messages()
            continue

        if requested_phase == Phase.SCENESCAN:
            if not wait_for_field_initial_pose(label="Scene scan"):
                drain_phase_messages()
                continue
            target_xy = pose_identifier.field_pose_init[0][:2]
            scene_initial_poses_by_stone = wait_for_scene_initial_poses()
            if not scene_initial_poses_by_stone:
                scene_icp_result_pub.publish(
                    requested_step_log_dir,
                    "failed",
                    "missing_scene_poseid_init",
                )
                drain_phase_messages()
                continue
            pose_identifier.reset_get_pose_flag()
            print(f"Scene scan xy: {target_xy}")
            scan_ok, scan_reason = scan_scene_pcd(
                target_xy,
                SCENE_SCAN_DWELL,
                joint_node_pub,
                joint_node_sub,
                pose_identifier,
                excavator_model,
                os.path.join(step_log_dir, "scene_scan"),
                scene_pcd_pub,
            )
            if not scan_ok:
                scene_icp_result_pub.publish(
                    requested_step_log_dir,
                    "failed",
                    f"scene_scan:{scan_reason}",
                )
                publish_scene_scan_done(requested_step_log_dir)
                drain_phase_messages()
                continue
            publish_scene_scan_done(requested_step_log_dir)
            drain_phase_messages()
            continue

        if not wait_for_inhand_initial_poses(drain_state):
            drain_phase_messages()
            continue
        pose_identifier.reset_get_pose_flag()
        print(f"Initial in-hand pose: {pose_identifier.inhand_pose_init}")
        print(f"Initial field/recovery pose: {pose_identifier.field_pose_init}")

        q = q_scan.copy()
        settle_inhand_scan_pose(
            q_scan,
            v,
            joint_node_pub,
            joint_node_sub,
            pose_identifier,
        )
        print("Initialization done")

        # Probe a single rotation first to detect grasp failure cheaply,
        # before committing to the full multi-rotation scan sweep.
        probe_data = []
        while len(probe_data) == 0:
            probe_data = scan_inhand_pcd(
                q,
                q_rot_list[:1],
                duration_time,
                joint_node_pub,
                joint_node_sub,
                pose_identifier,
            )
            print(f"Probe collected {len(probe_data)} in-hand point clouds")

        probe_gripper_pcds = [
            data_to_gripper_frame(topic, pcd, q_pcd, excavator_model)
            for _, topic, pcd, q_pcd in probe_data
        ]
        probe_pcd_gripper = np.concatenate(probe_gripper_pcds, axis=0)
        grasp_failed = pose_identifier.detect_grasp_failure(probe_pcd_gripper)

        if grasp_failed:
            data = probe_data
            pose_opt = None
        else:
            remaining_data = []
            if len(q_rot_list) > 1:
                remaining_data = scan_inhand_pcd(
                    q,
                    q_rot_list[1:],
                    duration_time,
                    joint_node_pub,
                    joint_node_sub,
                    pose_identifier,
                )
                if BUCKET_ANGLE_CHANGE_ON:
                    q[3] += BUCKET_SCAN_ANGLE_OFFSET
                    remaining_data_ = scan_inhand_pcd(
                        q,
                        q_rot_list,
                        duration_time,
                        joint_node_pub,
                        joint_node_sub,
                        pose_identifier,
                    )
                    remaining_data.extend(remaining_data_)
                print(
                    f"Collected {len(remaining_data)} additional in-hand point clouds"
                )
                position_control(
                    Q_HOME,
                    v,
                    joint_node_pub,
                    joint_node_sub,
                    error_tol=0.2,
                    time_limit=5.0,
                )
            data = probe_data + remaining_data

        pcd_list = []
        q_list = []
        print("The number of point clouds collected: ", len(data))
        for ts, topic, pcd, q_pcd in data:
            # Use the visual-geom frame to match the rest of the codebase
            # (sim places the held stone in this frame, desktop computes
            # inhand_T in this frame; reading the link frame here would
            # introduce a constant offset and the cropped PCD would miss
            # the stone entirely).
            pcd_list.append(data_to_gripper_frame(topic, pcd, q_pcd, excavator_model))
            q_list.append(q_pcd)

        publish_inhand_scan_pcds(data, pcd_list, scene_pcd_pub)

        if not grasp_failed:
            pose_opt = pose_identifier.identify_inhand_pose_multiple(
                pcd_list,
                pose_identifier.inhand_pose_init,
                pose_identifier.inhand_id,
                VISUALIZATION_ON,
            )

        print("Optimized pose: ", pose_opt)
        if pose_opt is not None:
            opening_angle_opt = pose_identifier.last_opening_angle
            if opening_angle_opt is not None and np.isfinite(opening_angle_opt):
                opening_angle_opt = float(opening_angle_opt)
                print(
                    "Optimized in-hand opening angle: " f"{opening_angle_opt:.6f} rad"
                )
            else:
                opening_angle_opt = None
                print(
                    "Optimized in-hand opening angle unavailable; desktop will "
                    "fall back to the planned grasp opening angle."
                )
            for _ in range(10):
                grasp_status_pub.publish(True)
            for _ in range(1000):
                if opening_angle_opt is not None:
                    opening_angle_opt_pub.publish(opening_angle_opt)
                pose_node_pub.publish(
                    pose_opt[:3, -1],
                    Rotation.from_matrix(pose_opt[:3, :3]).as_quat(),
                    pose_identifier.inhand_id,
                )
            if SAVE_DATA:
                continue
            os.makedirs(step_log_dir, exist_ok=True)
            for i, pcd in enumerate(pcd_list):
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
                o3d.io.write_point_cloud(
                    os.path.join(step_log_dir, f"pcd_{i}.ply"), pcd_o3d
                )
            q_save = np.array(q_list)
            np.save(os.path.join(step_log_dir, "q_list.npy"), q_save)
            np.save(
                os.path.join(step_log_dir, "inhand_pose_init_pos.npy"),
                pose_identifier.inhand_pose_init[0],
            )
            np.save(
                os.path.join(step_log_dir, "inhand_pose_init_rot.npy"),
                pose_identifier.inhand_pose_init[1],
            )
            np.save(
                os.path.join(step_log_dir, "inhand_id.npy"),
                pose_identifier.inhand_id,
            )
            np.save(
                os.path.join(step_log_dir, "inhand_pose_opt_pos.npy"),
                pose_opt[:3, -1],
            )
            np.save(
                os.path.join(step_log_dir, "inhand_pose_opt_rot.npy"),
                Rotation.from_matrix(pose_opt[:3, :3]).as_quat(),
            )
            if opening_angle_opt is not None:
                np.save(
                    os.path.join(step_log_dir, "inhand_opening_angle_opt.npy"),
                    np.array(opening_angle_opt, dtype=np.float64),
                )
        else:
            print(
                "Sensing grasp failed. Detected opening angle below threshold. "
                "Starting field re-scan recovery."
            )
            target_id = pose_identifier.inhand_id

            for _ in range(10):
                grasp_status_pub.publish(False)

            publish_field_recovery(target_id, step_log_dir)
        drain_phase_messages()
