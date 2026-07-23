#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d

import rclpy
from rclpy.clock import Clock

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ros2.joint_node import CtrlJointPublisher, CtrlJointSubscriber
from ros2.pcd_node import PointCloudSubscriber
from ros2.control import position_control

from planning import Q_SCAN

LIDAR2_TOPIC = "/iv_points21"
LIDAR2_LINK = "lidar2_link"
DEFAULT_BUCKET_OFFSETS = (0.0, 0.6, -0.9)
DEFAULT_TILT_OFFSETS = (0.0, 0.2, -0.2)
DEFAULT_N_ROTS = 10
DEFAULT_DURATION_TIME = 2.0
DEFAULT_ROTATION_SETTLE_TIME = 1.0
DEFAULT_BUCKET_TILT_SETTLE_TIME = 5.0
DEFAULT_MAX_SAMPLES_PER_POSE = 5
GRAB_OPENING_ANGLE = {
    "open": 0.0,
    "close": -1.14,
    "stay": 0.0,
}


def _parse_float_list(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _resolve_output_dir(path: str) -> Path:
    save_root = Path(path)
    if not save_root.is_absolute():
        save_root = Path.cwd() / save_root
    return save_root


def _rotation_targets(q_init: np.ndarray, n_rots: int) -> list[float]:
    drot = 2 * np.pi / n_rots
    return [float(q_init[5] + drot * i) for i in range(n_rots)]


def _hold_pose(
    q: np.ndarray,
    v: np.ndarray,
    duration: float,
    joint_node_pub: CtrlJointPublisher,
    joint_node_sub: CtrlJointSubscriber,
    pcd_node_sub: PointCloudSubscriber,
    grab: str,
) -> None:
    start = time.time()
    while time.time() - start < duration:
        joint_node_pub.publish(q, v, grab=grab)
        rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
        rclpy.spin_once(pcd_node_sub, timeout_sec=0.01)


def scan_pcd(
    q: np.ndarray,
    v: np.ndarray,
    q_rot_list: Iterable[float],
    duration_time: float,
    rotation_settle_time: float,
    max_samples_per_pose: int,
    joint_node_pub: CtrlJointPublisher,
    joint_node_sub: CtrlJointSubscriber,
    pcd_node_sub: PointCloudSubscriber,
    grab: str,
    opening_angle: float,
    pose_label: str,
    bucket_offset: float,
    tilt_offset: float,
) -> list[dict]:
    data = []
    for rotation_index, q_rot in enumerate(q_rot_list):
        q = q.copy()
        q[5] = q_rot
        position_control(q, v, joint_node_pub, joint_node_sub, grab=grab)
        _hold_pose(
            q,
            v,
            rotation_settle_time,
            joint_node_pub,
            joint_node_sub,
            pcd_node_sub,
            grab=grab,
        )

        pcd_node_sub.reset_get_flag()
        joint_node_sub.reset_get_flag()
        start_time = time.time()
        samples_for_pose = 0
        while time.time() - start_time < duration_time:
            joint_node_pub.publish(q, v, grab=grab)
            rclpy.spin_once(pcd_node_sub, timeout_sec=0.1)
            rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
            if pcd_node_sub.get_flag() and joint_node_sub.get_flag():
                ts = Clock().now().nanoseconds
                data.append(
                    {
                        "time": int(ts),
                        "points": pcd_node_sub.points.copy(),
                        "joint": joint_node_sub.pos.copy(),
                        "command_joint": q.copy(),
                        "pose_label": pose_label,
                        "bucket_offset": float(bucket_offset),
                        "tilt_offset": float(tilt_offset),
                        "rotation_index": rotation_index,
                        "pcd_topic": LIDAR2_TOPIC,
                        "lidar_link": LIDAR2_LINK,
                        "grab": grab,
                        "opening_angle": float(opening_angle),
                    }
                )
                samples_for_pose += 1
                if (
                    max_samples_per_pose > 0
                    and samples_for_pose >= max_samples_per_pose
                ):
                    break
            pcd_node_sub.reset_get_flag()
            joint_node_sub.reset_get_flag()

    return data


def _write_scan(save_root: Path, data: list[dict], metadata: dict) -> None:
    pcd_dir = save_root / "pcd"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_root / "joint_log.csv"
    metadata_path = save_root / "metadata.json"

    fieldnames = (
        [
            "time",
            "pose_label",
            "bucket_offset",
            "tilt_offset",
            "rotation_index",
            "pcd_topic",
            "lidar_link",
            "grab",
            "opening_angle",
        ]
        + [f"joint_{i}" for i in range(6)]
        + [f"command_joint_{i}" for i in range(6)]
    )

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            csv_writer.writeheader()

        for sample in data:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sample["points"])
            pcd_filename = pcd_dir / f"{sample['time']}.pcd"
            o3d.io.write_point_cloud(str(pcd_filename), pcd)

            row = {
                "time": sample["time"],
                "pose_label": sample["pose_label"],
                "bucket_offset": sample["bucket_offset"],
                "tilt_offset": sample["tilt_offset"],
                "rotation_index": sample["rotation_index"],
                "pcd_topic": sample["pcd_topic"],
                "lidar_link": sample["lidar_link"],
                "grab": sample["grab"],
                "opening_angle": sample["opening_angle"],
            }
            row.update(
                {f"joint_{i}": float(value) for i, value in enumerate(sample["joint"])}
            )
            row.update(
                {
                    f"command_joint_{i}": float(value)
                    for i, value in enumerate(sample["command_joint"])
                }
            )
            csv_writer.writerow(row)

    metadata = dict(metadata)
    metadata["num_samples"] = len(data)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture LiDAR-2 point clouds of the manipulator gripper and log the "
            "corresponding six-joint feedback."
        )
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        required=True,
        help="Directory where pcd/ and joint_log.csv will be written.",
    )
    parser.add_argument(
        "--n-rots",
        type=int,
        default=DEFAULT_N_ROTS,
        help="Number of full-turn rotate-joint scan bins.",
    )
    parser.add_argument(
        "--bucket-offsets",
        type=_parse_float_list,
        default=list(DEFAULT_BUCKET_OFFSETS),
        help="Comma-separated bucket-joint offsets relative to the scan pose.",
    )
    parser.add_argument(
        "--tilt-offsets",
        type=_parse_float_list,
        default=list(DEFAULT_TILT_OFFSETS),
        help="Comma-separated tilt-joint offsets relative to the scan pose.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_TIME,
        help="Seconds to collect point clouds at each scan pose.",
    )
    parser.add_argument(
        "--rotation-settle-time",
        type=float,
        default=DEFAULT_ROTATION_SETTLE_TIME,
        help="Seconds to hold after each rotate-joint move before accepting clouds.",
    )
    parser.add_argument(
        "--bucket-tilt-settle-time",
        type=float,
        default=DEFAULT_BUCKET_TILT_SETTLE_TIME,
        help="Seconds to hold after each bucket/tilt offset move.",
    )
    parser.add_argument(
        "--max-samples-per-pose",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_POSE,
        help="Maximum PCD samples per pose; 0 keeps collecting for --duration.",
    )
    parser.add_argument(
        "--grab",
        choices=["open", "close", "stay"],
        default="open",
        help="Gripper command published while moving and scanning.",
    )
    parser.add_argument(
        "--opening-angle",
        type=float,
        default=None,
        help=(
            "Opening angle in radians to store for FK visualization. If omitted, "
            "a value is inferred from --grab."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    save_root = _resolve_output_dir(args.dir)
    save_root.mkdir(parents=True, exist_ok=True)

    if args.n_rots <= 0:
        raise ValueError("--n-rots must be positive")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.rotation_settle_time < 0.0:
        raise ValueError("--rotation-settle-time cannot be negative")
    if args.bucket_tilt_settle_time < 0.0:
        raise ValueError("--bucket-tilt-settle-time cannot be negative")

    q_init = Q_SCAN.copy()
    v = np.zeros(6)
    q_rot_list = _rotation_targets(q_init, args.n_rots)
    opening_angle = (
        GRAB_OPENING_ANGLE[args.grab]
        if args.opening_angle is None
        else float(args.opening_angle)
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pcd_topic": LIDAR2_TOPIC,
        "lidar_link": LIDAR2_LINK,
        "scan_pose": "q_scan",
        "q_init": q_init.tolist(),
        "q_rot_list": [float(q) for q in q_rot_list],
        "bucket_offsets": [float(x) for x in args.bucket_offsets],
        "tilt_offsets": [float(x) for x in args.tilt_offsets],
        "duration": float(args.duration),
        "rotation_settle_time": float(args.rotation_settle_time),
        "bucket_tilt_settle_time": float(args.bucket_tilt_settle_time),
        "max_samples_per_pose": int(args.max_samples_per_pose),
        "grab": args.grab,
        "opening_angle": float(opening_angle),
    }

    print("Gripper LiDAR-2 PCD scan starts...")
    print(f"Saving to: {save_root}")
    print(f"Rotation targets: {len(q_rot_list)}")
    print(f"Bucket offsets: {metadata['bucket_offsets']}")
    print(f"Tilt offsets: {metadata['tilt_offsets']}")
    print(f"Rotation settle time: {args.rotation_settle_time:.2f}s")
    print(f"Bucket/tilt settle time: {args.bucket_tilt_settle_time:.2f}s")
    print(f"Stored opening angle: {opening_angle:.4f} rad")

    rclpy.init()
    joint_node_pub = CtrlJointPublisher()
    joint_node_sub = CtrlJointSubscriber()
    pcd_node_sub = PointCloudSubscriber(topic=LIDAR2_TOPIC)

    try:
        print("To the initial position...")
        position_control(q_init, v, joint_node_pub, joint_node_sub, grab=args.grab)
        _hold_pose(
            q_init,
            v,
            max(args.bucket_tilt_settle_time, 1.0),
            joint_node_pub,
            joint_node_sub,
            pcd_node_sub,
            grab=args.grab,
        )
        print("Initialization done")

        data = []
        pose_index = 0
        previous_pose_q = None
        for bucket_offset in args.bucket_offsets:
            for tilt_offset in args.tilt_offsets:
                q = q_init.copy()
                q[3] += bucket_offset
                q[4] += tilt_offset

                if previous_pose_q is not None:
                    q_rotate_home = previous_pose_q.copy()
                    q_rotate_home[5] = q_init[5]
                    position_control(
                        q_rotate_home,
                        v,
                        joint_node_pub,
                        joint_node_sub,
                        grab=args.grab,
                    )
                    _hold_pose(
                        q_rotate_home,
                        v,
                        args.rotation_settle_time,
                        joint_node_pub,
                        joint_node_sub,
                        pcd_node_sub,
                        grab=args.grab,
                    )

                position_control(
                    q,
                    v,
                    joint_node_pub,
                    joint_node_sub,
                    grab=args.grab,
                )
                _hold_pose(
                    q,
                    v,
                    args.bucket_tilt_settle_time,
                    joint_node_pub,
                    joint_node_sub,
                    pcd_node_sub,
                    grab=args.grab,
                )

                pose_label = (
                    f"bucket_offset_{bucket_offset:+.3f}_"
                    f"tilt_offset_{tilt_offset:+.3f}"
                )
                print(f"Scanning {pose_label}...")
                data.extend(
                    scan_pcd(
                        q,
                        v,
                        q_rot_list,
                        args.duration,
                        args.rotation_settle_time,
                        args.max_samples_per_pose,
                        joint_node_pub,
                        joint_node_sub,
                        pcd_node_sub,
                        grab=args.grab,
                        opening_angle=opening_angle,
                        pose_label=pose_label,
                        bucket_offset=bucket_offset,
                        tilt_offset=tilt_offset,
                    )
                )
                previous_pose_q = q.copy()
                previous_pose_q[5] = q_rot_list[-1]
                pose_index += 1

        if pose_index == 0:
            print("No bucket/tilt offset combinations were requested.")
        else:
            print("Returning to the initial position...")
            try:
                q_rotate_home = previous_pose_q.copy()
                q_rotate_home[5] = q_init[5]
                position_control(
                    q_rotate_home,
                    v,
                    joint_node_pub,
                    joint_node_sub,
                    grab=args.grab,
                )
                _hold_pose(
                    q_rotate_home,
                    v,
                    args.rotation_settle_time,
                    joint_node_pub,
                    joint_node_sub,
                    pcd_node_sub,
                    grab=args.grab,
                )
                position_control(
                    q_init,
                    v,
                    joint_node_pub,
                    joint_node_sub,
                    grab=args.grab,
                )
                _hold_pose(
                    q_init,
                    v,
                    args.bucket_tilt_settle_time,
                    joint_node_pub,
                    joint_node_sub,
                    pcd_node_sub,
                    grab=args.grab,
                )
            except Exception as exc:
                print(f"Warning: failed to return to the initial position: {exc}")

        _write_scan(save_root, data, metadata)
        print(
            f"Saved {len(data)} samples to:\n"
            f" - {save_root / 'pcd'}\n"
            f" - {save_root / 'joint_log.csv'}\n"
            f" - {save_root / 'metadata.json'}"
        )
    finally:
        joint_node_pub.destroy_node()
        joint_node_sub.destroy_node()
        pcd_node_sub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
