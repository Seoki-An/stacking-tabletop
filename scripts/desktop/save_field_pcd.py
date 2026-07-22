#!/usr/bin/env python3
"""Desktop receiver for the field scan PCD streamed by scan_field_stone.py.

Subscribes to /field_pcd, merges the streamed base-frame frames into one
cloud, and saves it as field_scan.pcd under a unique .data/field_pcd/<date>
directory, mirroring the save layout of scripts/nuc/scan_field_stone.py.

The NUC publishes an explicit "done" on /field_pcd_done when the scan
finishes; that is the primary stop condition (with a short drain for in-flight
frames). A long idle timeout is the fallback if the NUC dies.

Run on the desktop:  python -m scripts.desktop.save_field_pcd
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ros2.sceneid_node import (
    SceneIdentifierSubscriber,
    SceneScanDoneSubscriber,
    JointLogSubscriber,
)
from utils import get_unique_dir

FIELD_PCD_TOPIC = "/field_pcd"
FIELD_PCD_DONE_TOPIC = "/field_pcd_done"
FIELD_JOINT_LOG_TOPIC = "/field_joint_log"
ROOT_SAVE_DIR = ".data/field_pcd"

# Primary stop is the NUC's done signal; drain briefly afterwards for stragglers.
DONE_DRAIN = 2.0  # (s)
# Fallback only: stop if frames go silent this long WITHOUT a done signal.
# Must exceed inter-pose arm-motion gaps, so it is deliberately generous.
IDLE_TIMEOUT = 30.0  # (s)
# Overall safety cap so we never block forever waiting for the first frame.
FIRST_FRAME_TIMEOUT = 300.0  # (s)


def _safe_tag(frame_id: str, fallback: str) -> str:
    """Sanitize a frame_id into a filename-safe tag."""
    tag = (frame_id or "").strip().strip("/").replace("/", "_")
    return tag or fallback


def _write_joint_log(save_dir, rows) -> None:
    """Write the joint log as scan_field_stone.py does (one row per pose)."""
    path = os.path.join(save_dir, "joint_log.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in np.asarray(rows):
            writer.writerow([float(x) for x in row])
    print(f"[desktop] saved joint_log.csv ({len(np.asarray(rows))} rows) to {path}")


def collect_and_save(field_sub, done_sub, joint_log_sub, save_dir):
    """Subscribe repeatedly until done, saving each frame by its frame_id tag.

    Each frame's header.frame_id identifies it (e.g. ``lidar1_3_raw`` or
    ``lidar1_3``), so it is saved as ``field_scan_<tag>.pcd``, mirroring
    scan_field_stone.py. Transformed (non-``_raw``) frames are also merged into
    the returned cloud, which the caller saves as field_scan.pcd.
    """
    merged = o3d.geometry.PointCloud()
    num_frames = 0

    print(f"[desktop] waiting for field PCD stream on {FIELD_PCD_TOPIC} ...")
    first_deadline = time.time() + FIRST_FRAME_TIMEOUT
    last_frame_time = None
    drain_deadline = None

    while True:
        rclpy.spin_once(field_sub, timeout_sec=0.1)
        rclpy.spin_once(done_sub, timeout_sec=0.0)
        rclpy.spin_once(joint_log_sub, timeout_sec=0.0)

        if joint_log_sub.get_flag() and joint_log_sub.rows is not None:
            _write_joint_log(save_dir, joint_log_sub.rows)
            joint_log_sub.reset_get_flag()

        if field_sub.get_flag():
            pts = field_sub.points.copy()
            frame_id = field_sub.last_msg.header.frame_id if field_sub.last_msg else ""
            field_sub.reset_get_flag()
            if pts.size:
                tag = _safe_tag(frame_id, fallback=f"field_scan_frame{num_frames}")
                frame = o3d.geometry.PointCloud()
                frame.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                o3d.io.write_point_cloud(
                    os.path.join(save_dir, f"{tag}.pcd"), frame
                )
                # Accumulate only the base-frame (transformed) clouds, like
                # scan_field_stone's scene_pcd.
                if not tag.endswith("_raw"):
                    merged += frame
                num_frames += 1
                last_frame_time = time.time()
                print(
                    f"[desktop] frame {num_frames} [{tag}]: +{len(pts)} pts, "
                    f"merged total {len(merged.points)}"
                )
            continue

        if done_sub.get_flag() and drain_deadline is None:
            print("[desktop] NUC signaled scan done; draining remaining frames...")
            drain_deadline = time.time() + DONE_DRAIN
        if drain_deadline is not None:
            if time.time() > drain_deadline:
                print("[desktop] drain complete; field scan finished.")
                break
            continue

        if last_frame_time is None:
            if time.time() > first_deadline:
                print("[desktop] timed out waiting for the first frame.")
                break
        elif time.time() - last_frame_time > IDLE_TIMEOUT:
            print(
                f"[desktop] stream idle {IDLE_TIMEOUT}s with no done signal; "
                "stopping."
            )
            break

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save into. Defaults to a unique "
            f"{ROOT_SAVE_DIR}/<date> dir, like scan_field_stone.py."
        ),
    )
    parser.add_argument(
        "--merge-voxel",
        type=float,
        default=0.0,
        help="Optional voxel size (m) to dedupe overlapping frames. 0 = off.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip the Open3D viewer (just save).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Choose the save directory up front so per-frame files can be written as
    # they arrive (same layout as scan_field_stone.py).
    if args.output_dir is not None:
        save_dir = str(args.output_dir)
    else:
        today_str = datetime.now().strftime("%y%m%d")
        save_dir = get_unique_dir(ROOT_SAVE_DIR, today_str)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[desktop] saving field scan PCDs to {save_dir}")

    rclpy.init()
    field_sub = SceneIdentifierSubscriber(
        name="field_pcd_subscriber", topic=FIELD_PCD_TOPIC
    )
    done_sub = SceneScanDoneSubscriber(
        name="field_pcd_done_subscriber", topic=FIELD_PCD_DONE_TOPIC
    )
    joint_log_sub = JointLogSubscriber(
        name="field_joint_log_subscriber", topic=FIELD_JOINT_LOG_TOPIC
    )

    try:
        merged = collect_and_save(field_sub, done_sub, joint_log_sub, save_dir)
    finally:
        rclpy.shutdown()

    if merged.is_empty():
        print("[desktop] no transformed frames received; no field_scan.pcd written.")
        return

    if args.merge_voxel:
        merged = merged.voxel_down_sample(args.merge_voxel)
        print(f"[desktop] merged cloud after dedupe: {len(merged.points)} pts")

    out_path = os.path.join(save_dir, "field_scan.pcd")
    o3d.io.write_point_cloud(out_path, merged)
    print(f"[desktop] saved merged {len(merged.points)} pts to {out_path}")

    if not args.no_visualize:
        merged.paint_uniform_color([0.4, 0.6, 0.9])
        o3d.visualization.draw_geometries(
            [merged, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)],
            window_name="field PCD",
        )


if __name__ == "__main__":
    main()
