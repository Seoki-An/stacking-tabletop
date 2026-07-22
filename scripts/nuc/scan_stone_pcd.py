import argparse
import csv
import os
import time
import numpy as np
import open3d as o3d
from typing import List, Tuple

import rclpy
from rclpy.clock import Clock

from ros2.joint_node import CtrlJointPublisher, CtrlJointSubscriber
from ros2.pcd_node import PointCloudSubscriber
from ros2.control import position_control, project_angle

from planning import Q_SCAN


def scan_pcd(
    q: np.ndarray,
    v: np.ndarray,
    q_rot_list: List[float],
    duration_time: float,
    joint_node_pub: CtrlJointPublisher,
    joint_node_sub: CtrlJointSubscriber,
    pcd_node_sub: PointCloudSubscriber,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    data = []
    for q_rot in q_rot_list:
        q[5] = q_rot
        position_control(q, v, joint_node_pub, joint_node_sub)
        time.sleep(0.5)
        start_time = time.time()
        while time.time() - start_time < duration_time:
            rclpy.spin_once(pcd_node_sub, timeout_sec=0.1)
            rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
            if pcd_node_sub.get_flag() and joint_node_sub.get_flag():
                ts = Clock().now().nanoseconds
                data.append(
                    (ts, pcd_node_sub.points.copy(), joint_node_sub.pos.copy())
                )
            pcd_node_sub.reset_get_flag()
            joint_node_sub.reset_get_flag()

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", "-d", type=str, required=True, help="specify pcd save path"
    )
    args = parser.parse_args()
    save_root = args.dir

    print("Stone pcd scan starts...")
    n_rots = 10
    duration_time = 2.0  # (s)
    bucket_angle_pos = 0.6  # (rad)
    bucket_angle_neg = -0.9
    tilt_angle = 0.4  # (rad)
    drot = 2 * np.pi / n_rots
    q_init = Q_SCAN.copy()
    v = np.zeros(6)

    q_rot_list = []
    for i in range(n_rots):
        q_rot = q_init[5] + drot * i
        if (
            abs(project_angle(q_rot)) < np.pi / 4
            or abs(project_angle(q_rot)) > 3 * np.pi / 4
        ):
            q_rot_list.append(q_rot)

    data = []
    #######################################################################

    rclpy.init()
    joint_node_pub = CtrlJointPublisher()
    joint_node_sub = CtrlJointSubscriber()
    pcd_node_sub = PointCloudSubscriber(topic="/iv_points21")

    print("To the initial position...")
    position_control(q_init, v, joint_node_pub, joint_node_sub)
    print("Initialization done")

    ############################### Pose 1 #################################
    q = q_init.copy()
    d = scan_pcd(
        q,
        v,
        q_rot_list,
        duration_time,
        joint_node_pub,
        joint_node_sub,
        pcd_node_sub,
    )
    data.extend(d)

    ############################### Pose 2 #################################
    q = q_init.copy()
    position_control(q_init, np.zeros_like(q_init), joint_node_pub, joint_node_sub)
    time.sleep(2.0)
    q[-3] += bucket_angle_pos
    d = scan_pcd(
        q,
        v,
        q_rot_list,
        duration_time,
        joint_node_pub,
        joint_node_sub,
        pcd_node_sub,
    )
    data.extend(d)

    ############################### Pose 3 #################################
    q = q_init.copy()
    position_control(q_init, np.zeros_like(q_init), joint_node_pub, joint_node_sub)
    time.sleep(2.0)
    q[-3] += bucket_angle_neg
    d = scan_pcd(
        q,
        v,
        q_rot_list,
        duration_time,
        joint_node_pub,
        joint_node_sub,
        pcd_node_sub,
    )
    data.extend(d)

    save_root = os.path.join(os.getcwd(), save_root)
    pcd_dir = os.path.join(save_root, "pcd")
    os.makedirs(pcd_dir, exist_ok=True)
    csv_path = os.path.join(save_root, "joint_log.csv")

    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        if write_header:
            csv_writer.writerow(
                [
                    "time",
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                ]
            )

        for ts, pcd_array, joint_array in data:

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_array)
            pcd_filename = os.path.join(pcd_dir, f"{ts}.pcd")
            o3d.io.write_point_cloud(pcd_filename, pcd)

            csv_writer.writerow([ts] + joint_array.tolist())

    print(f"Saved {len(data)} samples to:\n - {pcd_dir}\n - {csv_path}")

    joint_node_pub.destroy_node()
    joint_node_sub.destroy_node()
    pcd_node_sub.destroy_node()
    rclpy.shutdown()
