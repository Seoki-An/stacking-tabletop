import os
import time
import numpy as np
import open3d as o3d

import rclpy
import csv
from datetime import datetime

from model import get_excavator_model

from ros2.joint_node import CtrlJointPublisher, CtrlJointSubscriber
from ros2.pcd_node import PointCloudSubscriber
from ros2.sceneid_node import (
    SceneIdentifierPublisher,
    SceneScanDonePublisher,
    JointLogPublisher,
)
from ros2.control import position_control

from utils import get_unique_dir

SCAN_MODE = 2
SCAN_WAIT_TIMEOUT = 5.0

# Stream the base-frame scene scan to the desktop over WiFi. Each per-LiDAR
# frame is published as it is captured; a modest voxel keeps payload bounded.
PUBLISH_SCENE_PCD = True
SCENE_PCD_TOPIC = "/scene_pcd"
SCENE_SCAN_DONE_TOPIC = "/scene_scan_done"
SCENE_JOINT_LOG_TOPIC = "/scene_joint_log"
SCENE_PUBLISH_VOXEL = 0.005


def wait_for_scan_data(
    q,
    joint_node_pub,
    joint_node_sub,
    pcd_nodes,
    timeout=SCAN_WAIT_TIMEOUT,
):
    joint_node_sub.reset_get_flag()
    for pcd_node in pcd_nodes:
        pcd_node.reset_get_flag()
        pcd_node.points = np.zeros([0, 3])

    start = time.time()
    while time.time() - start < timeout:
        joint_node_pub.publish(q)
        rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
        for pcd_node in pcd_nodes:
            rclpy.spin_once(pcd_node, timeout_sec=0.05)

        pcd_flags = [pcd_node.get_flag() for pcd_node in pcd_nodes]
        if joint_node_sub.get_flag() and any(pcd_flags):
            if not all(pcd_flags):
                print(f"Scene scan warning: missing LiDAR topics flags={pcd_flags}")
            return joint_node_sub.pos.copy()

    pcd_counts = [len(pcd_node.points) for pcd_node in pcd_nodes]
    pcd_flags = [pcd_node.get_flag() for pcd_node in pcd_nodes]
    raise RuntimeError(
        "Failed to receive non-empty scene scan data: "
        f"joint={joint_node_sub.get_flag()}, "
        f"pcd_flags={pcd_flags}, pcd_counts={pcd_counts}"
    )


if __name__ == "__main__":

    q_home = np.array([0.0, np.pi / 4, -np.pi / 2, 0.0, 0.0, 0.0])
    v = np.zeros(6)
    duration_time = 10.0
    root_dir = ".data/scene_pcd"
    today_str = datetime.now().strftime("%y%m%d")
    save_dir = get_unique_dir(root_dir, today_str)
    os.makedirs(save_dir, exist_ok=True)

    if SCAN_MODE == 0:
        excavator_q_list = [
            np.array([-np.pi / 2, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 2, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 2, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 2, np.pi / 12, -np.pi / 6, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 2, np.pi / 12, -np.pi / 6, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 2, np.pi / 12, -np.pi / 6, 0.70, 0.0, 0.0]),
        ]
        excavator_q_list[0][0] += np.pi / 4
        excavator_q_list[2][0] -= np.pi / 4
        excavator_q_list[3][0] -= np.pi / 4
        excavator_q_list[5][0] += np.pi / 4
    elif SCAN_MODE == 1:
        excavator_q_list = [
            np.array([-np.pi / 3, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 6, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([0 * np.pi, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([np.pi / 6, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([np.pi / 3, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
        ]
    elif SCAN_MODE == 2:
        rad40 = 40.0 / 180.0 * np.pi
        rad70 = 70.0 / 180.0 * np.pi
        excavator_q_list = [
            np.array([-np.pi / 3, rad40, -rad70, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 6, rad40, -rad70, 0.70, 0.0, 0.0]),
            np.array([0 * np.pi, rad40, -rad70, 0.70, 0.0, 0.0]),
            np.array([np.pi / 6, rad40, -rad70, 0.70, 0.0, 0.0]),
            np.array([np.pi / 3, rad40, -rad70, 0.70, 0.0, 0.0]),
        ]
    else:
        raise RuntimeError(f"Not implemented scan mode {SCAN_MODE}")

    rclpy.init()
    joint_node_pub = CtrlJointPublisher(topic="/joint_ctrl")
    joint_node_sub = CtrlJointSubscriber(topic="/joint_rcv")
    pcd1_node_sub = PointCloudSubscriber(
        name="iv_points22_subscriber", topic="/iv_points22"
    )
    pcd2_node_sub = PointCloudSubscriber(
        name="iv_points21_subscriber", topic="/iv_points21"
    )
    pcd3_node_sub = PointCloudSubscriber(
        name="iv_points20_subscriber", topic="/iv_points20"
    )

    scene_pcd_pub = None
    scene_scan_done_pub = None
    scene_joint_log_pub = None
    if PUBLISH_SCENE_PCD:
        scene_pcd_pub = SceneIdentifierPublisher(
            name="scene_pcd_publisher",
            topic=SCENE_PCD_TOPIC,
            voxel=SCENE_PUBLISH_VOXEL,
        )
        scene_scan_done_pub = SceneScanDonePublisher(
            name="scene_scan_done_publisher", topic=SCENE_SCAN_DONE_TOPIC
        )
        scene_joint_log_pub = JointLogPublisher(
            name="scene_joint_log_publisher", topic=SCENE_JOINT_LOG_TOPIC
        )

    excavator_model, _ = get_excavator_model()

    position_control(q_home, v, joint_node_pub, joint_node_sub)

    scene_pcd = o3d.geometry.PointCloud()
    q_prev = q_home
    q_list = []
    for i, q in enumerate(excavator_q_list):
        position_control(
            0.5 * (q + q_prev), v, joint_node_pub, joint_node_sub, error_tol=0.1
        )
        position_control(q, v, joint_node_pub, joint_node_sub)
        q_prev = q

        print("Complete moving to scanning pose")
        time.sleep(duration_time)

        q = wait_for_scan_data(
            q,
            joint_node_pub,
            joint_node_sub,
            [pcd1_node_sub, pcd2_node_sub, pcd3_node_sub],
        )
        print(f"Received PCD and joint: {q.tolist()}")

        q_list.append(q)
        q = np.concatenate([q, np.zeros(2)])
        excavator_model.SetState(q)

        lidar_frame = np.eye(4)
        lidar_frame[:3, :3] = excavator_model.GetLink("lidar1_link").GetRotation()
        lidar_frame[:3, -1] = excavator_model.GetLink("lidar1_link").GetPosition()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd1_node_sub.points.copy())
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"scene_scan_lidar1_{i}_raw.pcd"), pcd
        )
        if scene_pcd_pub is not None:
            scene_pcd_pub.publish(pcd, frame_id=f"scene_scan_lidar1_{i}_raw")
        pcd.transform(lidar_frame)
        scene_pcd += pcd
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"scene_scan_lidar1_{i}.pcd"), pcd
        )
        if scene_pcd_pub is not None:
            scene_pcd_pub.publish(pcd, frame_id=f"scene_scan_lidar1_{i}")

        lidar_frame = np.eye(4)
        lidar_frame[:3, :3] = excavator_model.GetLink("lidar2_link").GetRotation()
        lidar_frame[:3, -1] = excavator_model.GetLink("lidar2_link").GetPosition()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd2_node_sub.points.copy())
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"scene_scan_lidar2_{i}_raw.pcd"), pcd
        )
        if scene_pcd_pub is not None:
            scene_pcd_pub.publish(pcd, frame_id=f"scene_scan_lidar2_{i}_raw")
        pcd.transform(lidar_frame)
        scene_pcd += pcd
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"scene_scan_lidar2_{i}.pcd"), pcd
        )
        if scene_pcd_pub is not None:
            scene_pcd_pub.publish(pcd, frame_id=f"scene_scan_lidar2_{i}")

        lidar_frame = np.eye(4)
        lidar_frame[:3, :3] = excavator_model.GetLink("lidar3_link").GetRotation()
        lidar_frame[:3, -1] = excavator_model.GetLink("lidar3_link").GetPosition()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd3_node_sub.points.copy())

        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"scene_scan_lidar3_{i}_raw.pcd"), pcd
        )
        if scene_pcd_pub is not None:
            scene_pcd_pub.publish(pcd, frame_id=f"scene_scan_lidar3_{i}_raw")
        pcd = pcd.transform(lidar_frame)
        scene_pcd += pcd
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"scene_scan_lidar3_{i}.pcd"), pcd
        )
        if scene_pcd_pub is not None:
            scene_pcd_pub.publish(pcd, frame_id=f"scene_scan_lidar3_{i}")

        pcd1_node_sub.reset_get_flag()
        pcd2_node_sub.reset_get_flag()
        pcd3_node_sub.reset_get_flag()
        joint_node_sub.reset_get_flag()

    position_control(q_home, v, joint_node_pub, joint_node_sub)

    o3d.io.write_point_cloud(os.path.join(save_dir, "scene_scan.pcd"), scene_pcd)

    if scene_joint_log_pub is not None and q_list:
        for _ in range(10):
            scene_joint_log_pub.publish(q_list)
            time.sleep(0.02)

    if scene_scan_done_pub is not None:
        # Tell the desktop no more frames are coming for this scene scan.
        for _ in range(10):
            scene_scan_done_pub.publish()
            time.sleep(0.02)

    csv_path = os.path.join(save_dir, "joint_log.csv")
    with open(csv_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for q in q_list:
            csv_writer.writerow(q.tolist())
