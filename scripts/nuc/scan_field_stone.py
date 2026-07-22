import os
import time
import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import rclpy
import csv
from datetime import datetime

from model import get_stone_model, get_excavator_model
from perception import box_crop_largest_cluster, multiscale_icp

from ros2.joint_node import CtrlJointPublisher, CtrlJointSubscriber
from ros2.pose_node import PoseArrayPublisher
from ros2.pcd_node import PointCloudSubscriber
from ros2.sceneid_node import (
    SceneIdentifierPublisher,
    SceneScanDonePublisher,
    JointLogPublisher,
)
from ros2.control import position_control

from utils import get_unique_dir

SCAN_MODE = 2
POSE_IDENTIFICATION = False
SCAN_WAIT_TIMEOUT = 5.0

# Stream the (base-frame) field scan to the desktop over WiFi. Each per-LiDAR
# frame is published as it is captured; the desktop (save_field_pcd.py) merges
# and saves them. A modest voxel keeps it dense while bounding payload.
PUBLISH_FIELD_PCD = True
FIELD_PCD_TOPIC = "/field_pcd"
FIELD_PCD_DONE_TOPIC = "/field_pcd_done"
FIELD_JOINT_LOG_TOPIC = "/field_joint_log"
FIELD_PUBLISH_VOXEL = 0.005


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
                print(f"Field scan warning: missing LiDAR topics flags={pcd_flags}")
            return joint_node_sub.pos.copy()

    pcd_counts = [len(pcd_node.points) for pcd_node in pcd_nodes]
    pcd_flags = [pcd_node.get_flag() for pcd_node in pcd_nodes]
    raise RuntimeError(
        "Failed to receive non-empty field scan data: "
        f"joint={joint_node_sub.get_flag()}, "
        f"pcd_flags={pcd_flags}, pcd_counts={pcd_counts}"
    )


if __name__ == "__main__":

    q_home = np.array([0.0, np.pi / 4, -np.pi / 2, 0.0, 0.0, 0.0])
    v = np.zeros(6)
    duration_time = 10.0
    root_dir = ".data/field_pcd"
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
        excavator_q_list = [
            np.array([-np.pi / 2, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 3, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 6, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([0 * np.pi, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([np.pi / 6, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([np.pi / 3, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([np.pi / 2, np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0]),
            np.array([np.pi / 2, np.pi / 8, -np.pi / 3, 0.70, 0.0, 0.0]),
            np.array([np.pi / 3, np.pi / 8, -np.pi / 3, 0.70, 0.0, 0.0]),
            np.array([np.pi / 6, np.pi / 8, -np.pi / 3, 0.70, 0.0, 0.0]),
            np.array([0 * np.pi, np.pi / 8, -np.pi / 3, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 6, np.pi / 8, -np.pi / 3, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 3, np.pi / 8, -np.pi / 3, 0.70, 0.0, 0.0]),
            np.array([-np.pi / 2, np.pi / 8, -np.pi / 3, 0.70, 0.0, 0.0]),
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
    pose_array_pub = PoseArrayPublisher()

    field_pcd_pub = None
    field_pcd_done_pub = None
    field_joint_log_pub = None
    if PUBLISH_FIELD_PCD:
        field_pcd_pub = SceneIdentifierPublisher(
            name="field_pcd_publisher",
            topic=FIELD_PCD_TOPIC,
            voxel=FIELD_PUBLISH_VOXEL,
        )
        field_pcd_done_pub = SceneScanDonePublisher(
            name="field_pcd_done_publisher", topic=FIELD_PCD_DONE_TOPIC
        )
        field_joint_log_pub = JointLogPublisher(
            name="field_joint_log_publisher", topic=FIELD_JOINT_LOG_TOPIC
        )

    origin_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )

    excavator_model, _ = get_excavator_model()
    stone_meshes, _, pcds, _ = get_stone_model()

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
            os.path.join(save_dir, f"field_scan_lidar1_{i}_raw.pcd"), pcd
        )
        if field_pcd_pub is not None:
            field_pcd_pub.publish(pcd, frame_id=f"field_scan_lidar1_{i}_raw")
        pcd.transform(lidar_frame)
        scene_pcd += pcd
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"field_scan_lidar1_{i}.pcd"), pcd
        )
        if field_pcd_pub is not None:
            field_pcd_pub.publish(pcd, frame_id=f"field_scan_lidar1_{i}")

        lidar_frame = np.eye(4)
        lidar_frame[:3, :3] = excavator_model.GetLink("lidar2_link").GetRotation()
        lidar_frame[:3, -1] = excavator_model.GetLink("lidar2_link").GetPosition()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd2_node_sub.points.copy())
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"field_scan_lidar2_{i}_raw.pcd"), pcd
        )
        if field_pcd_pub is not None:
            field_pcd_pub.publish(pcd, frame_id=f"field_scan_lidar2_{i}_raw")
        pcd.transform(lidar_frame)
        scene_pcd += pcd
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"field_scan_lidar2_{i}.pcd"), pcd
        )
        if field_pcd_pub is not None:
            field_pcd_pub.publish(pcd, frame_id=f"field_scan_lidar2_{i}")

        lidar_frame = np.eye(4)
        lidar_frame[:3, :3] = excavator_model.GetLink("lidar3_link").GetRotation()
        lidar_frame[:3, -1] = excavator_model.GetLink("lidar3_link").GetPosition()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd3_node_sub.points.copy())

        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"field_scan_lidar3_{i}_raw.pcd"), pcd
        )
        if field_pcd_pub is not None:
            field_pcd_pub.publish(pcd, frame_id=f"field_scan_lidar3_{i}_raw")
        pcd = pcd.transform(lidar_frame)
        scene_pcd += pcd
        o3d.io.write_point_cloud(
            os.path.join(save_dir, f"field_scan_lidar3_{i}.pcd"), pcd
        )
        if field_pcd_pub is not None:
            field_pcd_pub.publish(pcd, frame_id=f"field_scan_lidar3_{i}")

        pcd1_node_sub.reset_get_flag()
        pcd2_node_sub.reset_get_flag()
        pcd3_node_sub.reset_get_flag()
        joint_node_sub.reset_get_flag()

    position_control(q_home, v, joint_node_pub, joint_node_sub)

    o3d.io.write_point_cloud(os.path.join(save_dir, "field_scan.pcd"), scene_pcd)

    # Send the joint log (one row per scan pose) before the done signal so the
    # desktop has it in hand when it stops collecting.
    if field_joint_log_pub is not None and q_list:
        for _ in range(10):
            field_joint_log_pub.publish(q_list)
            time.sleep(0.02)

    if field_pcd_done_pub is not None:
        # Tell the desktop no more frames are coming for this field scan.
        for _ in range(10):
            field_pcd_done_pub.publish()
            time.sleep(0.02)

    csv_path = os.path.join(save_dir, "joint_log.csv")
    with open(csv_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for q in q_list:
            csv_writer.writerow(q.tolist())

    if not POSE_IDENTIFICATION:
        exit(0)

    scene_pcd = scene_pcd.voxel_down_sample(0.01)
    scene_pcd = box_crop_largest_cluster(
        # scene_pcd, [-10, 10], [-10, 0], [-1, 2], cluster=False
        scene_pcd,
        [0, 8],
        [-5, 5],
        [-1, 2],
        cluster=False,
    )

    o3d.visualization.draw_geometries([scene_pcd, origin_coord])

    plane_model, inliers = scene_pcd.segment_plane(
        distance_threshold=0.05, ransac_n=5, num_iterations=1000
    )
    stone_pcd = scene_pcd.select_by_index(inliers, invert=True)

    labels = np.array(
        stone_pcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=False)
    )
    if labels.size == 0 or labels.max() < 0:
        print("No stone clusters found after ground removal.")
        exit(0)

    max_label = labels.max()
    print(f"Cluster count: {max_label + 1}")

    cluster_pcds = []

    for cluster_id in range(max_label + 1):
        idx = np.where(labels == cluster_id)[0]
        cluster = stone_pcd.select_by_index(idx)
        cluster_pcds.append(cluster)

    cluster_pcds = sorted(cluster_pcds, key=lambda c: len(c.points), reverse=True)

    colors = np.random.rand(max_label + 1, 3)
    colored_points = [colors[label] if label >= 0 else [0, 0, 0] for label in labels]
    stone_pcd.colors = o3d.utility.Vector3dVector(colored_points)

    o3d.visualization.draw_geometries([stone_pcd])

    angles_x = [0, 90, 180, 270]
    angles_y = [0, 90, 180, 270]
    angles_z = [0, 90, 180, 270]

    angles_x, angles_y, angles_z = np.meshgrid(
        angles_x, angles_y, angles_z, indexing="ij"
    )
    angles_x = angles_x.flatten()
    angles_y = angles_y.flatten()
    angles_z = angles_z.flatten()
    rots = []
    for angle_x, angle_y, angle_z in zip(angles_x, angles_y, angles_z):
        rots.append(
            Rotation.from_euler(
                "xyz", [angle_x, angle_y, angle_z], degrees=True
            ).as_matrix()
        )

    poses = []
    pcds_backup = copy.deepcopy(pcds)

    for i in range(min(max_label + 1, len(pcds))):
        lidar_pcd = cluster_pcds[i]

        icp_results = []
        center = np.asarray(lidar_pcd.points).mean(0)
        init_T = np.eye(4)
        init_T[:3, -1] = center
        if len(pcds) == 0:
            continue

        for id, pcd in pcds.items():
            for rot in rots:
                init_T[:3, :3] = rot
                target_T, history = multiscale_icp(
                    pcd,
                    lidar_pcd,
                    init_T,
                    voxel_sizes=[0.1, 0.05, 0.02, 0.01],
                    max_iters=[50, 30, 14, 7],
                )
                icp_results.append((id, target_T, history[-1][1] - history[-1][2]))

        id = max(icp_results, key=lambda x: x[-1])[0]
        target_T = max(icp_results, key=lambda x: x[-1])[1]
        fitness = max(icp_results, key=lambda x: x[-1])[-1]
        stone_meshes[id].transform(target_T)
        poses.append(
            (target_T[:3, -1], Rotation.from_matrix(target_T[:3, :3]).as_quat(), id)
        )
        pcds.pop(id)
        print(
            f"ICP for {id} with center {center.tolist()} and fitness {fitness} is done"
        )

    for pos, quat, id in poses:
        print(f"model_{id} - pos = {pos.tolist()}, quat = {quat.tolist()}")

    plane_width, plane_length = 30.0, 30.0
    plane = o3d.geometry.TriangleMesh.create_box(plane_width, plane_length, 0.001)
    plane = plane.translate([-plane_width / 2, -plane_length / 2, -0.001])

    o3d.visualization.draw_geometries(
        [mesh for mesh in stone_meshes.values()] + [origin_coord, plane] + cluster_pcds
    )

    pose_array_pub.publish(poses)
