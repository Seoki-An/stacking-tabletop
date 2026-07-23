#!/usr/bin/env python3
import copy
import os
import time
import numpy as np
import open3d as o3d
from typing import Optional, Tuple, List, Union

from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from tp_msgs.msg import PoseWithId

from perception import multiscale_icp, box_crop_largest_cluster
from perception.utils.refine_pcd import remove_points_from_points
from model import get_stone_model, get_gripper_model, update_urdf_mesh

from .joint_node import CtrlJointPublisher, CtrlJointSubscriber
from .control import position_control

# Cropping intervals
WIDTH = np.array([-2.0, 2.0])
LENGTH = np.array([0.0, 2.0])
HEIGHT = np.array([-2.0, 2.0])

# Opening angle threshold
OPENING_ANGLE_THRESHOLD = -1.0

# Joints 1..5 used at the recovery field-scan pose; q0 (swing) is computed
# at runtime from the stone's xy in the base frame.
FIELD_SCAN_Q_TAIL = np.array([np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0])
SCENE_SCAN_Q_TAIL = np.array([40.0 / 180 * np.pi, -60.0 / 180 * np.pi, 0.0, 0.0, 0.0])
# Default link names for each subscribed pcd topic (in order).
DEFAULT_LIDAR_LINKS = ["lidar2_link", "lidar3_link"]


class PoseIdentifier(Node):
    def __init__(
        self,
        name: str = "pose_identifier",
        pcd_topics: List[str] = ["/iv_points21"],
        inhand_pose_topic: str = "/inhand_poseid_init",
        field_pose_topic: str = "/field_poseid_init",
        stone_model_dir: str = "assets/stone",
    ):
        super().__init__(name)

        # depth=1 (KEEP_LAST) so the subscriber buffer always holds only the
        # most recently published sample. With the previous depth=10, mid-motion
        # PCDs from before a scan dwell would queue up; reset_get_pcd_flag()
        # only clears the local flag, not the DDS buffer, so the next
        # spin_once would pop the OLDEST queued PCD (stale, wrong q) and pair
        # it with the current dwell-aligned joint feedback.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pcd_topics = list(pcd_topics)
        self.pcd_subscription_ = []
        self.points_per_topic = {topic: np.zeros([0, 3]) for topic in self.pcd_topics}
        self._get_pcd_flag_per_topic = {topic: False for topic in self.pcd_topics}
        for pcd_topic in self.pcd_topics:
            self.pcd_subscription_.append(
                self.create_subscription(
                    PointCloud2,
                    pcd_topic,
                    lambda msg, t=pcd_topic: self.pointcloud_callback(msg, t),
                    qos,
                )
            )
        self.inhand_poseid_subscription_ = self.create_subscription(
            PoseWithId, inhand_pose_topic, self.inhand_poseid_subscription_, qos
        )
        self.field_poseid_subscription_ = self.create_subscription(
            PoseWithId, field_pose_topic, self.field_poseid_subscription_, qos
        )

        self.last_pcd_msg = None
        self.points = np.zeros([0, 3])

        self.last_inhand_msg = None
        self.inhand_pose_init = None
        self.inhand_id = -1
        self._inhand_pose_seq = 0

        self.last_field_msg = None
        self.field_pose_init = None
        self.field_id = -1
        self._field_pose_seq = 0

        self.stone_meshes, self.stone_configs, self.stone_pcds, _ = get_stone_model(
            stone_model_dir
        )

        self._get_pcd_flag = False
        self._get_pose_flag = False
        self._get_inhand_pose_flag = False
        self._get_field_pose_flag = False

        coeff = np.load("assets/opening_angle_fit.npy")
        self.angle_poly_fit = np.poly1d(coeff)

        self.gripper_model, self.gripper_meshes = get_gripper_model()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        self.last_opening_angle = None

    def pointcloud_callback(self, msg: PointCloud2, topic: str):
        self.last_pcd_msg = msg
        try:
            points = point_cloud2.read_points_numpy(
                msg, field_names=["x", "y", "z"], skip_nans=True
            )
        except Exception as e:
            self.get_logger().error(
                f"Failed to convert PointCloud2 to numpy array: {e}"
            )
            return

        if points.size == 0:
            self.get_logger().warning(f"Received empty point cloud from {topic}")
            return

        self.points_per_topic[topic] = points
        self.points = points
        self._get_pcd_flag_per_topic[topic] = True
        self._get_pcd_flag = True

    def inhand_poseid_subscription_(self, msg: PoseWithId):
        self.last_inhand_msg = msg
        try:
            pos = msg.pose.position
            ori = msg.pose.orientation

            pos = np.array([pos.x, pos.y, pos.z])
            quat = np.array([ori.x, ori.y, ori.z, ori.w])

            self.inhand_pose_init = (pos, quat)
            self.inhand_id = msg.id
            self._inhand_pose_seq += 1
            self._get_pose_flag = True
            self._get_inhand_pose_flag = True

        except Exception as e:
            self.get_logger().error(f"Failed to process PoseWithId message: {e}")
            return

    def field_poseid_subscription_(self, msg: PoseWithId):
        self.last_field_msg = msg
        try:
            pos = msg.pose.position
            ori = msg.pose.orientation

            pos = np.array([pos.x, pos.y, pos.z])
            quat = np.array([ori.x, ori.y, ori.z, ori.w])

            self.field_pose_init = (pos, quat)
            self.field_id = msg.id
            self._field_pose_seq += 1
            self._get_pose_flag = True
            self._get_field_pose_flag = True

        except Exception as e:
            self.get_logger().error(f"Failed to process PoseWithId message: {e}")
            return

    def reset_get_pcd_flag(self, topic: Optional[str] = None):
        if topic is None:
            self._get_pcd_flag = False
            for t in self._get_pcd_flag_per_topic:
                self._get_pcd_flag_per_topic[t] = False
        else:
            self._get_pcd_flag_per_topic[topic] = False

    def get_pcd_flag(self, topic: Optional[str] = None) -> bool:
        if topic is None:
            return self._get_pcd_flag
        return self._get_pcd_flag_per_topic[topic]

    def reset_get_pose_flag(self):
        self._get_pose_flag = False
        self._get_inhand_pose_flag = False
        self._get_field_pose_flag = False

    def get_pose_flag(self) -> bool:
        return self._get_pose_flag

    def get_inhand_pose_flag(self) -> bool:
        return self._get_inhand_pose_flag

    def get_field_pose_flag(self) -> bool:
        return self._get_field_pose_flag

    def get_inhand_pose_seq(self) -> int:
        return self._inhand_pose_seq

    def get_field_pose_seq(self) -> int:
        return self._field_pose_seq

    def detect_grasp_failure(self, pcd: np.ndarray) -> bool:
        """Return True if the gripper appears empty.

        Uses the same opening-angle estimate as ``identify_inhand_pose_multiple``:
        crop the gripper-frame PCD, measure x-extent, map through
        ``angle_poly_fit``, and compare against ``OPENING_ANGLE_THRESHOLD``.

        Arguments:
            pcd: np.ndarray of shape (N, 3) in the gripper frame.
        """
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        pcd_cropped = box_crop_largest_cluster(
            pcd_o3d, WIDTH, LENGTH, HEIGHT, cluster=False
        )
        pts = np.asarray(pcd_cropped.points)
        if pts.size == 0:
            self.get_logger().warning(
                "detect_grasp_failure: cropped PCD is empty; treating as failure."
            )
            return True
        x_extent = pts[:, 0].max() - pts[:, 0].min()
        opening_angle = float(self.angle_poly_fit(x_extent))
        self.last_opening_angle = opening_angle
        print(f"Estimated opening angle (rad): {opening_angle}")
        return opening_angle < OPENING_ANGLE_THRESHOLD

    def identify_inhand_pose(
        self,
        pcd: np.ndarray,
        pose_init: Tuple[np.ndarray, np.ndarray],
        id: int,
        visualize: bool = False,
    ) -> Union[None, np.ndarray]:
        """
        Arguments:
            pcd: np.ndarray of shape (N, 3), point cloud in the manipulator base frame
            pose_init: Tuple of (pos, quat), where pos is np.ndarray of shape (3,) and quat is np.ndarray of shape (4,) in the same frame as pcd
            id: int, index of the stone type
        """

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        pcd_cropped = box_crop_largest_cluster(
            pcd_o3d, WIDTH, LENGTH, HEIGHT, cluster=False
        )
        pcd_cropped_np = np.asarray(pcd_cropped.points)
        x_extent = pcd_cropped_np[:, 0].max() - pcd_cropped_np[:, 0].min()
        opening_angle = float(self.angle_poly_fit(x_extent))
        self.last_opening_angle = opening_angle

        if opening_angle < OPENING_ANGLE_THRESHOLD:
            self.get_logger().warning("Detected opening angle below threshold")
            return None

        q = np.ones(2) * opening_angle

        gripper_meshes = update_urdf_mesh(self.gripper_model, self.gripper_meshes, q)
        gripper_pcd = o3d.geometry.PointCloud()
        for mesh in gripper_meshes.values():
            gripper_pcd += mesh.sample_points_uniformly(number_of_points=1000)
        perturbation_T = np.eye(4)
        perturbation_T[:3, 3] = np.array([0.1, -0.1, 0.05])
        pcd_init = copy.deepcopy(pcd_cropped)
        pcd_cropped.transform(perturbation_T)

        pose_gripper_T, _ = multiscale_icp(pcd_cropped, gripper_pcd, np.eye(4))
        target_pcd = copy.deepcopy(self.stone_pcds[id]).transform(
            np.linalg.inv(pose_gripper_T)
        )
        pose_init_T = np.eye(4)
        pose_init_T[:3, -1] = pose_init[0]
        pose_init_T[:3, :3] = Rotation.from_quat(pose_init[1]).as_matrix()

        # perturbation for test
        perturbation_T = np.eye(4)
        perturbation_T[:3, :3] = Rotation.from_euler(
            "xyz", [20, 15, 10], degrees=True
        ).as_matrix()
        pose_init_T = pose_init_T @ perturbation_T

        pose_opt, _ = multiscale_icp(self.stone_pcds[id], pcd_cropped, pose_init_T)

        target_pcd = copy.deepcopy(self.stone_pcds[id]).transform(pose_init_T)
        target_pcd_transformed = copy.deepcopy(self.stone_pcds[id]).transform(pose_opt)

        pcd_init.paint_uniform_color(
            [1, 1, 0]
        )  # Yellow color for the initial pose point cloud
        pcd_init.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        pcd_cropped.paint_uniform_color(
            [0, 0, 1]
        )  # Blue color for the cropped point cloud
        pcd_cropped.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        target_pcd.paint_uniform_color(
            [1, 0, 0]
        )  # Red color for the initial pose point cloud
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        target_pcd_transformed.paint_uniform_color(
            [0, 1, 0]
        )  # Green color for the target point cloud
        target_pcd_transformed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        if visualize:
            o3d.visualization.draw_geometries(
                [
                    # pcd_init,
                    pcd_cropped,
                    target_pcd,
                    target_pcd_transformed,
                    self.coord_frame,
                ]
                # + list(gripper_meshes.values())
            )

        return pose_opt

    def identify_inhand_pose_multiple(
        self,
        pcds: List[np.ndarray],
        pose_init: Tuple[np.ndarray, np.ndarray],
        id: int,
        visualize: bool = False,
    ) -> Union[None, np.ndarray]:
        """
        Arguments:
            pcds: list of np.ndarray of shape (N, 3), point clouds in the manipulator base frame
            pose_init: Tuple of (pos, quat), where pos is np.ndarray of shape (3,) and quat is np.ndarray of shape (4,) in the same frame as pcd
            id: int, index of the stone type
        """
        print(
            f"Stone {id} initial pose: ", pose_init[0].tolist() + pose_init[1].tolist()
        )
        pose_init_T = np.eye(4)
        pose_init_T[:3, -1] = pose_init[0]
        pose_init_R = Rotation.from_quat(pose_init[1]).as_matrix()
        pose_init_T[:3, :3] = pose_init_R

        target_pcd = copy.deepcopy(self.stone_pcds[id])
        pcd_merged = o3d.geometry.PointCloud()
        pcd_merged_vis = copy.deepcopy(pcd_merged)

        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(pcds[0])
        pcd0_cropped = box_crop_largest_cluster(
            pcd0, WIDTH, LENGTH, HEIGHT, cluster=False
        )
        pcd_cropped_np = np.asarray(pcd0_cropped.points)
        x_extent = pcd_cropped_np[:, 0].max() - pcd_cropped_np[:, 0].min()
        opening_angle = float(self.angle_poly_fit(x_extent))
        self.last_opening_angle = opening_angle
        print("Estimated opening angle (rad): ", opening_angle)
        if visualize:
            o3d.visualization.draw_geometries([pcd0_cropped])
        if opening_angle < OPENING_ANGLE_THRESHOLD:
            self.get_logger().warning("Detected opening angle below threshold")
            return None

        q = np.ones(2) * opening_angle

        gripper_meshes = update_urdf_mesh(self.gripper_model, self.gripper_meshes, q)
        gripper_pcd = o3d.geometry.PointCloud()
        for mesh in gripper_meshes.values():
            gripper_pcd += mesh.sample_points_uniformly(number_of_points=10000)

        for pcd in pcds:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
            pcd_cropped = box_crop_largest_cluster(
                pcd_o3d, WIDTH, LENGTH, HEIGHT, cluster=False
            )

            pose_gripper_T, _ = multiscale_icp(pcd_cropped, gripper_pcd, np.eye(4))
            pcd_cropped.transform(pose_gripper_T)

            # pcd_merged += pcd_cropped
            pcd_merged += remove_points_from_points(
                gripper_pcd, pcd_cropped, threshold=0.06
            )
            pcd_merged_vis += pcd_cropped

        pcd_merged.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd_merged_vis.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd_merged.paint_uniform_color(
            [0.5, 0.5, 0.5]
        )  # Gray color for the merged point cloud
        pcd_merged_vis.paint_uniform_color([0.5, 0.5, 0.5])
        # perturbation for test
        if False:
            perturbation_T = np.eye(4)
            perturbation_T[:3, :3] = Rotation.from_euler(
                "xyz", [20, 15, 10], degrees=True
            ).as_matrix()
            pose_init_T = pose_init_T @ perturbation_T

        icp_results = []
        for angle_x in [0, 90, 180, 270]:
            init_T = pose_init_T.copy()
            init_T[:3, :3] = (
                Rotation.from_euler("x", angle_x, degrees=True).as_matrix()
                @ pose_init_R
            )
            pose_candidate, history = multiscale_icp(
                self.stone_pcds[id],
                pcd_merged,
                init_T,
            )
            score = history[-1][1] - history[-1][2]
            icp_results.append((pose_candidate, init_T, angle_x, score))

        pose_opt, best_pose_init_T, best_angle_x, best_score = max(
            icp_results, key=lambda x: x[-1]
        )
        print(
            "In-hand pose ICP best x-rotation seed: "
            f"{best_angle_x} deg, score={best_score}"
        )

        target_pcd = copy.deepcopy(self.stone_pcds[id]).transform(best_pose_init_T)
        target_pcd_transformed = copy.deepcopy(self.stone_pcds[id]).transform(pose_opt)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        gripper_pcd.paint_uniform_color([0.2, 1.0, 0.2])

        target_pcd.paint_uniform_color(
            [1, 0, 0]
        )  # Red color for the initial pose point cloud
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        target_pcd_transformed.paint_uniform_color(
            [0, 1, 0]
        )  # Green color for the target point cloud
        target_pcd_transformed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        if visualize:
            o3d.visualization.draw_geometries([pcd_merged])
            o3d.visualization.draw_geometries(
                [
                    pcd_merged,
                    target_pcd,
                    target_pcd_transformed,
                    coord_frame,
                ]
                + list(gripper_meshes.values())
            )
            o3d.visualization.draw_geometries([pcd_merged_vis, target_pcd, coord_frame])
            o3d.visualization.draw_geometries(
                [pcd_merged_vis, target_pcd_transformed, coord_frame]
            )

        return pose_opt

    def recover_field_pose(
        self,
        target_id: int,
        dwell_time: float,
        joint_node_pub: CtrlJointPublisher,
        joint_node_sub: CtrlJointSubscriber,
        excavator_model,
        pose_init: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        lidar_links: Optional[List[str]] = None,
        log_dir: Optional[str] = None,
        scene_pcd_pub=None,
        swing_variation: float = 0.0,
        swing_num: int = 1,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Move to one or more scan poses aimed at the failed stone, fuse the
        point clouds from this node's subscribed LiDAR topics into the base
        frame, and ICP against the stone's model to re-identify its field
        pose after a failed grasp.

        Arguments:
            target_id: id of the stone to re-identify.
            dwell_time: dwell time (s) at each scan pose for LiDAR accumulation.
            pose_init: (pos, quat) initial pose guess in the base frame.
                Defaults to ``self.field_pose_init`` (set by the publisher
                on ``field_pose_topic``).
            lidar_links: link names matching ``self.pcd_topics`` order.
                Defaults to ``DEFAULT_LIDAR_LINKS`` truncated to that length.
            log_dir: optional directory where the scan inputs/intermediates
                should be saved for debugging.
            scene_pcd_pub: optional publisher used by desktop execution to
                transfer the same base-frame scan PCDs back to the desktop.
            swing_variation: half-range (rad) of swing-angle offsets around q0.
            swing_num: number of evenly-spaced swing poses to visit.

        Returns:
            (pos, quat) of the re-identified stone, or None on failure.
        """
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        def save_array(name: str, value):
            if log_dir is None:
                return
            np.save(os.path.join(log_dir, name), value)

        def save_pcd(name: str, pcd: o3d.geometry.PointCloud):
            if log_dir is None:
                return
            o3d.io.write_point_cloud(os.path.join(log_dir, name), pcd)

        def save_status(message: str):
            if log_dir is None:
                return
            with open(os.path.join(log_dir, "field_scan_status.txt"), "w") as f:
                f.write(message + "\n")

        def safe_topic_name(topic: str) -> str:
            return topic.strip("/").replace("/", "_") or "pcd"

        if pose_init is None:
            pose_init = self.field_pose_init
        if pose_init is None:
            self.get_logger().error(
                "recover_field_pose: no pose_init provided and "
                "self.field_pose_init has not been received yet."
            )
            save_status("failed: no field pose init")
            return None

        if lidar_links is None:
            lidar_links = DEFAULT_LIDAR_LINKS[: len(self.pcd_topics)]
        assert len(lidar_links) == len(
            self.pcd_topics
        ), "lidar_links must have one entry per subscribed pcd topic"

        pos_init = pose_init[0]
        v = np.zeros(6)

        q0 = float(np.arctan2(pos_init[1], pos_init[0]))
        q0_offsets = (
            np.linspace(-swing_variation, swing_variation, swing_num)
            if swing_num > 1
            else [0.0]
        )
        q_swing_list = [
            np.concatenate([[q0 + offset], FIELD_SCAN_Q_TAIL]) for offset in q0_offsets
        ]
        save_array("field_id.npy", np.array(target_id, dtype=np.int64))
        save_array("field_pose_init_pos.npy", pose_init[0])
        save_array("field_pose_init_rot.npy", pose_init[1])
        save_array("field_scan_q_swing_list.npy", np.array(q_swing_list))

        scene_pcd = o3d.geometry.PointCloud()
        for swing_idx, q_swing in enumerate(q_swing_list):
            print(
                f"Field re-scan for stone {target_id} "
                f"(swing {swing_idx + 1}/{len(q_swing_list)}): "
                f"moving to q={q_swing.tolist()} "
                f"(q0 derived from stone xy {pos_init[:2].tolist()})."
            )
            position_control(
                q_swing,
                v,
                joint_node_pub,
                joint_node_sub,
                error_tol=0.05,
                time_limit=15.0,
            )
            print("Reached scanning pose; dwelling for LiDAR accumulation.")
            time.sleep(dwell_time)

            self.reset_get_pcd_flag()
            joint_node_sub.reset_get_flag()

            start = time.time()
            while time.time() - start < 0.3:
                rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
                rclpy.spin_once(self, timeout_sec=0.1)

            wait_start = time.time()
            while not joint_node_sub.get_flag():
                if time.time() - wait_start > 5.0:
                    print("Field re-scan failed: timed out waiting for joint feedback.")
                    save_status("failed: timed out waiting for joint feedback")
                    return None
                rclpy.spin_once(joint_node_sub, timeout_sec=0.01)
                joint_node_pub.publish(q_swing, v)
            for topic in self.pcd_topics:
                wait_start = time.time()
                while not self.get_pcd_flag(topic):
                    if time.time() - wait_start > max(5.0, dwell_time + 2.0):
                        print(
                            "Field re-scan warning: timed out waiting for point cloud "
                            f"topic {topic}. Received flags: "
                            f"{ {t: self.get_pcd_flag(t) for t in self.pcd_topics} }"
                        )
                        break
                    rclpy.spin_once(self, timeout_sec=0.1)
                    joint_node_pub.publish(q_swing, v)

            q_actual = joint_node_sub.pos.copy()
            save_array(f"field_scan_q_actual_swing{swing_idx}.npy", q_actual)
            excavator_model.SetState(np.concatenate([q_actual, np.zeros(2)]))

            for topic, link_name in zip(self.pcd_topics, lidar_links):
                pts = self.points_per_topic[topic].copy()
                save_array(
                    f"field_raw_{safe_topic_name(topic)}_swing{swing_idx}.npy", pts
                )
                if pts.size == 0:
                    continue
                lidar_frame = np.eye(4)
                lidar_frame[:3, :3] = excavator_model.GetLink(link_name).GetRotation()
                lidar_frame[:3, -1] = excavator_model.GetLink(link_name).GetPosition()
                save_array(
                    f"field_lidar_frame_{safe_topic_name(topic)}_swing{swing_idx}.npy",
                    lidar_frame,
                )
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.transform(lidar_frame)
                save_pcd(
                    f"field_base_{safe_topic_name(topic)}_swing{swing_idx}.ply", pcd
                )
                if scene_pcd_pub is not None:
                    scene_pcd_pub.publish(
                        pcd,
                        frame_id=f"field_base_{safe_topic_name(topic)}_swing{swing_idx}",
                    )
                scene_pcd += pcd

        save_pcd("field_scene_merged.ply", scene_pcd)
        if len(scene_pcd.points) == 0:
            print("Field re-scan failed: no points received from LiDARs.")
            save_status("failed: no points received from LiDARs")
            return None

        min_bound = pos_init - np.array([0.8, 0.8, 0.5])
        max_bound = pos_init + np.array([0.8, 0.8, 1.0])
        save_array("field_crop_min_bound.npy", min_bound)
        save_array("field_crop_max_bound.npy", max_bound)
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        scene_pcd_cropped = scene_pcd.crop(bbox)
        save_pcd("field_scene_cropped.ply", scene_pcd_cropped)

        if len(scene_pcd_cropped.points) < 100:
            print(
                "Field re-scan failed: too few points "
                f"({len(scene_pcd_cropped.points)}) around initial pose guess."
            )
            save_status(
                "failed: too few points "
                f"({len(scene_pcd_cropped.points)}) around initial pose guess"
            )
            return None

        scene_pcd_cropped = scene_pcd_cropped.voxel_down_sample(0.01)
        save_pcd("field_scene_cropped_downsampled.ply", scene_pcd_cropped)

        # Grid-search initial orientations (same scheme as
        # scripts/nuc/scan_field_stone.py) since the post-failure stone may
        # have rolled and quat_init is unreliable. Position is pinned to
        # pos_init; we pick the rotation that maximises fitness - rmse.
        angles = [0, 90, 180, 270]
        ax_grid, ay_grid, az_grid = np.meshgrid(angles, angles, angles, indexing="ij")
        rots = [
            Rotation.from_euler("xyz", [ax, ay, az], degrees=True).as_matrix()
            for ax, ay, az in zip(
                ax_grid.flatten(), ay_grid.flatten(), az_grid.flatten()
            )
        ]

        init_T = np.eye(4)
        init_T[:3, -1] = pos_init

        icp_results = []
        for rot in rots:
            init_T[:3, :3] = rot
            target_T, history = multiscale_icp(
                self.stone_pcds[target_id],
                scene_pcd_cropped,
                init_T,
                voxel_sizes=[0.1, 0.05, 0.02, 0.01],
                max_iters=[50, 30, 14, 7],
            )
            icp_results.append((target_T, history[-1][1] - history[-1][2]))

        pose_opt_T, best_score = max(icp_results, key=lambda x: x[-1])
        print(f"Field re-scan best ICP score (fitness - rmse): {best_score}")
        save_array("field_icp_score.npy", np.array(best_score))
        save_array("field_pose_opt_T.npy", pose_opt_T)

        pos_opt = pose_opt_T[:3, -1]
        quat_opt = Rotation.from_matrix(pose_opt_T[:3, :3]).as_quat()
        save_array("field_pose_opt_pos.npy", pos_opt)
        save_array("field_pose_opt_rot.npy", quat_opt)
        save_status("success")
        print(
            f"Field re-identification done. pos={pos_opt.tolist()}, "
            f"quat={quat_opt.tolist()}"
        )
        return pos_opt, quat_opt
