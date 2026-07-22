#!/usr/bin/env python3
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def numpy_to_pointcloud2(points: np.ndarray, frame_id="map"):
    """
    points: Nx3 np.array (dtype=float32 recommended)
    """
    header = Header()
    header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    return point_cloud2.create_cloud(header, fields, points)


def open3d_to_pointcloud2(pcd: o3d.geometry.PointCloud, frame_id="map"):
    points = np.asarray(pcd.points, dtype=np.float32)
    return numpy_to_pointcloud2(points, frame_id)


class PointCloudSubscriber(Node):
    def __init__(self, name: str = "iv_points_subscriber", topic: str = "/iv_points21"):
        super().__init__(name)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subscription_ = self.create_subscription(
            PointCloud2, topic, self.pointcloud_callback, qos
        )
        self.last_msg = None
        self.points = np.zeros([0, 3])
        self._get_flag = False

    def pointcloud_callback(self, msg: PointCloud2):
        self.last_msg = msg
        try:
            self.points = point_cloud2.read_points_numpy(
                msg,
                field_names=["x", "y", "z"],
                skip_nans=True,
            )  # shape: (N, 3)

        except Exception as e:
            self.get_logger().error(f"PointCloud2 conversion error: {e}")
            return

        if self.points.size == 0:
            self.get_logger().warning("Received empty point cloud")
            return
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class PointCloudPublisher(Node):
    def __init__(self, name: str = "iv_points_publisher", topic: str = "/iv_points21"):
        super().__init__(name)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.publisher_ = self.create_publisher(PointCloud2, topic, qos)

    def publish(self, points: o3d.geometry.PointCloud):
        self.publisher_.publish(open3d_to_pointcloud2(points))
