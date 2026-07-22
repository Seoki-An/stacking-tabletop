#!/usr/bin/env python3
"""Scene-identification PCD pair for the constrained NUC -> desktop WiFi link.

Scene identification runs on the desktop, but the scene cloud is captured on
the NUC. The LiDAR-facing ``PointCloudSubscriber`` in ``pcd_node`` keeps
reading the raw sensor cloud over the wired connection; this pair takes the
resulting Nx3 array on the NUC, shrinks it (voxel downsample + int16
quantization, ~4-10x smaller), and ships it over WiFi as a standard
``PointCloud2`` so the desktop side decodes back to an Nx3 float32 array.

``SceneIdentifier`` is split into a publisher (NUC) and a subscriber (desktop)
because it spans two machines, unlike the single-machine ``PoseIdentifier``.
"""
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Float32MultiArray, Bool, MultiArrayDimension

# 1 mm resolution; int16 range (+-32767) -> +-32.7 m about the frame origin.
DEFAULT_SCALE = 0.001
# 5 mm voxel; set to None/0 to disable downsampling.
DEFAULT_VOXEL = 0.005


def _crop_aabb(points: np.ndarray, crop_min, crop_max) -> np.ndarray:
    """Keep only points inside the axis-aligned box [crop_min, crop_max].

    Either bound may be None (unbounded on that side). Done first, before
    downsampling, since dropping out-of-region points is the biggest
    bandwidth win for scene identification (we only need the cloud around
    the target structure).
    """
    if crop_min is None and crop_max is None:
        return points
    lo = np.full(3, -np.inf) if crop_min is None else np.asarray(crop_min, np.float32)
    hi = np.full(3, np.inf) if crop_max is None else np.asarray(crop_max, np.float32)
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    return points[mask]


def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if not voxel:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points, dtype=np.float32)


def encode_pointcloud2(
    points: np.ndarray,
    frame_id: str = "map",
    scale: float = DEFAULT_SCALE,
    voxel: float = DEFAULT_VOXEL,
    crop_min=None,
    crop_max=None,
    stamp=None,
) -> PointCloud2:
    """Nx3 float32 -> cropped, downsampled, int16-quantized PointCloud2."""
    points = np.asarray(points, dtype=np.float32)
    points = _crop_aabb(points, crop_min, crop_max)
    points = _voxel_downsample(points, voxel)

    header = Header()
    header.stamp = stamp if stamp is not None else rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id

    q = np.round(points / scale).astype(np.int16)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.INT16, count=1),
        PointField(name="y", offset=2, datatype=PointField.INT16, count=1),
        PointField(name="z", offset=4, datatype=PointField.INT16, count=1),
    ]
    return point_cloud2.create_cloud(header, fields, q)


def decode_pointcloud2(msg: PointCloud2, scale: float = DEFAULT_SCALE) -> np.ndarray:
    """int16-quantized PointCloud2 -> Nx3 float32 in metric units."""
    q = point_cloud2.read_points_numpy(
        msg, field_names=["x", "y", "z"], skip_nans=True
    )
    return q.astype(np.float32) * scale


class SceneIdentifierPublisher(Node):
    """NUC side: publish the (downsampled, quantized) scene cloud over WiFi."""

    def __init__(
        self,
        name: str = "scene_identifier_publisher",
        topic: str = "/scene_pcd",
        scale: float = DEFAULT_SCALE,
        voxel: float = DEFAULT_VOXEL,
        crop_min=None,
        crop_max=None,
    ):
        super().__init__(name)
        self.scale = scale
        self.voxel = voxel
        # Default region around the target structure; crop here on the NUC so
        # only the relevant cloud crosses WiFi. Override per-publish if needed.
        self.crop_min = crop_min
        self.crop_max = crop_max

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.publisher_ = self.create_publisher(PointCloud2, topic, qos)

    def publish(
        self, points, frame_id: str = "map", stamp=None, crop_min=None, crop_max=None
    ):
        """points: Nx3 np.array or o3d.geometry.PointCloud."""
        if isinstance(points, o3d.geometry.PointCloud):
            points = np.asarray(points.points, dtype=np.float32)
        msg = encode_pointcloud2(
            points,
            frame_id=frame_id,
            scale=self.scale,
            voxel=self.voxel,
            crop_min=self.crop_min if crop_min is None else crop_min,
            crop_max=self.crop_max if crop_max is None else crop_max,
            stamp=stamp,
        )
        self.publisher_.publish(msg)


class SceneIdentifierSubscriber(Node):
    """Desktop side: receive and decode the scene cloud into ``.points`` (Nx3)."""

    def __init__(
        self,
        name: str = "scene_identifier_subscriber",
        topic: str = "/scene_pcd",
        scale: float = DEFAULT_SCALE,
    ):
        super().__init__(name)
        self.scale = scale

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
            self.points = decode_pointcloud2(msg, scale=self.scale)
        except Exception as e:
            self.get_logger().error(f"PointCloud2 decode error: {e}")
            return

        if self.points.size == 0:
            self.get_logger().warning("Received empty point cloud")
            return
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


def _request_qos() -> QoSProfile:
    # Latched so a NUC that subscribes after the desktop has already published
    # still receives the latest scan request.
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class SceneScanRequestPublisher(Node):
    """Desktop side: ask the NUC to scan around a target structure region.

    Payload is the target structure's axis-aligned region as a center xyz and
    a half-extent xyz (6 floats). The NUC uses the center to aim the scan and
    the region to crop the cloud before shipping it back.
    """

    def __init__(
        self,
        name: str = "scene_scan_request_publisher",
        topic: str = "/scene_scan_request",
    ):
        super().__init__(name)
        self.publisher_ = self.create_publisher(
            Float32MultiArray, topic, _request_qos()
        )

    def publish(self, center, half_extent):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in (*center, *half_extent)]
        self.publisher_.publish(msg)


class SceneScanRequestSubscriber(Node):
    """NUC side: receive a scan request (target center + half-extent)."""

    def __init__(
        self,
        name: str = "scene_scan_request_subscriber",
        topic: str = "/scene_scan_request",
    ):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            Float32MultiArray, topic, self._callback, _request_qos()
        )
        self.center = None
        self.half_extent = None
        self._get_flag = False

    def _callback(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 6:
            self.get_logger().warning(
                f"Malformed scan request: expected 6 floats, got {data.size}"
            )
            return
        self.center = data[:3]
        self.half_extent = np.abs(data[3:6])
        self._get_flag = True

    @property
    def target_xy(self):
        return None if self.center is None else self.center[:2]

    @property
    def crop_min(self):
        if self.center is None:
            return None
        return self.center - self.half_extent

    @property
    def crop_max(self):
        if self.center is None:
            return None
        return self.center + self.half_extent

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class SceneScanDonePublisher(Node):
    """NUC side: signal that a scan finished (no more frames coming).

    Non-latched so the desktop only ever sees the signal for the scan it is
    currently collecting, never a stale one from a previous run.
    """

    def __init__(
        self,
        name: str = "scene_scan_done_publisher",
        topic: str = "/scene_pcd_done",
    ):
        super().__init__(name)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.publisher_ = self.create_publisher(Bool, topic, qos)

    def publish(self):
        msg = Bool()
        msg.data = True
        self.publisher_.publish(msg)


class SceneScanDoneSubscriber(Node):
    """Desktop side: receive the scan-finished signal."""

    def __init__(
        self,
        name: str = "scene_scan_done_subscriber",
        topic: str = "/scene_pcd_done",
    ):
        super().__init__(name)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.subscription_ = self.create_subscription(
            Bool, topic, self._callback, qos
        )
        self._get_flag = False

    def _callback(self, msg: Bool):
        if msg.data:
            self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class JointLogPublisher(Node):
    """Publish a batch joint log (one row per scan pose) to the desktop."""

    def __init__(
        self,
        name: str = "joint_log_publisher",
        topic: str = "/field_joint_log",
    ):
        super().__init__(name)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.publisher_ = self.create_publisher(Float32MultiArray, topic, qos)

    def publish(self, q_rows):
        """q_rows: list/array of joint configs, shape (num_poses, num_joints)."""
        rows = np.asarray(q_rows, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        msg = Float32MultiArray()
        # Carry the column count so the desktop can reshape robustly.
        msg.layout.dim = [
            MultiArrayDimension(
                label="cols", size=int(rows.shape[1]), stride=int(rows.shape[1])
            )
        ]
        msg.data = rows.reshape(-1).tolist()
        self.publisher_.publish(msg)


class JointLogSubscriber(Node):
    """Receive a batch joint log into ``.rows`` (num_poses x num_joints)."""

    def __init__(
        self,
        name: str = "joint_log_subscriber",
        topic: str = "/field_joint_log",
    ):
        super().__init__(name)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.subscription_ = self.create_subscription(
            Float32MultiArray, topic, self._callback, qos
        )
        self.rows = None
        self._get_flag = False

    def _callback(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        cols = msg.layout.dim[0].size if msg.layout.dim else 0
        if cols and data.size % cols == 0:
            self.rows = data.reshape(-1, cols)
        else:
            self.rows = data.reshape(1, -1)
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag
