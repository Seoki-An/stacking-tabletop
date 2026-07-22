#!/usr/bin/env python3

import json
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from std_msgs.msg import String


def _reliable_qos() -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
    )


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class SceneScanDonePublisher(Node):
    def __init__(
        self, name: str = "scene_scan_done_publisher", topic: str = "/scene_scan_done"
    ):
        super().__init__(name)
        self.publisher_ = self.create_publisher(String, topic, _latched_qos())

    def publish(self, step_log_dir: str):
        msg = String()
        msg.data = str(step_log_dir)
        self.publisher_.publish(msg)


class SceneScanDoneSubscriber(Node):
    def __init__(
        self, name: str = "scene_scan_done_subscriber", topic: str = "/scene_scan_done"
    ):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            String, topic, self._callback, _latched_qos()
        )
        self.step_log_dir: str = ""
        self._get_flag = False

    def _callback(self, msg: String):
        self.step_log_dir = msg.data
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class DiagnosticPcdRequestPublisher(Node):
    def __init__(
        self,
        name: str = "diagnostic_pcd_request_publisher",
        topic: str = "/diagnostic_pcd_request",
    ):
        super().__init__(name)
        self.publisher_ = self.create_publisher(String, topic, _reliable_qos())

    def publish(self, log_dir: str, label: str):
        msg = String()
        msg.data = json.dumps({"log_dir": str(log_dir), "label": str(label)})
        self.publisher_.publish(msg)


class DiagnosticPcdRequestSubscriber(Node):
    def __init__(
        self,
        name: str = "diagnostic_pcd_request_subscriber",
        topic: str = "/diagnostic_pcd_request",
    ):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            String, topic, self._callback, _reliable_qos()
        )
        self.log_dir: str = ""
        self.label: str = ""
        self._get_flag = False

    def _callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {"log_dir": msg.data, "label": ""}
        self.log_dir = str(payload.get("log_dir", ""))
        self.label = str(payload.get("label", ""))
        if self.log_dir:
            self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class SceneICPRequestPublisher(Node):
    def __init__(
        self,
        name: str = "scene_icp_request_publisher",
        topic: str = "/scene_icp_request",
    ):
        super().__init__(name)
        self.publisher_ = self.create_publisher(String, topic, _latched_qos())

    def publish(self, step_log_dir: str, run_icp: bool):
        msg = String()
        decision = "run" if run_icp else "skip"
        msg.data = f"{step_log_dir}|{decision}"
        self.publisher_.publish(msg)


class SceneICPRequestSubscriber(Node):
    def __init__(
        self,
        name: str = "scene_icp_request_subscriber",
        topic: str = "/scene_icp_request",
    ):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            String, topic, self._callback, _latched_qos()
        )
        self.step_log_dir: str = ""
        self.run_icp: bool | None = None
        self._get_flag = False

    def _callback(self, msg: String):
        path, sep, decision = msg.data.rpartition("|")
        if not sep:
            self.step_log_dir = msg.data
            self.run_icp = None
        else:
            self.step_log_dir = path
            self.run_icp = decision == "run"
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class SceneICPResultPublisher(Node):
    def __init__(
        self,
        name: str = "scene_icp_result_publisher",
        topic: str = "/scene_icp_result",
    ):
        super().__init__(name)
        self.publisher_ = self.create_publisher(String, topic, _latched_qos())

    def publish(self, step_log_dir: str, status: str, detail: str = ""):
        msg = String()
        msg.data = f"{step_log_dir}|{status}|{detail}"
        self.publisher_.publish(msg)


class SceneICPResultSubscriber(Node):
    def __init__(
        self,
        name: str = "scene_icp_result_subscriber",
        topic: str = "/scene_icp_result",
    ):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            String, topic, self._callback, _latched_qos()
        )
        self.step_log_dir: str = ""
        self.status: str = ""
        self.detail: str = ""
        self._get_flag = False

    def _callback(self, msg: String):
        parts = msg.data.split("|", 2)
        self.step_log_dir = parts[0] if len(parts) >= 1 else ""
        self.status = parts[1] if len(parts) >= 2 else ""
        self.detail = parts[2] if len(parts) >= 3 else ""
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag
