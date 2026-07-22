#!/usr/bin/env python3

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from std_msgs.msg import String


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class LogDirPublisher(Node):
    def __init__(self, name: str = "log_dir_publisher", topic: str = "/log_dir"):
        super().__init__(name)
        self.publisher_ = self.create_publisher(String, topic, _latched_qos())

    def publish(self, path: str):
        msg = String()
        msg.data = path
        self.publisher_.publish(msg)


class LogDirSubscriber(Node):
    def __init__(self, name: str = "log_dir_subscriber", topic: str = "/log_dir"):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            String, topic, self._callback, _latched_qos()
        )
        self.path: str = ""
        self._get_flag = False

    def _callback(self, msg: String):
        self.path = msg.data
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag
