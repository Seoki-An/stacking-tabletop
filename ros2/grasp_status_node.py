#!/usr/bin/env python3

from rclpy.node import Node
from std_msgs.msg import Bool


class GraspStatusPublisher(Node):
    def __init__(
        self, name: str = "grasp_status_publisher", topic: str = "/grasp_status"
    ):
        super().__init__(name)
        self.publisher_ = self.create_publisher(Bool, topic, 10)

    def publish(self, success: bool):
        msg = Bool()
        msg.data = bool(success)
        self.publisher_.publish(msg)


class GraspStatusSubscriber(Node):
    def __init__(
        self, name: str = "grasp_status_subscriber", topic: str = "/grasp_status"
    ):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            Bool, topic, self._callback, 10
        )
        self.success = None
        self._get_flag = False

    def _callback(self, msg: Bool):
        self.success = bool(msg.data)
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag
