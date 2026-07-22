#!/usr/bin/env python3

from rclpy.node import Node
from std_msgs.msg import String


class FieldRecoveryStatusPublisher(Node):
    def __init__(
        self,
        name: str = "field_recovery_status_publisher",
        topic: str = "/field_recovery_status",
    ):
        super().__init__(name)
        self.publisher_ = self.create_publisher(String, topic, 10)

    def publish(self, status: str, stone_id: int = -1, detail: str = ""):
        msg = String()
        msg.data = f"{status}|{int(stone_id)}|{detail}"
        self.publisher_.publish(msg)


class FieldRecoveryStatusSubscriber(Node):
    def __init__(
        self,
        name: str = "field_recovery_status_subscriber",
        topic: str = "/field_recovery_status",
    ):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            String,
            topic,
            self._callback,
            10,
        )
        self.status = ""
        self.stone_id = -1
        self.detail = ""
        self._get_flag = False

    def _callback(self, msg: String):
        parts = str(msg.data).split("|", 2)
        self.status = parts[0] if parts else ""
        try:
            self.stone_id = int(parts[1]) if len(parts) > 1 else -1
        except ValueError:
            self.stone_id = -1
        self.detail = parts[2] if len(parts) > 2 else ""
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag
