#!/usr/bin/env python3

from rclpy.node import Node
from tp_msgs.msg import Phase


class PhaseSubscriber(Node):
    def __init__(self, name: str = "phase_subscriber", topic: str = "/phase"):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            Phase, topic, self.phase_callback, 10
        )
        self.phase = Phase.FIELDSCAN
        self._get_flag = False

    def phase_callback(self, msg: Phase):
        self.phase = msg.phase
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class PhasePublisher(Node):
    def __init__(self, name: str = "phase_publisher", topic: str = "/phase"):
        super().__init__(name)
        self.publisher_ = self.create_publisher(Phase, topic, 10)

    def publish(self, phase: int):
        msg = Phase()
        msg.phase = phase
        self.publisher_.publish(msg)
