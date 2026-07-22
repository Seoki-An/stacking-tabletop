#!/usr/bin/env python3
import numpy as np
from typing import Union

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)
from tp_msgs.msg import CtrlJoint
from std_msgs.msg import Float64

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

GRAB_STATE = {"open": 2000, "close": -2000, "stay": 0}


class CtrlJointSubscriber(Node):
    def __init__(
        self, name: str = "ctrl_joint_node_subscriber", topic: str = "/joint_rcv"
    ):
        super().__init__(name)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        # qos = qos_profile_sensor_data

        self.subscription_ = self.create_subscription(
            CtrlJoint, topic, self.listener_callback, qos
        )
        self.last_msg = None
        self.pos = np.zeros(6)
        self.vel = np.zeros(6)
        self._get_flag = False

    def listener_callback(self, msg: CtrlJoint):
        self.last_msg = msg
        self.pos = DEG2RAD * np.array(
            [msg.swpos, msg.bmpos, msg.ampos, msg.bktrpos, -msg.tltpos, -msg.rotpos]
        )
        self.pos[0] = project_angle(self.pos[0])
        self.pos[5] = project_angle(self.pos[5])
        self.vel = DEG2RAD * np.array(
            [msg.swvel, msg.bmvel, msg.amvel, msg.bktrvel, -msg.tltvel, -msg.rotvel]
        )
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class CtrlJointPublisher(Node):
    def __init__(
        self, name: str = "ctrl_joint_node_publisher", topic: str = "/joint_ctrl"
    ):
        super().__init__(name)
        self.publisher_ = self.create_publisher(CtrlJoint, topic, 10)

    def publish(
        self, pos: np.ndarray, vel: np.ndarray = np.zeros(6), grab: str = "stay"
    ):
        """
        input: joint position np.ndarray(6), joint velocity np.ndarray(6)
        unit: [rad], [rad/s]
        """
        pos[0] = project_angle(pos[0])
        pos[5] = project_angle(pos[5])

        pos = pos.astype(np.float64) * RAD2DEG
        vel = vel.astype(np.float64) * RAD2DEG

        msg = CtrlJoint()
        msg.swpos = pos[0]  # swing joint
        msg.bmpos = pos[1]  # boom joint
        msg.ampos = pos[2]  # arm joint
        msg.bktrpos = pos[3]  # bucket joint
        msg.tltpos = -pos[4]  # tilt joint
        msg.rotpos = -pos[5]  # rotate joint

        msg.swvel = vel[0]  # swing joint
        msg.bmvel = vel[1]  # boom joint
        msg.amvel = vel[2]  # arm joint
        msg.bktrvel = vel[3]  # bucket joint
        msg.tltvel = -vel[4]  # tilt joint
        msg.rotvel = -vel[5]  # rotate joint

        msg.grab = float(GRAB_STATE[grab])

        self.publisher_.publish(msg)


class OpeningAngleSubscriber(Node):

    def __init__(
        self, name: str = "opening_angle_subscriber", topic: str = "opening_angle"
    ):
        super().__init__(name)

        self.sub = self.create_subscription(Float64, topic, self.callback, 10)
        self.data = 0.0
        self._get_flag = False

    def callback(self, msg):
        self.data = msg.data
        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class OpeningAnglePublisher(Node):

    def __init__(
        self, name: str = "opening_angle_publisher", topic: str = "opening_angle"
    ):
        super().__init__(name)

        self.pub = self.create_publisher(Float64, topic, 10)

        msg = Float64()
        msg.data = 0.0

    def publish(self, opening_angle: float):
        msg = Float64()
        msg.data = opening_angle
        self.pub.publish(msg)


def project_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (angle + np.pi) % (2 * np.pi) - np.pi
