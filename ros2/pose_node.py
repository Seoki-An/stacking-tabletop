#!/usr/bin/env python3

from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from rclpy.node import Node

from tp_msgs.msg import PoseWithId, PoseWithIdArray


class PoseSubscriber(Node):
    def __init__(self, name: str = "pose_subscriber", topic: str = "/estimated_pose"):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            PoseWithId,
            topic,
            self.pose_callback,
            10,
        )

        self.pos = np.zeros(3)
        self.quat = np.array([0, 0, 0, 1])  # x, y, z, w
        self.rot = np.eye(3)
        self.id = -1
        self._get_flag = False

    def pose_callback(self, msg: PoseWithId):
        pos = msg.pose.position
        ori = msg.pose.orientation
        self.id = msg.id
        self.pos = np.array([pos.x, pos.y, pos.z])
        self.quat = np.array([ori.x, ori.y, ori.z, ori.w])
        self.rot = Rotation.from_quat(self.quat).as_matrix()
        self._get_flag = True

    def transform_matrix(self):
        mat = np.eye(4)
        mat[:3, -1] = self.pos
        mat[:3, :3] = self.rot
        return mat

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class PosePublisher(Node):
    def __init__(self, name: str = "pose_publisher", topic: str = "/estimated_pose"):
        super().__init__(name)
        self.publisher_ = self.create_publisher(PoseWithId, topic, 10)

    def publish(self, pos: np.ndarray, quat: np.ndarray, id: int):
        """
        :param pos np.ndarray(3): x,y,z position
        :param quat np.ndarray(4): x,y,z,w quaternion (scalar last)
        """
        msg = PoseWithId()

        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]

        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        msg.id = int(id)

        self.publisher_.publish(msg)


class PoseArraySubscriber(Node):
    def __init__(
        self, name: str = "pose_array_subscriber", topic: str = "/initial_pose_array"
    ):
        super().__init__(name)
        self.subscription_ = self.create_subscription(
            PoseWithIdArray,
            topic,
            self.pose_callback,
            10,
        )

        self.pose_id = {}
        self._get_flag = False

    def pose_callback(self, msg: PoseWithIdArray):

        self.pose_id = {}
        for item in msg.items:

            pos = item.pose.position
            ori = item.pose.orientation

            pos = np.array([pos.x, pos.y, pos.z])
            quat = np.array([ori.x, ori.y, ori.z, ori.w])

            self.pose_id[item.id] = (pos, quat)

        self._get_flag = True

    def reset_get_flag(self):
        self._get_flag = False

    def get_flag(self) -> bool:
        return self._get_flag


class PoseArrayPublisher(Node):
    def __init__(
        self, name: str = "pose_array_publisher", topic: str = "/initial_pose_array"
    ):
        super().__init__(name)
        self.publisher_ = self.create_publisher(PoseWithIdArray, topic, 10)

    def publish(self, pose_with_id_list: List[Tuple[np.ndarray, np.ndarray, int]]):
        """
        :param pose_with_id_list List[(pos, quat, id)]
        :param pos np.ndarray(3): x,y,z position
        :param quat np.ndarray(4): x,y,z,w quaternion (scalar last)
        :param id int: index of the object
        """
        msg = PoseWithIdArray()

        for pos, quat, id in pose_with_id_list:
            assert pos.shape == (3,)
            assert quat.shape == (4,)

            item = PoseWithId()

            item.pose.position.x = pos[0]
            item.pose.position.y = pos[1]
            item.pose.position.z = pos[2]

            item.pose.orientation.x = quat[0]
            item.pose.orientation.y = quat[1]
            item.pose.orientation.z = quat[2]
            item.pose.orientation.w = quat[3]

            item.id = id

            msg.items.append(item)

        self.publisher_.publish(msg)
