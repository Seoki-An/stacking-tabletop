#!/usr/bin/env python3

"""Interactive UR5e visual-model and forward-kinematics viewer."""

from __future__ import annotations

import sys

import numpy as np
import open3d as o3d
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui.viewer import Open3DViewer
from model import get_ur5e_model, update_urdf_mesh


JOINTS = [
    ("Shoulder pan", 0, -360.0, 360.0),
    ("Shoulder lift", 1, -360.0, 360.0),
    ("Elbow", 2, -180.0, 180.0),
    ("Wrist 1", 3, -360.0, 360.0),
    ("Wrist 2", 4, -360.0, 360.0),
    ("Wrist 3", 5, -360.0, 360.0),
    ("Gripper", 6, 0.0, np.rad2deg(0.9)),
]

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "sr_gripper_left_finger_joint_1",
    "sr_gripper_left_finger_joint_2",
    "sr_gripper_right_finger_joint_1",
    "sr_gripper_right_finger_joint_2",
]

LINK_NAMES = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "flange",
    "tool0",
    "sr_gripper_base_link",
    "sr_gripper_left_finger_link_1",
    "sr_gripper_left_finger_link_2",
    "sr_gripper_right_finger_link_1",
    "sr_gripper_right_finger_link_2",
]

Q_HOME = np.concatenate(
    [np.deg2rad([0.0, -90.0, 0.0, -90.0, 0.0, 0.0]), np.zeros(4)]
)

C_UR5E = [0.75, 0.75, 0.78]
C_GRIPPER = [0.20, 0.35, 0.40]
C_FRAME = [0.1, 0.1, 0.1]
FRAME_SIZE = 0.08


class JointControl:
    def __init__(self, name, idx, min_deg, max_deg, on_change):
        self.name = name
        self.idx = idx
        self.scale = 10
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_deg * self.scale), int(max_deg * self.scale))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(min_deg, max_deg)
        self.spin.setDecimals(1)
        self.spin.setSingleStep(0.5)
        self.spin.setSuffix(" deg")
        self.value_label = QLabel("0.000 rad")
        self.value_label.setMinimumWidth(90)
        self._on_change = on_change

        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def set_deg(self, deg):
        deg = float(np.clip(deg, self.spin.minimum(), self.spin.maximum()))
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(int(round(deg * self.scale)))
        self.spin.setValue(deg)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
        self.value_label.setText(f"{np.deg2rad(deg): .3f} rad")

    def deg(self):
        return self.spin.value()

    def _slider_changed(self, value):
        deg = value / self.scale
        self.spin.blockSignals(True)
        self.spin.setValue(deg)
        self.spin.blockSignals(False)
        self.value_label.setText(f"{np.deg2rad(deg): .3f} rad")
        self._on_change()

    def _spin_changed(self, deg):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(deg * self.scale)))
        self.slider.blockSignals(False)
        self.value_label.setText(f"{np.deg2rad(deg): .3f} rad")
        self._on_change()


class UR5eModelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UR5e Model Joint Viewer")
        self.resize(760, 470)

        self.viewer = Open3DViewer("UR5e Model", initial_zoom=0.55)
        self.ur5e_model, self.ur5e_meshes = get_ur5e_model()
        self.q = Q_HOME.copy()

        self.origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.12, origin=[0, 0, 0]
        )

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        preset_row = QHBoxLayout()
        self.zero_btn = QPushButton("Zero")
        self.home_btn = QPushButton("Home")
        for button in (self.zero_btn, self.home_btn):
            button.setMinimumHeight(34)
            preset_row.addWidget(button)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        options_row = QHBoxLayout()
        self.frames_checkbox = QCheckBox("Show joint/link frames")
        self.frames_checkbox.setChecked(True)
        options_row.addWidget(self.frames_checkbox)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        joint_box = QGroupBox("Joint Angles")
        grid = QGridLayout()
        self.controls = []
        for row, spec in enumerate(JOINTS):
            control = JointControl(*spec, on_change=self._sync_from_controls)
            self.controls.append(control)
            grid.addWidget(QLabel(control.name), row, 0)
            grid.addWidget(control.slider, row, 1)
            grid.addWidget(control.spin, row, 2)
            grid.addWidget(control.value_label, row, 3)
        joint_box.setLayout(grid)
        layout.addWidget(joint_box)

        self.q_label = QLabel()
        self.q_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.q_label.setWordWrap(True)
        layout.addWidget(self.q_label)

        self.zero_btn.clicked.connect(lambda: self._set_q(np.zeros(10)))
        self.home_btn.clicked.connect(lambda: self._set_q(Q_HOME))
        self.frames_checkbox.toggled.connect(lambda *_: self._update_viewer())

        self._set_q(self.q)

    def _set_q(self, q):
        self.q = np.asarray(q, dtype=np.float64).copy()
        for control in self.controls:
            control.set_deg(np.rad2deg(self.q[control.idx]))
        self.q[6:10] = [self.q[6], -self.q[6], self.q[6], -self.q[6]]
        self._update_viewer()

    def _sync_from_controls(self):
        for control in self.controls:
            self.q[control.idx] = np.deg2rad(control.deg())
        self.q[6:10] = [self.q[6], -self.q[6], self.q[6], -self.q[6]]
        self._update_viewer()

    def _update_viewer(self):
        meshes = update_urdf_mesh(self.ur5e_model, self.ur5e_meshes, self.q.copy())
        geoms = [
            (
                mesh,
                C_GRIPPER if name.startswith("sr_gripper_") else C_UR5E,
            )
            for name, mesh in meshes.items()
        ]
        geoms.append(self.origin)

        if self.frames_checkbox.isChecked():
            geoms += [(frame, C_FRAME) for frame in self._joint_frames()]
            geoms += [(frame, C_FRAME) for frame in self._link_frames()]

        self.viewer.set_geometries(geoms)
        self.q_label.setText(
            "q rad: "
            + np.array2string(self.q, precision=4, suppress_small=True)
            + "\nq deg: "
            + np.array2string(np.rad2deg(self.q), precision=1, suppress_small=True)
        )

    def _joint_frames(self):
        frames = []
        for name in JOINT_NAMES:
            joint = self.ur5e_model.GetJoint(name)
            joint_child = joint.child
            if joint_child is None:
                continue

            joint_pose = np.eye(4)
            joint_pose[:3, :3] = joint_child.GetRotation()
            joint_pose[:3, -1] = joint_child.GetPosition()

            joint_pose_local = np.eye(4)
            joint_pose_local[:3, :3] = joint.GetRotation()
            joint_pose_local[:3, -1] = joint.GetPosition()
            joint_pose = joint_pose @ np.linalg.inv(joint_pose_local)

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=FRAME_SIZE
            )
            frame.transform(joint_pose)
            frames.append(frame)
        return frames

    def _link_frames(self):
        return [self._link_frame(name) for name in LINK_NAMES]

    def _link_frame(self, link_name):
        link = self.ur5e_model.GetLink(link_name)
        pose = np.eye(4)
        pose[:3, :3] = link.GetRotation()
        pose[:3, -1] = link.GetPosition()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=FRAME_SIZE)
        frame.transform(pose)
        return frame

    def closeEvent(self, event):
        self.viewer.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = UR5eModelWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
