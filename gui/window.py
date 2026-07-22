"""PySide6 main window for the stacking-planner execution GUI.

The window owns the status panel, prompt label, decision buttons and the log
view.  It is decoupled from the planning pipeline via Qt signals:

* The worker emits :pyattr:`request_decision`, :pyattr:`display_geometries`,
  :pyattr:`status`, :pyattr:`log`, :pyattr:`finished`, :pyattr:`failed`.
* The window emits :pyattr:`decided` and :pyattr:`abort_requested` in
  response to button clicks.
"""

from typing import Optional

from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .viewer import Open3DViewer


class MainWindow(QMainWindow):
    """Control panel that drives the planning worker."""

    decided = Signal(str)
    abort_requested = Signal()
    stone_pcd_mode_changed = Signal(bool)
    stone_mesh_mode_changed = Signal(bool)
    pick_grasp_adjust_requested = Signal(str, float)
    pick_grasp_adjust_reset_requested = Signal()

    # Maps a decision ``kind`` (as emitted by the worker) to (prompt text,
    # mapping of decision-id -> button label).  Buttons not listed are hidden.
    PROMPTS = {
        "plan_review": (
            "Review the motion plan in the 3D viewer.",
            {"continue": "Continue", "retry": "Retry"},
        ),
        "replan": (
            "How should the identified in-hand pose be handled?",
            {"continue": "Direct", "retry": "Regrasp", "skip": "Skip"},
        ),
        "replan_review": (
            "Review the replanned trajectory.",
            {"continue": "Use Replan", "retry": "Retry", "skip": "Original"},
        ),
        "press_enter": (
            "Ready to execute the place trajectory.",
            {"continue": "Continue"},
        ),
        "place_control_review": (
            "Ready to execute the place trajectory.",
            {"continue": "Place", "retry": "Retry Motion"},
        ),
        "place_retry_review": (
            "Review retried place trajectory.",
            {"continue": "Accept", "retry": "Retry", "skip": "Original"},
        ),
        "grasp_confirm": (
            "Ready to execute grasp control.",
            {"continue": "Confirm"},
        ),
        "pose_identification": (
            "Run in-hand pose identification before placing?",
            {"continue": "Run Pose ID", "skip": "Skip"},
        ),
        "scene_icp_request": (
            "Run scene pose ICP identification?",
            {"continue": "Run ICP", "retry": "Manual Init", "skip": "Skip"},
        ),
        "scene_pose_review": (
            "Accept the identified scene poses?",
            {"continue": "Accept", "retry": "Retry ICP", "skip": "Keep Current"},
        ),
    }

    def __init__(self, viewer: Optional[Open3DViewer] = None):
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("Excavator Stacking Operation")
        self.resize(640, 760)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Status group
        status_box = QGroupBox("Status")
        form = QFormLayout()
        self.step_lbl = QLabel("—")
        self.stone_lbl = QLabel("—")
        self.phase_lbl = QLabel("Idle")
        big = QFont()
        big.setPointSize(12)
        for w in (self.step_lbl, self.stone_lbl, self.phase_lbl):
            w.setFont(big)
        form.addRow("Step:", self.step_lbl)
        form.addRow("Stone ID:", self.stone_lbl)
        form.addRow("Phase:", self.phase_lbl)
        self.stone_pcd_checkbox = QCheckBox("Render stones as point clouds")
        form.addRow("Visualization:", self.stone_pcd_checkbox)
        self.stone_mesh_checkbox = QCheckBox("Render stones as high-poly meshes")
        form.addRow("", self.stone_mesh_checkbox)
        status_box.setLayout(form)
        layout.addWidget(status_box)

        # Prompt
        self.prompt_lbl = QLabel("Awaiting start...")
        prompt_font = QFont()
        prompt_font.setBold(True)
        prompt_font.setPointSize(11)
        self.prompt_lbl.setFont(prompt_font)
        self.prompt_lbl.setWordWrap(True)
        self.prompt_lbl.setStyleSheet(
            "padding: 8px; background: #f5f5f5; border: 1px solid #ccc;"
        )
        layout.addWidget(self.prompt_lbl)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_continue = QPushButton("Continue")
        self.btn_retry = QPushButton("Retry")
        self.btn_skip = QPushButton("Skip")
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setStyleSheet("color: white; background: #c0392b;")
        for b in (self.btn_continue, self.btn_retry, self.btn_skip):
            b.setEnabled(False)
        for b in (self.btn_continue, self.btn_retry, self.btn_skip, self.btn_abort):
            b.setMinimumHeight(40)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self.btn_continue.clicked.connect(lambda: self._decide("continue"))
        self.btn_retry.clicked.connect(lambda: self._decide("retry"))
        self.btn_skip.clicked.connect(lambda: self._decide("skip"))
        self.btn_abort.clicked.connect(self._abort)
        self.stone_pcd_checkbox.toggled.connect(self.stone_pcd_mode_changed.emit)
        self.stone_mesh_checkbox.toggled.connect(self.stone_mesh_mode_changed.emit)

        # Grasp adjustment
        adjust_box = QGroupBox("Grasp Adjustment")
        adjust_layout = QVBoxLayout()
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (deg)"))
        self.adjust_step_spin = QDoubleSpinBox()
        self.adjust_step_spin.setRange(0.1, 5.0)
        self.adjust_step_spin.setSingleStep(0.1)
        self.adjust_step_spin.setValue(0.5)
        step_row.addWidget(self.adjust_step_spin)
        adjust_layout.addLayout(step_row)

        swing_row = QHBoxLayout()
        self.swing_minus_btn = QPushButton("Swing -")
        self.swing_plus_btn = QPushButton("Swing +")
        swing_row.addWidget(self.swing_minus_btn)
        swing_row.addWidget(self.swing_plus_btn)
        adjust_layout.addLayout(swing_row)

        boom_row = QHBoxLayout()
        self.boom_minus_btn = QPushButton("Boom -")
        self.boom_plus_btn = QPushButton("Boom +")
        boom_row.addWidget(self.boom_minus_btn)
        boom_row.addWidget(self.boom_plus_btn)
        adjust_layout.addLayout(boom_row)

        bucket_row = QHBoxLayout()
        self.bucket_minus_btn = QPushButton("Bucket -")
        self.bucket_plus_btn = QPushButton("Bucket +")
        bucket_row.addWidget(self.bucket_minus_btn)
        bucket_row.addWidget(self.bucket_plus_btn)
        adjust_layout.addLayout(bucket_row)

        self.adjust_reset_btn = QPushButton("Reset offsets")
        adjust_layout.addWidget(self.adjust_reset_btn)
        self.adjust_offset_lbl = QLabel("Swing 0.0 deg, Boom 0.0 deg, Bucket 0.0 deg")
        adjust_layout.addWidget(self.adjust_offset_lbl)
        adjust_box.setLayout(adjust_layout)
        layout.addWidget(adjust_box)

        self._pick_adjust_enabled = False
        self._swing_offset_deg = 0.0
        self._boom_offset_deg = 0.0
        self._bucket_offset_deg = 0.0
        self._set_pick_adjust_enabled(False)
        self.swing_minus_btn.clicked.connect(lambda: self._adjust_pick("swing", -1.0))
        self.swing_plus_btn.clicked.connect(lambda: self._adjust_pick("swing", 1.0))
        self.boom_minus_btn.clicked.connect(lambda: self._adjust_pick("boom", -1.0))
        self.boom_plus_btn.clicked.connect(lambda: self._adjust_pick("boom", 1.0))
        self.bucket_minus_btn.clicked.connect(
            lambda: self._adjust_pick("bucket", -1.0)
        )
        self.bucket_plus_btn.clicked.connect(lambda: self._adjust_pick("bucket", 1.0))
        self.adjust_reset_btn.clicked.connect(self._reset_pick_adjust)

        # Log
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)
        self.log_view.setFont(mono)
        log_layout.addWidget(self.log_view)
        log_box.setLayout(log_layout)
        layout.addWidget(log_box, stretch=1)

        self.setStatusBar(QStatusBar(self))

    # --- internal -----------------------------------------------------------
    def _set_buttons(self, enabled):
        self.btn_continue.setEnabled("continue" in enabled)
        self.btn_retry.setEnabled("retry" in enabled)
        self.btn_skip.setEnabled("skip" in enabled)

    def _set_button_labels(self, labels):
        self.btn_continue.setText(labels.get("continue", "Continue"))
        self.btn_retry.setText(labels.get("retry", "Retry"))
        self.btn_skip.setText(labels.get("skip", "Skip"))

    def _decide(self, value: str):
        self._set_buttons(set())
        self._set_pick_adjust_enabled(False)
        self.prompt_lbl.setText("Working...")
        self.decided.emit(value)

    def _abort(self):
        self.append_log("Abort requested.")
        self.abort_requested.emit()
        self._set_buttons(set())
        self._set_pick_adjust_enabled(False)
        self.btn_abort.setEnabled(False)

    def _set_pick_adjust_enabled(self, enabled: bool):
        self._pick_adjust_enabled = enabled
        for button in (
            self.swing_minus_btn,
            self.swing_plus_btn,
            self.boom_minus_btn,
            self.boom_plus_btn,
            self.bucket_minus_btn,
            self.bucket_plus_btn,
            self.adjust_reset_btn,
        ):
            button.setEnabled(enabled)
        self.adjust_step_spin.setEnabled(enabled)

    def _adjust_pick(self, joint: str, sign: float):
        if not self._pick_adjust_enabled:
            return
        delta = sign * self.adjust_step_spin.value()
        if joint == "swing":
            self._swing_offset_deg += delta
        elif joint == "boom":
            self._boom_offset_deg += delta
        elif joint == "bucket":
            self._bucket_offset_deg += delta
        self._update_adjust_label()
        self.pick_grasp_adjust_requested.emit(joint, delta)

    def _reset_pick_adjust(self):
        self._swing_offset_deg = 0.0
        self._boom_offset_deg = 0.0
        self._bucket_offset_deg = 0.0
        self._update_adjust_label()
        self.pick_grasp_adjust_reset_requested.emit()

    def _update_adjust_label(self):
        self.adjust_offset_lbl.setText(
            f"Swing {self._swing_offset_deg:.1f} deg, "
            f"Boom {self._boom_offset_deg:.1f} deg, "
            f"Bucket {self._bucket_offset_deg:.1f} deg"
        )

    # --- slots --------------------------------------------------------------
    @Slot(str)
    def append_log(self, msg: str):
        self.log_view.appendPlainText(msg)

    @Slot(dict)
    def update_status(self, st: dict):
        if "step" in st and "total" in st:
            self.step_lbl.setText(f"{st['step']} / {st['total']}")
        if "stone_id" in st:
            self.stone_lbl.setText(str(st["stone_id"]))
        if "phase" in st:
            self.phase_lbl.setText(st["phase"])
            self.statusBar().showMessage(st["phase"])

    @Slot(str, dict)
    def on_request_decision(self, kind: str, payload: dict):
        text, labels = self.PROMPTS.get(
            kind, (f"Decision needed: {kind}", {"continue": "Continue"})
        )
        if "step" in payload:
            text = f"[Step {payload['step']}] {text}"
        if kind == "grasp_confirm":
            phase = payload.get("phase")
            if phase == "place":
                labels = {"continue": "Release", "skip": "Finish"}
            if phase == "pick_approach":
                text = "Ready to execute pick approach."
            elif phase == "pick":
                text = "Ready to execute pick grasp control."
            elif phase == "place_approach":
                text = "Ready to execute place approach."
            elif phase == "place":
                text = "Ready to release the stone."
        elif kind == "replan_review" and "candidate" in payload:
            text = (
                "Review replanned trajectory "
                f"{payload['candidate']}/{payload.get('num_candidates', '?')}."
            )
        elif kind == "place_retry_review" and "candidate" in payload:
            text = (
                "Review retried place trajectory "
                f"{payload['candidate']}/{payload.get('num_candidates', '?')}."
            )
        elif kind == "scene_pose_review":
            details = []
            num_poses = payload.get("num_poses")
            ground_height = payload.get("ground_height")
            if num_poses is not None:
                details.append(f"{num_poses} poses")
            if ground_height is not None:
                details.append(f"ground z={float(ground_height):.3f} m")
            if details:
                text = text + " (" + ", ".join(details) + ")"
        self.prompt_lbl.setText(text)
        self._set_button_labels(labels)
        self._set_buttons(set(labels.keys()))
        is_adjustable_grasp = kind == "grasp_confirm" and payload.get("phase") in (
            "pick_approach",
            "pick",
            "place",
            "place_approach",
        )
        self._set_pick_adjust_enabled(is_adjustable_grasp)
        if is_adjustable_grasp:
            self._swing_offset_deg = 0.0
            self._boom_offset_deg = 0.0
            self._bucket_offset_deg = 0.0
            self._update_adjust_label()

    @Slot(list, str)
    def on_display_geometries(self, geoms: list, title: str):
        if self.viewer is not None:
            self.viewer.set_geometries(geoms)

    @Slot()
    def on_finished(self):
        self.prompt_lbl.setText("Execution complete.")
        self._set_buttons(set())
        self.btn_abort.setEnabled(False)
        self.append_log("[DONE]")

    @Slot(str)
    def on_failed(self, msg: str):
        self.prompt_lbl.setText("Execution failed — see log.")
        self.append_log(f"[FAILED]\n{msg}")
        self._set_buttons(set())
