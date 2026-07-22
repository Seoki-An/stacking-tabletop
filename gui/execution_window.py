"""Open3D GUI window for desktop execution.

Styled after candidate_viewer: a left control panel with status labels,
a prompt, action buttons, grasp-adjustment controls, and a scrollable log,
plus a 3-D SceneWidget on the right.

All thread-safe update methods (append_log, update_status, request_decision,
display_geometries, update_live_joint_state, on_finished, on_failed) may be
called from any thread; they schedule the real work on the GUI thread via
post_to_main_thread.

Callback attributes (set by the caller before run()):
    on_decide            (value: str) -> None
    on_abort             () -> None
    on_pick_adjust       (joint: str, delta_deg: float) -> None
    on_pick_adjust_reset () -> None
    on_subgoal_pcd_scan  () -> None
    on_stone_pcd_mode    (enabled: bool) -> None
    on_stone_mesh_mode   (enabled: bool) -> None
    on_lidar_frame_mode  (enabled: bool) -> None
    on_refresh           () -> None
"""

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading

_PANEL_W = 290
_PANEL_MIN_W = 240
_PANEL_MAX_FRAC = 0.70
_SPLITTER_W = 10
_SCENE_EDGE_DRAG_W = 48
_LOG_CAP  = 500   # keep at most this many lines in memory
_LOG_VIEW_CHARS = 180


class ExecutionWindow:

    _PROMPTS = {
        "plan_review":        ("Review the motion plan in the 3D viewer.",   {"continue", "retry"}),
        "online_seed_review": ("Use the planned step or run online MCTS?",    {"continue", "retry"}),
        "replan":             ("How should the identified in-hand pose be handled?", {"continue", "retry", "skip"}),
        "replan_review":      ("Review the replanned trajectory.",           {"continue", "retry", "skip"}),
        "inhand_replan_failed": (
            "In-hand replanning failed. Choose how to recover.",
            {"continue", "skip"},
        ),
        "press_enter":        ("Ready to execute the place trajectory.",     {"continue"}),
        "place_control_review": (
            "Ready to execute the place trajectory.",
            {"continue", "retry"},
        ),
        "place_retry_review": (
            "Review retried place trajectory.",
            {"continue", "retry", "skip"},
        ),
        "intermediate_regrasp_review": (
            "Intermediate pose identified. Choose the next motion.",
            {"continue", "skip"},
        ),
        "intermediate_field_scan": (
            "Run field pose identification at the intermediate regrasp pose?",
            {"continue", "skip"},
        ),
        "grasp_confirm":      ("Ready to execute grasp control.",            {"continue"}),
        "pose_identification":("Run in-hand pose identification?",           {"continue", "skip"}),
        "scene_icp_request":  ("Run scene pose ICP identification?",         {"continue", "retry", "skip"}),
        "scene_pose_review":  ("Accept the identified scene poses?",         {"continue", "retry", "skip"}),
    }

    def __init__(self):
        self._log_lines: list = []
        self._adjust_offsets = {
            "swing": 0.0,
            "boom": 0.0,
            "arm": 0.0,
            "bucket": 0.0,
            "rotate": 0.0,
            "tilt": 0.0,
        }
        self._place_z_offset = 0.0
        self._decision_kind = None
        self._camera_initialized = False
        self._panel_w = _PANEL_W
        self._resizing_panel = False
        self._resize_start_x = 0
        self._resize_start_panel_w = _PANEL_W
        self._log_x_offset = 0
        self._display_lock = threading.Lock()
        self._pending_display_geoms = None
        self._display_update_posted = False
        self._live_lock = threading.Lock()
        self._pending_live_state = None
        self._live_update_posted = False

        # Wired by desktop execution entrypoints before run()
        self.on_decide:            object = None
        self.on_abort:             object = None
        self.on_pick_adjust:       object = None
        self.on_pick_adjust_reset: object = None
        self.on_place_z_offset:    object = None
        self.on_manual_place:      object = None
        self.on_subgoal_pcd_scan:  object = None
        self.on_stone_pcd_mode:    object = None
        self.on_stone_mesh_mode:   object = None
        self.on_lidar_frame_mode:  object = None
        self.on_refresh:           object = None

        gui.Application.instance.initialize()
        self._build_window()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def run(self):
        gui.Application.instance.run()

    # ------------------------------------------------------------------
    # Thread-safe update methods (worker thread → GUI thread)
    # ------------------------------------------------------------------
    def append_log(self, msg: str):
        gui.Application.instance.post_to_main_thread(
            self._win, lambda m=msg: self._append_log_impl(m)
        )

    def update_status(self, st: dict):
        gui.Application.instance.post_to_main_thread(
            self._win, lambda s=st: self._update_status_impl(s)
        )

    def request_decision(self, kind: str, payload: dict):
        gui.Application.instance.post_to_main_thread(
            self._win, lambda k=kind, p=payload: self._request_decision_impl(k, p)
        )

    def display_geometries(self, geoms: list, title: str):
        del title
        with self._display_lock:
            self._pending_display_geoms = geoms
            if self._display_update_posted:
                return
            self._display_update_posted = True
        gui.Application.instance.post_to_main_thread(
            self._win, self._apply_pending_display
        )

    def update_live_joint_state(self, payload: dict):
        with self._live_lock:
            self._pending_live_state = payload
            if self._live_update_posted:
                return
            self._live_update_posted = True
        gui.Application.instance.post_to_main_thread(
            self._win, self._apply_pending_live_state
        )

    def on_finished(self):
        gui.Application.instance.post_to_main_thread(
            self._win, self._on_finished_impl
        )

    def on_failed(self, msg: str):
        gui.Application.instance.post_to_main_thread(
            self._win, lambda m=msg: self._on_failed_impl(m)
        )

    # ------------------------------------------------------------------
    # Window / layout
    # ------------------------------------------------------------------
    def _build_window(self):
        self._win = gui.Application.instance.create_window(
            "Excavator Stacking Operation", 1400, 900
        )
        em      = self._win.theme.font_size
        margin  = int(0.50 * em)
        spacing = int(0.40 * em)

        # 3-D scene (right, flexible)
        self._sw = gui.SceneWidget()
        self._sw.scene = rendering.Open3DScene(self._win.renderer)
        self._sw.scene.set_background([0.12, 0.12, 0.12, 1.0])
        self._sw.scene.scene.set_indirect_light_intensity(30000)
        self._sw.set_on_mouse(self._on_scene_mouse)
        self._win.add_child(self._sw)

        # Persistent visual splitter between the panel and the 3-D scene.
        self._splitter = gui.Vert()
        self._splitter.background_color = gui.Color(0.22, 0.22, 0.22, 1.0)
        self._win.add_child(self._splitter)

        # Left panel
        panel = gui.Vert(spacing, gui.Margins(margin, margin, margin, margin))

        # Status
        status_row = gui.Horiz(spacing)
        status_row.add_child(gui.Label("Status"))
        self._btn_panel_smaller = gui.Button("<")
        self._btn_panel_larger = gui.Button(">")
        self._btn_panel_smaller.set_on_clicked(lambda: self._resize_panel_by(-60))
        self._btn_panel_larger.set_on_clicked(lambda: self._resize_panel_by(60))
        status_row.add_child(self._btn_panel_smaller)
        status_row.add_child(self._btn_panel_larger)
        panel.add_child(status_row)
        self._step_lbl  = gui.Label("Step: —")
        self._stone_lbl = gui.Label("Stone: —")
        self._phase_lbl = gui.Label("Phase: Idle")
        for lbl in (self._step_lbl, self._stone_lbl, self._phase_lbl):
            panel.add_child(lbl)
        self._stone_pcd_cb = gui.Checkbox("Render stones as point clouds")
        self._stone_mesh_cb = gui.Checkbox("Render stones as high-poly meshes")
        self._lidar_frame_cb = gui.Checkbox("Render LiDAR frames")
        self._stone_pcd_cb.set_on_checked(self._on_stone_pcd_checked)
        self._stone_mesh_cb.set_on_checked(self._on_stone_mesh_checked)
        self._lidar_frame_cb.set_on_checked(self._on_lidar_frame_checked)
        panel.add_child(self._stone_pcd_cb)
        panel.add_child(self._stone_mesh_cb)
        panel.add_child(self._lidar_frame_cb)
        self._btn_refresh = gui.Button("Refresh")
        self._btn_refresh.set_on_clicked(self._do_refresh)
        panel.add_child(self._btn_refresh)
        panel.add_fixed(spacing)

        # Prompt
        self._prompt_lbl = gui.Label("Awaiting start...")
        panel.add_child(self._prompt_lbl)
        panel.add_fixed(spacing)

        # Action buttons
        btn_row = gui.Horiz(spacing)
        self._btn_continue = gui.Button("Continue")
        self._btn_retry    = gui.Button("Retry")
        self._btn_skip     = gui.Button("Skip")
        for b in (self._btn_continue, self._btn_retry, self._btn_skip):
            b.enabled = False
            btn_row.add_child(b)
        panel.add_child(btn_row)

        self._btn_manual_place = gui.Button("Manual Place")
        self._btn_manual_place.enabled = False
        self._btn_manual_place.set_on_clicked(self._do_manual_place)
        panel.add_child(self._btn_manual_place)

        self._btn_subgoal_pcd_scan = gui.Button("Scan PCD")
        self._btn_subgoal_pcd_scan.enabled = False
        self._btn_subgoal_pcd_scan.set_on_clicked(self._do_subgoal_pcd_scan)
        panel.add_child(self._btn_subgoal_pcd_scan)

        self._btn_abort = gui.Button("ABORT")
        panel.add_child(self._btn_abort)

        self._btn_continue.set_on_clicked(lambda: self._decide("continue"))
        self._btn_retry.set_on_clicked(lambda: self._decide("retry"))
        self._btn_skip.set_on_clicked(lambda: self._decide("skip"))
        self._btn_abort.set_on_clicked(self._do_abort)
        panel.add_fixed(spacing)

        # Motion retry place adjustment
        panel.add_child(gui.Label("Motion Retry"))
        place_z_row = gui.Horiz(spacing)
        place_z_row.add_child(gui.Label("Place Z offset (m)"))
        self._place_z_offset_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._place_z_offset_edit.set_limits(-0.20, 0.30)
        self._place_z_offset_edit.double_value = self._place_z_offset
        self._place_z_offset_edit.set_on_value_changed(
            self._on_place_z_offset_changed
        )
        self._place_z_offset_edit.enabled = False
        place_z_row.add_child(self._place_z_offset_edit)
        panel.add_child(place_z_row)

        self._btn_place_z_reset = gui.Button("Reset place Z")
        self._btn_place_z_reset.enabled = False
        self._btn_place_z_reset.set_on_clicked(self._do_reset_place_z_offset)
        panel.add_child(self._btn_place_z_reset)
        panel.add_fixed(spacing)

        # Grasp adjustment
        panel.add_child(gui.Label("Grasp Adjust"))
        step_row = gui.Horiz(spacing)
        step_row.add_child(gui.Label("Step (deg)"))
        self._adjust_step = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._adjust_step.set_limits(0.1, 5.0)
        self._adjust_step.double_value = 0.5
        step_row.add_child(self._adjust_step)
        panel.add_child(step_row)

        self._adjust_btns: list = []
        for joint in ("Swing", "Boom", "Arm", "Bucket", "Rotate", "Tilt"):
            row = gui.Horiz(spacing)
            btn_m = gui.Button(f"{joint} -")
            btn_p = gui.Button(f"{joint} +")
            btn_m.enabled = False
            btn_p.enabled = False
            btn_m.set_on_clicked(lambda j=joint.lower(): self._do_adjust(j, -1.0))
            btn_p.set_on_clicked(lambda j=joint.lower(): self._do_adjust(j,  1.0))
            self._adjust_btns.extend([btn_m, btn_p])
            row.add_child(btn_m)
            row.add_child(btn_p)
            panel.add_child(row)

        self._btn_reset = gui.Button("Reset offsets")
        self._btn_reset.enabled = False
        self._btn_reset.set_on_clicked(self._do_reset_adjust)
        self._adjust_btns.append(self._btn_reset)
        panel.add_child(self._btn_reset)

        self._offsets_lbl = gui.Label("")
        self._update_offsets_label()
        panel.add_child(self._offsets_lbl)
        panel.add_fixed(spacing)

        # Log
        panel.add_child(gui.Label("Log"))
        self._log_list = gui.ListView()
        self._log_list.set_max_visible_items(20)
        panel.add_child(self._log_list)

        log_scroll_row = gui.Horiz(spacing)
        log_scroll_row.add_child(gui.Label("Scroll"))
        self._log_x_slider = gui.Slider(gui.Slider.INT)
        self._log_x_slider.set_limits(0, 0)
        self._log_x_slider.set_on_value_changed(self._on_log_scroll_changed)
        log_scroll_row.add_child(self._log_x_slider)
        panel.add_child(log_scroll_row)

        self._panel = panel
        self._win.add_child(panel)
        self._win.set_on_layout(self._on_layout)

    def _on_layout(self, _ctx):
        r = self._win.content_rect
        max_panel_w = self._max_panel_width(r)
        self._panel_w = max(_PANEL_MIN_W, min(self._panel_w, max_panel_w))
        self._panel.frame = gui.Rect(r.x, r.y, self._panel_w, r.height)
        self._splitter.frame = gui.Rect(
            r.x + self._panel_w, r.y, _SPLITTER_W, r.height
        )
        self._sw.frame    = gui.Rect(
            r.x + self._panel_w + _SPLITTER_W,
            r.y,
            r.width - self._panel_w - _SPLITTER_W,
            r.height,
        )

    def _max_panel_width(self, rect):
        return max(_PANEL_MIN_W, int(rect.width * _PANEL_MAX_FRAC))

    def _on_scene_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            if self._resizing_panel:
                self._resizing_panel = False
                return gui.Widget.EventCallbackResult.CONSUMED
            return gui.Widget.EventCallbackResult.IGNORED

        rect = self._win.content_rect
        scene_left_x = rect.x + self._panel_w + _SPLITTER_W
        near_left_edge = (
            event.x <= _SCENE_EDGE_DRAG_W
            or abs(event.x - scene_left_x) <= _SCENE_EDGE_DRAG_W
        )
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and near_left_edge:
            self._resizing_panel = True
            self._resize_start_x = event.x
            self._resize_start_panel_w = self._panel_w
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.DRAG and self._resizing_panel:
            delta_x = event.x - self._resize_start_x
            self._panel_w = max(
                _PANEL_MIN_W,
                min(
                    int(self._resize_start_panel_w + delta_x),
                    self._max_panel_width(rect),
                ),
            )
            self._win.set_needs_layout()
            self._win.post_redraw()
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.WHEEL and near_left_edge:
            step = -20 if event.wheel_dy > 0 else 20
            self._resize_panel_by(step)
            return gui.Widget.EventCallbackResult.CONSUMED

        return gui.Widget.EventCallbackResult.IGNORED

    def _resize_panel_by(self, delta: int):
        rect = self._win.content_rect
        self._panel_w = max(
            _PANEL_MIN_W,
            min(int(self._panel_w + delta), self._max_panel_width(rect)),
        )
        self._win.set_needs_layout()
        self._win.post_redraw()

    # ------------------------------------------------------------------
    # GUI-thread event handlers
    # ------------------------------------------------------------------
    def _decide(self, value: str):
        decision_kind = self._decision_kind
        self._set_action_buttons_enabled(set())
        self._set_adjust_enabled(False)
        self._set_place_z_offset_enabled(False)
        keep_manual_enabled = (
            decision_kind == "replan" and value in {"continue", "retry"}
        ) or (decision_kind == "replan_review" and value == "retry")
        self._set_manual_place_enabled(keep_manual_enabled)
        self._set_subgoal_pcd_scan_enabled(False)
        self._prompt_lbl.text = "Working..."
        if self.on_decide:
            self.on_decide(value)

    def _do_abort(self):
        self._btn_abort.enabled = False
        self._set_action_buttons_enabled(set())
        self._set_adjust_enabled(False)
        self._set_place_z_offset_enabled(False)
        self._set_manual_place_enabled(False)
        self._set_subgoal_pcd_scan_enabled(False)
        self._append_log_impl("Abort requested.")
        if self.on_abort:
            self.on_abort()

    def _do_adjust(self, joint: str, sign: float):
        delta = sign * float(self._adjust_step.double_value)
        self._adjust_offsets[joint] += delta
        self._update_offsets_label()
        if self.on_pick_adjust:
            self.on_pick_adjust(joint, delta)

    def _do_reset_adjust(self):
        for k in self._adjust_offsets:
            self._adjust_offsets[k] = 0.0
        self._update_offsets_label()
        if self.on_pick_adjust_reset:
            self.on_pick_adjust_reset()

    def _on_place_z_offset_changed(self, _value):
        self._place_z_offset = float(self._place_z_offset_edit.double_value)
        if self.on_place_z_offset:
            self.on_place_z_offset(self._place_z_offset)

    def _do_reset_place_z_offset(self):
        self._place_z_offset = 0.0
        self._place_z_offset_edit.double_value = 0.0
        if self.on_place_z_offset:
            self.on_place_z_offset(0.0)

    def _do_manual_place(self):
        self._set_manual_place_enabled(False)
        self._prompt_lbl.text = "Manual place requested..."
        self._append_log_impl("Manual place requested.")
        if self.on_manual_place:
            self.on_manual_place()

    def _do_subgoal_pcd_scan(self):
        self._append_log_impl("Diagnostic PCD scan requested.")
        if self.on_subgoal_pcd_scan:
            self.on_subgoal_pcd_scan()

    def _do_refresh(self):
        self._append_log_impl("Refresh requested.")
        if self.on_refresh:
            self.on_refresh()

    def _on_stone_pcd_checked(self, enabled: bool):
        if self.on_stone_pcd_mode:
            self.on_stone_pcd_mode(bool(enabled))

    def _on_stone_mesh_checked(self, enabled: bool):
        if self.on_stone_mesh_mode:
            self.on_stone_mesh_mode(bool(enabled))

    def _on_lidar_frame_checked(self, enabled: bool):
        if self.on_lidar_frame_mode:
            self.on_lidar_frame_mode(bool(enabled))

    def _update_offsets_label(self):
        o = self._adjust_offsets
        self._offsets_lbl.text = "  ".join(
            [
                f"Swing {o['swing']:.1f}°",
                f"Boom {o['boom']:.1f}°",
                f"Arm {o['arm']:.1f}°",
                f"Bucket {o['bucket']:.1f}°",
                f"Rotate {o['rotate']:.1f}°",
                f"Tilt {o['tilt']:.1f}°",
            ]
        )

    def _set_action_buttons_enabled(self, enabled: set):
        self._btn_continue.enabled = "continue" in enabled
        self._btn_retry.enabled    = "retry"    in enabled
        self._btn_skip.enabled     = "skip"     in enabled

    def _set_adjust_enabled(self, enabled: bool):
        for b in self._adjust_btns:
            b.enabled = enabled

    def _set_place_z_offset_enabled(self, enabled: bool):
        self._place_z_offset_edit.enabled = enabled
        self._btn_place_z_reset.enabled = enabled

    def _set_manual_place_enabled(self, enabled: bool):
        self._btn_manual_place.enabled = bool(enabled)

    def _set_subgoal_pcd_scan_enabled(self, enabled: bool):
        self._btn_subgoal_pcd_scan.enabled = bool(enabled)

    # ------------------------------------------------------------------
    # Main-thread implementations (scheduled by thread-safe methods)
    # ------------------------------------------------------------------
    def _append_log_impl(self, msg: str):
        self._log_lines.extend(msg.split("\n"))
        if len(self._log_lines) > _LOG_CAP:
            self._log_lines = self._log_lines[-_LOG_CAP:]
        self._refresh_log_view()

    def _on_log_scroll_changed(self, value):
        self._log_x_offset = int(value)
        self._refresh_log_view(update_slider=False)

    def _refresh_log_view(self, update_slider: bool = True):
        max_offset = max((len(line) for line in self._log_lines), default=0)
        max_offset = max(0, max_offset - 1)
        if update_slider:
            self._log_x_slider.set_limits(0, max_offset)
        self._log_x_offset = max(0, min(self._log_x_offset, max_offset))
        if update_slider:
            self._log_x_slider.int_value = self._log_x_offset

        offset = self._log_x_offset
        visible = [line[offset:offset + _LOG_VIEW_CHARS] for line in self._log_lines]
        self._log_list.set_items(visible)
        self._log_list.selected_index = len(visible) - 1

    def _update_status_impl(self, st: dict):
        if "step" in st and "total" in st:
            self._step_lbl.text = f"Step: {st['step']} / {st['total']}"
        if "stone_id" in st:
            self._stone_lbl.text = f"Stone: {st['stone_id']}"
        if "phase" in st:
            self._phase_lbl.text = f"Phase: {st['phase']}"

    def _request_decision_impl(self, kind: str, payload: dict):
        self._decision_kind = kind
        prompt, enabled = self._PROMPTS.get(
            kind, (f"Decision needed: {kind}", {"continue"})
        )
        step_prefix = f"[Step {payload['step']}] " if "step" in payload else ""
        if kind == "grasp_confirm":
            phase = payload.get("phase")
            if phase == "place":
                # continue = small release step, retry = full open, skip = finish
                enabled = {"continue", "retry", "skip"}
            elif phase == "manual_release":
                # continue = full open, retry = small open step
                enabled = {"continue", "retry"}
            if phase == "pick_approach":
                prompt = "Ready to execute pick approach."
            elif phase == "pick":
                prompt = "Ready to execute pick grasp control."
            elif phase == "place_approach":
                prompt = "Ready to execute place approach."
            elif phase == "place":
                prompt = "Ready to release the stone."
            elif phase == "manual_place":
                prompt = "Manually adjust to the place pose."
            elif phase == "manual_release":
                prompt = "Ready to manually open the gripper."
            elif phase == "manual_retreat":
                prompt = "Manually adjust to the retreat pose."
        elif kind == "press_enter" and payload.get("phase") == "manual_place":
            prompt = "Ready to start manual place control."
        elif kind == "place_control_review":
            prompt = "Ready to execute place trajectory."
        elif kind == "replan_review" and "candidate" in payload:
            prompt = (
                "Review replanned trajectory "
                f"{payload['candidate']}/{payload.get('num_candidates', '?')}."
            )
        elif kind == "place_retry_review" and "candidate" in payload:
            prompt = (
                "Review retried place trajectory "
                f"{payload['candidate']}/{payload.get('num_candidates', '?')}."
            )
        elif kind == "scene_pose_review":
            num_poses = payload.get("num_poses")
            ground_height = payload.get("ground_height")
            details = []
            if num_poses is not None:
                details.append(f"{num_poses} poses")
            if ground_height is not None:
                details.append(f"ground z={float(ground_height):.3f} m")
            if details:
                prompt = prompt + " (" + ", ".join(details) + ")"
        self._prompt_lbl.text = step_prefix + prompt

        self._btn_continue.text = "Continue"
        self._btn_retry.text = "Retry"
        self._btn_skip.text = "Skip"
        self._set_action_buttons_enabled(enabled)

        if kind == "online_seed_review":
            self._btn_continue.text = "Use Planned"
            self._btn_retry.text = "Replan"
        elif kind == "place_control_review":
            self._btn_continue.text = "Place"
            self._btn_retry.text = "Retry Motion"
        elif kind == "place_retry_review":
            self._btn_continue.text = "Accept"
            self._btn_retry.text = "Retry"
            self._btn_skip.text = "Original"
        elif kind == "intermediate_regrasp_review":
            self._btn_continue.text = "Direct"
            self._btn_skip.text = "Original"
        elif kind == "intermediate_field_scan":
            self._btn_continue.text = "Scan"
            self._btn_skip.text = "Skip"
        elif kind == "scene_pose_review":
            self._btn_continue.text = "Accept"
            self._btn_retry.text = "Retry ICP"
            self._btn_skip.text = "Keep Current"
        elif kind == "scene_icp_request":
            self._btn_continue.text = "Run ICP"
            self._btn_retry.text = "Manual Init"
            self._btn_skip.text = "Skip"
        elif kind == "replan":
            self._btn_continue.text = "Direct"
            self._btn_retry.text = "Regrasp"
            self._btn_skip.text = "Skip"
        elif kind == "replan_review":
            self._btn_continue.text = "Accept"
            self._btn_retry.text = "Next"
            self._btn_skip.text = "Put Down"
        elif kind == "inhand_replan_failed":
            self._btn_continue.text = "Original"
            self._btn_skip.text = "Put Down"
        elif kind == "grasp_confirm":
            phase = payload.get("phase")
            if phase == "place":
                self._btn_continue.text = "Release"
                self._btn_retry.text = "Open Full"
                self._btn_skip.text = "Finish"
            elif phase == "manual_release":
                self._btn_continue.text = "Open"
                self._btn_retry.text = "Open Step"
            elif phase == "manual_retreat":
                self._btn_continue.text = "Finish"

        is_adjustable = kind == "grasp_confirm" and payload.get("phase") in (
            "pick_approach",
            "pick",
            "place",
            "place_approach",
            "manual_place",
            "manual_release",
            "manual_retreat",
        )
        self._set_adjust_enabled(is_adjustable)
        self._set_place_z_offset_enabled(
            kind in {"plan_review", "place_control_review", "place_retry_review"}
        )
        self._set_manual_place_enabled(
            kind in {"replan", "replan_review", "inhand_replan_failed"}
        )
        self._set_subgoal_pcd_scan_enabled(
            kind == "grasp_confirm"
            and payload.get("phase")
            in {
                "pick_approach",
                "pick",
                "place_approach",
                "place",
                "manual_place",
                "manual_release",
                "manual_retreat",
            }
        )
        if is_adjustable:
            for k in self._adjust_offsets:
                self._adjust_offsets[k] = 0.0
            self._update_offsets_label()

    def _apply_pending_display(self):
        with self._display_lock:
            geoms = self._pending_display_geoms
            self._pending_display_geoms = None
            self._display_update_posted = False
        if geoms is None:
            return

        self._display_geometries_impl(geoms)

        with self._display_lock:
            if self._pending_display_geoms is None or self._display_update_posted:
                return
            self._display_update_posted = True
        gui.Application.instance.post_to_main_thread(
            self._win, self._apply_pending_display
        )

    def _apply_pending_live_state(self):
        with self._live_lock:
            payload = self._pending_live_state
            self._pending_live_state = None
            self._live_update_posted = False
        if payload is None:
            return

        self._update_live_joint_state_impl(payload)

        with self._live_lock:
            if self._pending_live_state is None or self._live_update_posted:
                return
            self._live_update_posted = True
        gui.Application.instance.post_to_main_thread(
            self._win, self._apply_pending_live_state
        )

    def _display_geometries_impl(self, geoms: list):
        sc = self._sw.scene
        sc.clear_geometry()
        for i, entry in enumerate(geoms):
            g, color, meta = self._geometry_from_entry(entry)
            if g is None:
                continue
            if not self._has_renderable_geometry(g):
                continue
            try:
                name = str(meta.get("name") or f"g_{i}")
                mat = rendering.MaterialRecord()
                alpha = self._material_alpha(meta)
                mat.base_color = (color or [0.7, 0.7, 0.75]) + [alpha]
                if hasattr(g, "lines"):
                    mat.shader = "unlitLine"
                    mat.line_width = 4.0
                elif hasattr(g, "vertices"):
                    if not g.has_vertex_normals():
                        g.compute_vertex_normals()
                    if not g.has_triangle_normals():
                        g.compute_triangle_normals()
                    if alpha < 1.0:
                        mat.shader = "defaultLitTransparency"
                        mat.has_alpha = True
                    else:
                        mat.shader = "defaultLit"
                else:
                    mat.shader = "defaultUnlit"
                sc.add_geometry(name, g, mat)
                transform = meta.get("transform")
                if transform is not None:
                    self._set_geometry_transform(name, transform)
            except Exception:
                continue
        bounds = sc.bounding_box
        if not bounds.is_empty() and not self._camera_initialized:
            self._sw.setup_camera(60.0, bounds, bounds.get_center())
            self._camera_initialized = True

    @staticmethod
    def _has_renderable_geometry(g) -> bool:
        try:
            if hasattr(g, "lines"):
                points = np.asarray(g.points, dtype=np.float64)
                lines = np.asarray(g.lines, dtype=np.int64)
                if points.ndim != 2 or points.shape[1] != 3 or lines.size == 0:
                    return False
                if not np.all(np.isfinite(points)):
                    return False
                lines = lines.reshape(-1, 2)
                valid = np.all((0 <= lines) & (lines < len(points)), axis=1)
                if not np.any(valid):
                    return False
                segment_lengths = np.linalg.norm(
                    points[lines[valid, 1]] - points[lines[valid, 0]],
                    axis=1,
                )
                return bool(np.any(segment_lengths > 1e-9))

            if hasattr(g, "vertices"):
                vertices = np.asarray(g.vertices, dtype=np.float64)
                if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
                    return False
                if not np.all(np.isfinite(vertices)):
                    return False
                if hasattr(g, "triangles") and len(g.triangles) == 0:
                    return False
                return bool(np.ptp(vertices, axis=0).max() > 1e-9)

            if hasattr(g, "points"):
                points = np.asarray(g.points, dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                    return False
                if not np.all(np.isfinite(points)):
                    return False
                return bool(np.ptp(points, axis=0).max() > 1e-9)
        except Exception:
            return False
        return True

    def _update_live_joint_state_impl(self, payload: dict):
        transforms = payload.get("transforms") or {}
        for name, transform in transforms.items():
            self._set_geometry_transform(name, transform)
        self._win.post_redraw()

    def _set_geometry_transform(self, name: str, transform):
        sc = self._sw.scene
        try:
            if hasattr(sc, "has_geometry") and not sc.has_geometry(name):
                return
            transform = np.asarray(transform, dtype=np.float64)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                return
            sc.set_geometry_transform(name, transform)
        except Exception:
            return

    def _material_alpha(self, meta):
        try:
            alpha = float(meta.get("alpha", 1.0))
        except (TypeError, ValueError):
            alpha = 1.0
        if not np.isfinite(alpha):
            return 1.0
        return float(np.clip(alpha, 0.0, 1.0))

    def _geometry_from_entry(self, entry):
        if isinstance(entry, dict):
            color = list(entry.get("color") or [0.7, 0.7, 0.75])[:3]
            meta = self._geometry_meta_from_entry(entry)
            if entry.get("kind") == "mesh":
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(
                    np.asarray(entry["vertices"], dtype=np.float64)
                )
                mesh.triangles = o3d.utility.Vector3iVector(
                    np.asarray(entry["triangles"], dtype=np.int32)
                )
                if entry.get("style") == "wireframe":
                    return self._mesh_wireframe(mesh, color), color, meta
                return mesh, color, meta
            if entry.get("kind") == "pcd":
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    np.asarray(entry["points"], dtype=np.float64)
                )
                return pcd, color, meta
            if entry.get("kind") == "lineset":
                lineset = o3d.geometry.LineSet()
                lineset.points = o3d.utility.Vector3dVector(
                    np.asarray(entry["points"], dtype=np.float64)
                )
                lineset.lines = o3d.utility.Vector2iVector(
                    np.asarray(entry["lines"], dtype=np.int32)
                )
                if "colors" in entry:
                    lineset.colors = o3d.utility.Vector3dVector(
                        np.asarray(entry["colors"], dtype=np.float64)
                    )
                elif len(lineset.lines) > 0:
                    colors = np.tile(
                        np.asarray(color, dtype=np.float64),
                        (len(lineset.lines), 1),
                    )
                    lineset.colors = o3d.utility.Vector3dVector(colors)
                return lineset, color, meta
            return None, None, {}

        g, color = entry, None
        meta = {}
        style = None
        if isinstance(entry, tuple):
            g = entry[0]
            color = list(entry[1])[:3] if len(entry) > 1 else None
            for extra in entry[2:]:
                if isinstance(extra, dict):
                    meta.update(extra)
                elif isinstance(extra, str):
                    style = extra
        if style == "wireframe" and hasattr(g, "vertices") and hasattr(g, "triangles"):
            return self._mesh_wireframe(g, color), color, meta
        if hasattr(g, "vertices") or hasattr(g, "points") or hasattr(g, "lines"):
            return g, color, meta
        return None, None, {}

    def _mesh_wireframe(self, mesh, color):
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        lineset = o3d.geometry.LineSet()
        lineset.points = mesh.vertices
        if triangles.size == 0:
            lineset.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
            return lineset
        edges = np.vstack(
            [
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            ]
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        lineset.lines = o3d.utility.Vector2iVector(edges)
        if color is not None:
            colors = np.tile(np.asarray(color, dtype=np.float64), (len(edges), 1))
            lineset.colors = o3d.utility.Vector3dVector(colors)
        return lineset

    def _geometry_meta_from_entry(self, entry: dict):
        meta = {}
        if entry.get("name"):
            meta["name"] = str(entry["name"])
        if "transform" in entry:
            transform = np.asarray(entry["transform"], dtype=np.float64)
            if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
                meta["transform"] = transform
        if "alpha" in entry:
            meta["alpha"] = self._material_alpha(entry)
        return meta

    def _on_finished_impl(self):
        self._prompt_lbl.text = "Execution complete."
        self._set_action_buttons_enabled(set())
        self._btn_abort.enabled = False
        self._append_log_impl("[DONE]")

    def _on_failed_impl(self, msg: str):
        self._prompt_lbl.text = "Execution failed — see log."
        self._set_action_buttons_enabled(set())
        self._append_log_impl(f"[FAILED] {msg}")
