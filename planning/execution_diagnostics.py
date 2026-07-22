"""Operator-triggered diagnostic PCD capture during execution subgoals."""

from .execution_common import *


SUBGOAL_PCD_WAIT_TIMEOUT = 60.0
SUBGOAL_PCD_DONE_DRAIN_SECONDS = 12.0
SUBGOAL_PCD_QUIET_SECONDS = 1.0
SUBGOAL_PCD_MERGE_VOXEL = 0.003
SUBGOAL_PCD_DISPLAY_VOXEL = 0.02
SUBGOAL_PCD_COLOR = [0.95, 0.85, 0.10]
SUBGOAL_PCD_SETTLE_ERROR_TOL = 0.03
SUBGOAL_PCD_SETTLE_TIME_LIMIT = 8.0
REQUESTED_SCAN_PCD_MERGE_VOXEL = 0.003
REQUESTED_SCAN_PCD_DONE_DRAIN_SECONDS = 0.5


class ExecutionDiagnosticsMixin:
    def _start_requested_scan_pcd_transfer(self, scan_dir: str, label: str) -> None:
        os.makedirs(scan_dir, exist_ok=True)
        self._requested_scan_pcd_transfer = {
            "scan_dir": scan_dir,
            "label": label,
            "merged": o3d.geometry.PointCloud(),
            "frames": 0,
            "last_frame_time": None,
        }
        self.scene_pcd_sub.reset_get_flag()
        self._log(f"Saving requested scan PCD transfer locally under {scan_dir}.")

    def _requested_scan_pcd_frame_prefixes(self, label: str) -> tuple[str, ...]:
        label = str(label or "")
        if label.startswith("inhand"):
            return ("inhand_gripper_",)
        if label.startswith("field"):
            return ("field_base_",)
        return ()

    def _spin_requested_scan_pcd_transfer(self, timeout_sec: float = 0.0) -> None:
        if self._requested_scan_pcd_transfer is None:
            return
        with self._ros_spin_lock:
            self.rclpy.spin_once(self.scene_pcd_sub, timeout_sec=timeout_sec)
        self._collect_requested_scan_pcd_frame()

    def _collect_requested_scan_pcd_frame(self) -> bool:
        transfer = self._requested_scan_pcd_transfer
        if transfer is None or not self.scene_pcd_sub.get_flag():
            return False

        pts = self.scene_pcd_sub.points.copy()
        frame_id = (
            self.scene_pcd_sub.last_msg.header.frame_id
            if self.scene_pcd_sub.last_msg is not None
            else ""
        )
        self.scene_pcd_sub.reset_get_flag()
        if pts.size == 0:
            return False
        expected_prefixes = self._requested_scan_pcd_frame_prefixes(
            str(transfer["label"])
        )
        if expected_prefixes and not str(frame_id).startswith(expected_prefixes):
            self._log(
                "Ignoring unrelated requested-scan PCD frame "
                f"{frame_id or '<empty>'}; expected prefix {expected_prefixes}."
            )
            return False

        frame = o3d.geometry.PointCloud()
        frame.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        frame_count = int(transfer["frames"])
        tag = self._safe_scene_pcd_tag(
            frame_id,
            fallback=f"{transfer['label']}_frame{frame_count}",
        )
        if os.path.splitext(tag)[1]:
            tag = os.path.splitext(tag)[0]

        scan_dir = transfer["scan_dir"]
        save_path = os.path.join(scan_dir, f"{tag}.ply")
        if os.path.exists(save_path):
            save_path = os.path.join(scan_dir, f"{tag}_{frame_count}.ply")
        o3d.io.write_point_cloud(save_path, frame)

        transfer["merged"] += frame
        transfer["frames"] = frame_count + 1
        transfer["last_frame_time"] = time.time()
        self._log(
            "Saved requested scan PCD frame "
            f"{transfer['frames']}: {os.path.basename(save_path)} "
            f"({len(frame.points)} points)"
        )
        return True

    def _drain_requested_scan_pcd_transfer(self) -> None:
        if self._requested_scan_pcd_transfer is None:
            return
        drain_seconds = float(
            self.planning_params.get(
                "requested_scan_pcd_done_drain_seconds",
                REQUESTED_SCAN_PCD_DONE_DRAIN_SECONDS,
            )
        )
        deadline = time.time() + max(0.0, drain_seconds)
        while time.time() < deadline and not self._abort.is_set():
            self._spin_requested_scan_pcd_transfer(timeout_sec=0.05)

    def _finish_requested_scan_pcd_transfer(self, scan_done: bool):
        if self._requested_scan_pcd_transfer is None:
            return None
        self._drain_requested_scan_pcd_transfer()
        transfer = self._requested_scan_pcd_transfer
        self._requested_scan_pcd_transfer = None

        scan_dir = transfer["scan_dir"]
        label = self._safe_scene_pcd_tag(transfer["label"], "requested_scan")
        merged = transfer["merged"]
        frame_count = int(transfer["frames"])
        status_path = os.path.join(scan_dir, f"{label}_desktop_pcd_status.txt")
        if merged.is_empty():
            with open(status_path, "w") as f:
                f.write("failed: desktop_received_no_requested_scan_pcd_frames\n")
                f.write(f"scan_done: {scan_done}\n")
                f.write(f"transferred_frames: {frame_count}\n")
            self._log(f"No requested scan PCD frames were received for {label}.")
            return None

        voxel = float(
            self.planning_params.get(
                "requested_scan_pcd_merge_voxel",
                REQUESTED_SCAN_PCD_MERGE_VOXEL,
            )
        )
        if voxel > 0.0:
            merged = merged.voxel_down_sample(voxel)

        merged_path = os.path.join(scan_dir, f"{label}_merged_desktop.ply")
        o3d.io.write_point_cloud(merged_path, merged)
        with open(status_path, "w") as f:
            f.write("succeeded\n")
            f.write(f"scan_done: {scan_done}\n")
            f.write(f"transferred_frames: {frame_count}\n")
            f.write(f"merge_voxel: {voxel}\n")
            f.write(f"merged_points: {len(merged.points)}\n")

        self._log(
            "Saved requested scan PCD locally: "
            f"{frame_count} frame(s), {len(merged.points)} merged points, "
            f"path={merged_path}."
        )
        if str(label).startswith("field"):
            self._show_requested_scan_pcd_preview(merged, label)
        return merged

    def _show_requested_scan_pcd_preview(self, merged, label: str) -> None:
        pcd = o3d.geometry.PointCloud(merged)
        voxel = float(
            self.planning_params.get(
                "requested_scan_pcd_display_voxel",
                SCENE_PCD_DISPLAY_VOXEL,
            )
        )
        if voxel > 0.0:
            pcd = pcd.voxel_down_sample(voxel)
        if pcd.is_empty():
            self._log(f"Requested scan PCD preview for {label} is empty.")
            return

        pcd.paint_uniform_color(C_SCENE_SCAN_PCD)
        entry = (
            pcd,
            C_SCENE_SCAN_PCD,
            {"name": f"{label}_requested_scan_pcd", "alpha": 0.95},
        )
        self._remove_overlay_entries_by_name(f"{label}_requested_scan_pcd")
        self._viz_overlay_entries = self._viz_overlay_entries + [entry]
        self._last_live_layout_key = None
        q = self._last_live_q
        if q is None:
            q = getattr(self, "q_joint", None)
        if q is None:
            q = getattr(self, "q_home", None)
        if q is None:
            q = np.zeros(6, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64).copy()[:6]
        self._log(
            "Requested scan GUI preview PCD: "
            f"{len(pcd.points)} point(s), label={label}, "
            f"display_voxel={voxel}."
        )
        self._on_display_geometries(
            self._base_scene(q) + self._viz_overlay_entries + [self.origin_frame],
            f"Requested scan PCD: {label}",
        )

    def on_subgoal_pcd_scan(self):
        if self._pick_adjust_q is None:
            self._log("Diagnostic PCD ignored: no active pick/place subgoal prompt.")
            return
        if not PERCEPTION_ON:
            self._log("Diagnostic PCD ignored: perception is disabled.")
            return
        if not ROS_CONTROL_ON:
            self._log("Diagnostic PCD ignored: ROS control is disabled.")
            return
        if not hasattr(self, "diagnostic_pcd_request_pub"):
            self._log("Diagnostic PCD ignored: request publisher is not initialized.")
            return
        if not self._subgoal_pcd_scan_lock.acquire(blocking=False):
            self._log("Diagnostic PCD request ignored: a scan is already running.")
            return

        q = self._pick_adjust_q.copy()
        phase = str(self._pick_adjust_phase or "subgoal")
        grab = str(self._pick_adjust_grab or "close")
        step_idx = getattr(self, "_current_step_index", None)
        target_id = getattr(self, "_current_target_id", None)
        thread = threading.Thread(
            target=self._run_subgoal_pcd_scan,
            args=(q, phase, grab, step_idx, target_id),
            daemon=True,
        )
        self._subgoal_pcd_scan_thread = thread
        thread.start()

    def _run_subgoal_pcd_scan(self, q, phase, grab, step_idx, target_id):
        try:
            phase_label = phase.replace("_", "-")
            step_label = "step_unknown" if step_idx is None else f"step_{step_idx + 1}"
            label = f"{step_label}_{phase_label}"
            scan_dir = self._subgoal_pcd_scan_dir(step_idx, label)

            self._set_status(phase=f"Diagnostic PCD: {phase_label}")
            self._log(
                "Requesting diagnostic PCD at current subgoal "
                f"({label}, grab={grab})."
            )
            q_render = self._publish_waiting_subgoal_q(q, grab)
            self._start_subgoal_pcd_transfer(scan_dir)
            for _ in range(5):
                self.diagnostic_pcd_request_pub.publish(scan_dir, label)
                with self._ros_spin_lock:
                    self.rclpy.spin_once(self.scene_pcd_sub, timeout_sec=0.0)
                    self.rclpy.spin_once(
                        self.diagnostic_pcd_done_sub, timeout_sec=0.0
                    )
                self._collect_subgoal_pcd_frame()
                time.sleep(0.05)

            scan_done = self._wait_for_subgoal_pcd_done(scan_dir)
            merged = self._finish_subgoal_pcd_transfer(scan_dir, scan_done)
            if merged is None or merged.is_empty():
                return
            self._show_subgoal_pcd_preview(
                merged,
                q_render,
                phase,
                phase_label,
                step_idx,
                target_id,
            )
        finally:
            self._subgoal_pcd_scan_lock.release()

    def _subgoal_pcd_scan_dir(self, step_idx, label: str) -> str:
        if step_idx is None:
            step_dir = os.path.join(self.base_log_dir, "step_unknown")
        else:
            step_dir = os.path.join(self.base_log_dir, f"step_{step_idx + 1}")
        self._subgoal_pcd_scan_counter += 1
        scan_name = f"{self._subgoal_pcd_scan_counter:03d}_{label}"
        return os.path.join(step_dir, "subgoal_pcd", scan_name)

    def _publish_waiting_subgoal_q(self, q, grab: str) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).copy()
        v = np.zeros_like(q)
        tol = float(SUBGOAL_PCD_SETTLE_ERROR_TOL)
        time_limit = float(SUBGOAL_PCD_SETTLE_TIME_LIMIT)
        self._log(
            "Settling robot at diagnostic subgoal before PCD capture "
            f"(tol={tol:.3f} rad, timeout={time_limit:.1f}s)."
        )
        try:
            self.position_control(
                q,
                v,
                self.joint_node_pub,
                self.joint_node_sub,
                error_tol=tol,
                time_limit=time_limit,
                grab=grab,
                state_cb=self._live_joint_state_cb(),
                spin_lock=self._ros_spin_lock,
            )
        except Exception as exc:
            self._log(
                "Diagnostic PCD subgoal settle failed before scan request: "
                f"{exc}. Continuing with requested q."
            )

        q_feedback = self._subgoal_pcd_joint_feedback_q()
        q_render = q_feedback if q_feedback is not None else q
        err = self._subgoal_pcd_joint_error(q_render, q)
        if err is not None:
            self._log(f"Diagnostic PCD frame q error after settle: {err:.4f} rad.")
        if err is not None and err > max(0.10, 3.0 * tol):
            self._log(
                "Warning: diagnostic PCD joint feedback is far from the requested "
                "subgoal q; rendering preview in the feedback frame."
            )

        with self._ros_spin_lock:
            for _ in range(100):
                self.joint_node_pub.publish(q, v, grab)
        if LIVE_JOINT_VIEWER_ON:
            self._live_state_cb(q_render)
        return q_render

    def _subgoal_pcd_joint_feedback_q(self):
        node = getattr(self, "joint_node_sub", None)
        if node is None:
            return None
        q = getattr(node, "pos", None)
        if q is None:
            return None
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] < 6 or not np.all(np.isfinite(q[:6])):
            return None
        return q[:6].copy()

    def _subgoal_pcd_joint_error(self, q_actual, q_desired):
        try:
            q_actual = np.asarray(q_actual, dtype=np.float64).reshape(-1)[:6]
            q_desired = np.asarray(q_desired, dtype=np.float64).reshape(-1)[:6]
            if q_actual.shape[0] != 6 or q_desired.shape[0] != 6:
                return None
            diff = (q_actual - q_desired + np.pi) % (2.0 * np.pi) - np.pi
            return float(np.linalg.norm(diff))
        except Exception:
            return None

    def _start_subgoal_pcd_transfer(self, scan_dir: str) -> None:
        os.makedirs(scan_dir, exist_ok=True)
        label = os.path.basename(scan_dir)
        if "_" in label:
            label = label.split("_", 1)[1]
        self._subgoal_pcd_transfer = {
            "scan_dir": scan_dir,
            "label": label,
            "expected_frame_prefix": f"diagnostic_{label}_",
            "merged": o3d.geometry.PointCloud(),
            "frames": 0,
            "last_frame_time": None,
        }
        self.scene_pcd_sub.reset_get_flag()
        self.diagnostic_pcd_done_sub.reset_get_flag()
        self._drain_stale_subgoal_pcd_done(scan_dir)

    def _drain_stale_subgoal_pcd_done(self, scan_dir: str) -> None:
        drained = []
        with self._ros_spin_lock:
            for _ in range(20):
                self.rclpy.spin_once(self.diagnostic_pcd_done_sub, timeout_sec=0.0)
                if not self.diagnostic_pcd_done_sub.get_flag():
                    continue
                done_dir = self.diagnostic_pcd_done_sub.step_log_dir
                self.diagnostic_pcd_done_sub.reset_get_flag()
                if self._same_diagnostic_scan_dir(done_dir, scan_dir):
                    continue
                drained.append(done_dir)
        if drained:
            unique = []
            for item in drained:
                if item not in unique:
                    unique.append(item)
            self._log(
                "Cleared stale diagnostic PCD completion message(s): "
                + ", ".join(unique[:3])
                + (" ..." if len(unique) > 3 else "")
            )

    def _collect_subgoal_pcd_frame(self) -> bool:
        transfer = self._subgoal_pcd_transfer
        if transfer is None or not self.scene_pcd_sub.get_flag():
            return False

        pts = self.scene_pcd_sub.points.copy()
        frame_id = (
            self.scene_pcd_sub.last_msg.header.frame_id
            if self.scene_pcd_sub.last_msg is not None
            else ""
        )
        self.scene_pcd_sub.reset_get_flag()
        if pts.size == 0:
            return False
        frame_id = str(frame_id)
        expected_prefix = str(transfer.get("expected_frame_prefix") or "")
        if expected_prefix and not frame_id.startswith(expected_prefix):
            self._log(
                "Ignoring diagnostic PCD frame for another request: "
                f"{frame_id or '<empty>'} (expected prefix {expected_prefix})"
            )
            return False
        if not frame_id.startswith("diagnostic_"):
            self._log(
                "Ignoring non-diagnostic PCD frame during diagnostic transfer: "
                f"{frame_id or '<empty>'}"
            )
            return False

        frame = o3d.geometry.PointCloud()
        frame.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        frame_count = int(transfer["frames"])
        tag = self._safe_scene_pcd_tag(
            frame_id,
            fallback=f"diagnostic_pcd_frame{frame_count}",
        )
        if not tag.startswith("diagnostic_"):
            tag = f"diagnostic_{tag}"
        if os.path.splitext(tag)[1]:
            tag = os.path.splitext(tag)[0]

        scan_dir = transfer["scan_dir"]
        save_path = os.path.join(scan_dir, f"{tag}.ply")
        if os.path.exists(save_path):
            save_path = os.path.join(scan_dir, f"{tag}_{frame_count}.ply")
        o3d.io.write_point_cloud(save_path, frame)

        transfer["merged"] += frame
        transfer["frames"] = frame_count + 1
        transfer["last_frame_time"] = time.time()
        self._log(
            "Received diagnostic PCD frame "
            f"{transfer['frames']}: {os.path.basename(save_path)} "
            f"({len(frame.points)} points)"
        )
        return True

    def _wait_for_subgoal_pcd_done(self, scan_dir: str) -> bool:
        timeout = float(
            self.planning_params.get(
                "diagnostic_pcd_done_wait_timeout",
                SUBGOAL_PCD_WAIT_TIMEOUT,
            )
        )
        self._log(
            "Waiting for diagnostic PCD completion "
            f"(/diagnostic_pcd_done, timeout={timeout:.1f}s)..."
        )
        deadline = time.time() + timeout
        while time.time() < deadline and not self._abort.is_set():
            with self._ros_spin_lock:
                self.rclpy.spin_once(self.scene_pcd_sub, timeout_sec=0.05)
                self.rclpy.spin_once(self.diagnostic_pcd_done_sub, timeout_sec=0.0)
            self._collect_subgoal_pcd_frame()
            if not self.diagnostic_pcd_done_sub.get_flag():
                continue
            done_dir = self.diagnostic_pcd_done_sub.step_log_dir
            self.diagnostic_pcd_done_sub.reset_get_flag()
            if self._same_diagnostic_scan_dir(done_dir, scan_dir):
                self._drain_subgoal_pcd_after_done()
                return True
            self._log(
                "Ignoring diagnostic PCD completion for another dir: "
                f"{done_dir} (expected {scan_dir})"
            )
        return False

    def _same_diagnostic_scan_dir(self, done_dir: str, scan_dir: str) -> bool:
        return equivalent_log_paths(done_dir, scan_dir)

    def _drain_subgoal_pcd_after_done(self) -> None:
        transfer = self._subgoal_pcd_transfer
        if transfer is None:
            return
        drain_seconds = float(
            self.planning_params.get(
                "diagnostic_pcd_done_drain_seconds",
                SUBGOAL_PCD_DONE_DRAIN_SECONDS,
            )
        )
        quiet_seconds = float(
            self.planning_params.get(
                "diagnostic_pcd_quiet_seconds",
                SUBGOAL_PCD_QUIET_SECONDS,
            )
        )
        deadline = time.time() + max(0.0, drain_seconds)
        while time.time() < deadline and not self._abort.is_set():
            with self._ros_spin_lock:
                self.rclpy.spin_once(self.scene_pcd_sub, timeout_sec=0.05)
            self._collect_subgoal_pcd_frame()
            frame_count = int(transfer["frames"])
            last_frame_time = transfer.get("last_frame_time")
            if frame_count > 0 and last_frame_time is not None:
                if time.time() - float(last_frame_time) >= max(0.0, quiet_seconds):
                    break

    def _finish_subgoal_pcd_transfer(self, scan_dir: str, scan_done: bool):
        transfer = self._subgoal_pcd_transfer
        if transfer is None:
            return None
        if not scan_done or transfer["frames"] == 0:
            self._drain_subgoal_pcd_after_done()
        self._subgoal_pcd_transfer = None

        merged = transfer["merged"]
        frame_count = int(transfer["frames"])
        status_path = os.path.join(scan_dir, "desktop_diagnostic_pcd_status.txt")
        if merged.is_empty():
            with open(status_path, "w") as f:
                f.write("failed: desktop_received_no_diagnostic_pcd_frames\n")
                f.write(f"scan_done: {scan_done}\n")
                f.write(f"transferred_frames: {frame_count}\n")
            self._log("No diagnostic PCD frames were received from the NUC.")
            return None

        voxel = float(
            self.planning_params.get(
                "diagnostic_pcd_merge_voxel",
                SUBGOAL_PCD_MERGE_VOXEL,
            )
        )
        if voxel > 0.0:
            merged = merged.voxel_down_sample(voxel)

        merged_path = os.path.join(scan_dir, "diagnostic_pcd_merged_desktop.ply")
        o3d.io.write_point_cloud(merged_path, merged)
        with open(status_path, "w") as f:
            f.write("succeeded\n")
            f.write(f"scan_done: {scan_done}\n")
            f.write(f"transferred_frames: {frame_count}\n")
            f.write(f"merge_voxel: {voxel}\n")
            f.write(f"merged_points: {len(merged.points)}\n")

        self._log(
            "Saved diagnostic PCD preview: "
            f"{frame_count} frame(s), {len(merged.points)} merged points."
        )
        if not scan_done:
            self._log(
                "Diagnostic PCD done flag timed out; rendering transferred frames "
                "received so far."
            )
        return merged

    def _show_subgoal_pcd_preview(
        self,
        merged,
        q,
        phase: str,
        phase_label: str,
        step_idx,
        target_id,
    ) -> None:
        if not self._subgoal_pcd_prompt_is_active(phase, step_idx, target_id):
            self._log(
                "Diagnostic PCD saved, but not rendered because the "
                f"{phase_label} prompt already ended."
            )
            return

        pcd = o3d.geometry.PointCloud(merged)
        voxel = float(
            self.planning_params.get(
                "diagnostic_pcd_display_voxel",
                SUBGOAL_PCD_DISPLAY_VOXEL,
            )
        )
        if voxel > 0.0:
            pcd = pcd.voxel_down_sample(voxel)
        if pcd.is_empty():
            self._log("Diagnostic PCD preview is empty after display downsample.")
            return

        pcd.paint_uniform_color(SUBGOAL_PCD_COLOR)
        entry = (
            pcd,
            SUBGOAL_PCD_COLOR,
            {"name": "subgoal_diagnostic_pcd", "alpha": 0.95},
        )
        self._replace_subgoal_pcd_overlay(entry)
        step_text = "?" if step_idx is None else str(step_idx + 1)
        stone_text = "?" if target_id is None else str(target_id)
        self._log(
            "Diagnostic PCD GUI preview: "
            f"{len(pcd.points)} point(s), display_voxel={voxel}, "
            f"step={step_text}, stone={stone_text}, phase={phase_label}."
        )
        self._on_display_geometries(
            self._base_scene(q) + self._viz_overlay_entries + [self.origin_frame],
            f"Diagnostic PCD: step {step_text} {phase_label}",
        )

    def _replace_subgoal_pcd_overlay(self, entry) -> None:
        self._remove_overlay_entries_by_name("subgoal_diagnostic_pcd")
        self._viz_overlay_entries = self._viz_overlay_entries + [entry]
        self._last_live_layout_key = None

    def _subgoal_pcd_prompt_is_active(self, phase, step_idx, target_id) -> bool:
        if self._pick_adjust_q is None:
            return False
        if str(self._pick_adjust_phase or "") != str(phase):
            return False
        if step_idx is not None and self._current_step_index != step_idx:
            return False
        if target_id is not None and self._current_target_id != target_id:
            return False
        return True

    def _clear_subgoal_pcd_overlay(self, q=None, title: str | None = None) -> None:
        if not self._remove_overlay_entries_by_name("subgoal_diagnostic_pcd"):
            return
        if q is None:
            return
        self._on_display_geometries(
            self._base_scene(q) + self._viz_overlay_entries + [self.origin_frame],
            title or "Diagnostic PCD cleared",
        )
