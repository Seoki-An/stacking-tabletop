"""Place-solution execution, remaining grasp segments, and pose/path conversion helpers."""

from .execution_common import *


class ExecutionMotionSegmentsMixin:
    def _execute_place_solution(self, place_result, opening_angle_closed, phase_label):
        return self._run_remaining_grasp_control(
            place_result, opening_angle_closed, phase_label
        )

    def _run_remaining_grasp_control(
        self,
        result,
        default_opening_angle_closed,
        phase_label,
        n_step=None,
        target_id=None,
        place_body_id=None,
        place_config=None,
    ):
        if not ROS_CONTROL_ON:
            return None

        segments = self._remaining_grasp_segments(result)
        self._set_status(phase=phase_label)
        self._viz_overlay_entries = []
        with self._pause_live_joint_polling():
            for i_segment, (segment_idx, mode) in enumerate(segments, start=1):
                if len(segments) > 1:
                    self._set_status(
                        phase=(
                            f"{phase_label}: {mode.title()} "
                            f"{i_segment}/{len(segments)}"
                        )
                    )

                if mode == "place":
                    opening_angle_closed = self._place_segment_opening_angle(
                        result, segment_idx, default_opening_angle_closed
                    )
                else:
                    opening_angle_closed = self._segment_opening_angle(
                        result, segment_idx, default_opening_angle_closed
                    )
                inhand_T = self._prepare_segment_viewer_state(
                    result, segment_idx, mode, opening_angle_closed
                )
                path1, path2, path3, path4 = self._split_grasp_segment_paths(
                    result, segment_idx
                )
                self._show_segment_execution_preview(
                    result,
                    segment_idx,
                    mode,
                    target_id,
                    path1,
                    path2,
                    path3,
                    path4,
                    opening_angle_closed,
                    phase_label,
                    i_segment,
                    len(segments),
                )
                self.sequential_grasp_control(
                    path1,
                    path2,
                    path3,
                    path4,
                    self.joint_node_pub,
                    self.joint_node_sub,
                    mode,
                    confirm_cb=self._segment_confirm_cb(mode),
                    pre_grasp_confirm_cb=self._segment_pre_grasp_confirm_cb(mode),
                    state_cb=self._live_joint_state_cb(),
                    phase_cb=self._segment_phase_cb(
                        mode, opening_angle_closed, inhand_T
                    ),
                    spin_lock=self._ros_spin_lock,
                    move_error_tol=self.move_control_error_tol,
                    move_convergence_time_limit=(
                        self.move_control_convergence_time_limit
                    ),
                )
                if self._segment_needs_final_place_recommit(
                    segments,
                    i_segment - 1,
                    mode,
                    target_id,
                    place_config,
                ):
                    place_body_id = self._commit_final_place_after_intermediate_pick(
                        target_id,
                        place_body_id,
                        place_config,
                    )
                if self._segment_needs_intermediate_field_recovery(
                    segments,
                    i_segment - 1,
                    mode,
                ):
                    intermediate_q = self._segment_end_joint_state(
                        path1,
                        path2,
                        path3,
                        path4,
                    )
                    outcome = self._recover_regrasp_intermediate_field_pose(
                        result,
                        segment_idx,
                        n_step,
                        target_id,
                        place_body_id,
                        place_config,
                        intermediate_q,
                    )
                    if outcome == "direct_done":
                        return None
                    if outcome == "abort":
                        return "abort"
        return None

    @staticmethod
    def _segment_needs_intermediate_field_recovery(segments, segment_i, mode):
        if mode != "place":
            return False
        if segment_i + 1 >= len(segments):
            return False
        return segments[segment_i + 1][1] == "pick"

    @staticmethod
    def _segment_needs_final_place_recommit(
        segments,
        segment_i,
        mode,
        target_id,
        place_config,
    ):
        if mode != "pick" or target_id is None or place_config is None:
            return False
        if segment_i + 1 >= len(segments):
            return False
        return segments[segment_i + 1][1] == "place"

    def _commit_final_place_after_intermediate_pick(
        self,
        target_id,
        place_body_id,
        place_config,
    ):
        target_id = int(target_id)
        pick_id = self.pick_ids.pop(target_id, None)
        if pick_id is not None:
            try:
                self.context.remove_body(pick_id)
            except Exception:
                pass
        return self._commit_place_body(place_body_id, target_id, place_config)

    def _recover_regrasp_intermediate_field_pose(
        self,
        result,
        segment_idx,
        n_step,
        target_id,
        place_body_id,
        place_config,
        intermediate_q,
    ):
        if n_step is None or target_id is None:
            self._log(
                "Skipping intermediate field pose identification: "
                "missing step or target context."
            )
            return None
        target_id = int(target_id)
        intermediate_T = self._segment_settled_or_release_target_pose_matrix(
            result,
            segment_idx,
        )
        if intermediate_T is None:
            self._log(
                "Skipping intermediate field pose identification: "
                "could not recover the intermediate target pose from the plan."
            )
            return None

        if place_body_id is not None:
            try:
                self.context.remove_body(place_body_id)
            except Exception:
                pass
            self.place_body_ids_by_stone.pop(target_id, None)

        intermediate_config = copy.deepcopy(self.stone_configs[target_id])
        intermediate_pose = self._pose_array_from_matrix(intermediate_T)
        intermediate_config.pose.setPosition(intermediate_pose[:3])
        intermediate_config.pose.setOrientation(intermediate_pose[3:7])
        self._set_scene_stone_pose(target_id, intermediate_config)

        step_log_dir = f"{self.base_log_dir}/step_{int(n_step) + 1}"
        self._log(
            "Intermediate regrasp put-down complete. Choose whether to run "
            "field pose re-identification before replanning the remaining "
            "place motion."
        )
        self._set_status(phase="Intermediate field scan review")
        scan_decision = self._wait_for_decision(
            "intermediate_field_scan",
            {"step": int(n_step) + 1, "stone_id": target_id},
        )
        if scan_decision == "abort":
            return "abort"
        if scan_decision == "continue":
            self._log(
                "Requesting field pose re-identification at the intermediate "
                "regrasp pose."
            )
            self._recover_field_pose_after_intermediate_putdown(
                target_id,
                intermediate_config,
                step_log_dir,
                reason="regrasp_intermediate_field_recovery",
                return_q=intermediate_q,
            )
        else:
            self._log(
                "Skipping intermediate field pose re-identification; using "
                "the generated intermediate pose."
            )
            self._use_generated_intermediate_pose_after_putdown(
                target_id,
                intermediate_config,
                reason="regrasp_intermediate_field_recovery_skipped",
                return_q=intermediate_q,
            )
        self._set_status(phase="Intermediate regrasp review")
        decision = self._wait_for_decision(
            "intermediate_regrasp_review",
            {"step": int(n_step) + 1, "stone_id": target_id},
        )
        if decision == "abort":
            return "abort"
        if decision == "continue":
            direct_outcome = self._try_direct_motion_from_intermediate(
                n_step,
                target_id,
                place_config,
                q_start=intermediate_q,
            )
            if direct_outcome == "direct_done":
                return "direct_done"
            if direct_outcome == "abort":
                return "abort"
            self._log(
                "Direct pick-and-place from the identified intermediate pose "
                "failed or was rejected; continuing the original regrasp path."
            )

        self._log("Continuing the original remaining regrasp path.")
        return "continue_original"

    @staticmethod
    def _segment_end_joint_state(path1, path2, path3, path4):
        for path in (path4, path3, path2, path1):
            if not path:
                continue
            q = np.asarray(path[-1], dtype=np.float64).reshape(-1)
            if q.shape[0] >= 6 and np.all(np.isfinite(q[:6])):
                return q[:6].copy()
        return None

    def _show_segment_execution_preview(
        self,
        result,
        segment_idx,
        mode,
        target_id,
        path1,
        path2,
        path3,
        path4,
        opening_angle,
        phase_label,
        i_segment,
        n_segments,
    ):
        if target_id is None:
            return
        q_path = list(path1) + list(path2) + list(path3) + list(path4)
        if not q_path:
            return
        target_path = []
        for idx in (segment_idx, segment_idx + 1):
            paths = list(getattr(result, "target_path_sequence", []) or [])
            if idx < len(paths):
                target_path.extend(
                    self._pose_to_matrix(pose) for pose in (paths[idx] or [])
                )
        trajectory_lines = [
            line
            for line in (
                self._trajectory_lineset(
                    [self._grip_body_world(q_t)[:3, 3] for q_t in q_path],
                    C_GRIPPER,
                ),
                self._trajectory_lineset(
                    [pose_t[:3, 3] for pose_t in target_path],
                    C_REPLANNED_TRAJECTORY,
                ),
            )
            if line is not None
        ]
        target_markers = self._target_trajectory_markers(
            target_path,
            C_REPLANNED_TRAJECTORY,
            max_markers=PLANNING_PREVIEW_MAX_MARKERS,
        )
        self._viz_overlay_entries = trajectory_lines + target_markers
        self._on_display_geometries(
            self._base_scene(q_path[0], opening_angle=opening_angle)
            + [self.origin_frame]
            + self._viz_overlay_entries,
            (
                f"{phase_label}: {mode.title()} "
                f"{i_segment}/{n_segments} trajectory"
            ),
        )

    def _try_direct_motion_from_intermediate(
        self,
        n_step,
        target_id,
        place_config,
        q_start=None,
    ):
        if place_config is None:
            self._log("Direct intermediate replan skipped: no final place config.")
            return "failed"
        target_id = int(target_id)
        pick_config = copy.deepcopy(self.stone_configs[target_id])
        direct_place_config = copy.deepcopy(place_config)
        if q_start is None:
            q_start = self._current_desktop_joint_state()
        if q_start is None:
            q_start = np.asarray(
                getattr(self, "q_joint", self.q_home),
                dtype=np.float64,
            )
        q_start = np.asarray(q_start, dtype=np.float64).copy()[:6]
        self.q_joint = q_start.copy()
        self._log(
            "Trying direct pick-and-place from the identified intermediate "
            f"pose with q_start={q_start.tolist()}."
        )
        direct_result = self.regrasp_planning(
            self.context,
            pick_config,
            direct_place_config,
            q_start,
            self.regrasp_xy_pos,
            self.n_move,
            self.n_grasp,
            1,
            q_mid=q_start,
            q_end=q_start,
        )
        self._log(
            "Intermediate direct motion result: "
            f"{self.motion_result_summary(direct_result)}"
        )
        if (
            not self._planning_result_is_valid(direct_result)
            or len(direct_result.q_path_sequence) != 4
        ):
            return "failed"

        self._show_planning_preview(
            n_step,
            target_id,
            pick_config,
            direct_place_config,
            direct_result,
        )
        decision = self._wait_for_decision(
            "place_retry_review",
            {
                "step": int(n_step) + 1,
                "stone_id": target_id,
                "candidate": 1,
                "num_candidates": 1,
            },
        )
        if decision == "abort":
            return "abort"
        if decision != "continue":
            return "failed"

        place_body_id = None
        pick_id = self.pick_ids.pop(target_id, None)
        if pick_id is not None:
            try:
                self.context.remove_body(pick_id)
            except Exception:
                pass
        place_body_id = self._commit_place_body(
            place_body_id,
            target_id,
            direct_place_config,
        )
        opening_angle = direct_result.grasp_sequence[0].opening_angle
        self._run_pick_control(direct_result, opening_angle)
        place_outcome = self._run_remaining_grasp_control(
            direct_result,
            opening_angle,
            "Direct place from intermediate",
            n_step=n_step,
            target_id=target_id,
            place_body_id=place_body_id,
            place_config=direct_place_config,
        )
        if place_outcome == "abort":
            return "abort"
        return "direct_done"

    def _remaining_grasp_segments(self, result):
        n_paths = len(result.q_path_sequence)
        if n_paths == 2:
            segments = [(0, "place")]
        elif n_paths == 4:
            segments = [(2, "place")]
        elif n_paths == 6:
            segments = [(0, "place"), (2, "pick"), (4, "place")]
        elif n_paths == 8:
            segments = [(2, "place"), (4, "pick"), (6, "place")]
        elif n_paths == 10:
            segments = [
                (0, "place"),
                (2, "pick"),
                (4, "place"),
                (6, "pick"),
                (8, "place"),
            ]
        else:
            raise ValueError(f"Unexpected path sequence length: {n_paths}")
        return segments

    def _split_grasp_segment_paths(self, result, segment_idx):
        path1 = result.q_path_sequence[segment_idx][: self.n_move]
        path2 = result.q_path_sequence[segment_idx][self.n_move :]
        path3 = result.q_path_sequence[segment_idx + 1][: self.n_grasp]
        path4 = result.q_path_sequence[segment_idx + 1][self.n_grasp :]
        path1, path2, path3, path4 = self.normalize_joint_branches(
            [path1, path2, path3, path4], self.q_home
        )
        return path1, path2, path3, path4

    def _segment_opening_angle(self, result, segment_idx, default_opening_angle):
        grasps = list(getattr(result, "grasp_sequence", []) or [])
        if len(grasps) == 0:
            return default_opening_angle
        grasp_idx = min(segment_idx // 2, len(grasps) - 1)
        return grasps[grasp_idx].opening_angle

    def _place_segment_opening_angle(self, result, segment_idx, default_opening_angle):
        override = getattr(self, "_place_opening_angle_override", None)
        if (
            self._should_use_place_inhand_override(result, segment_idx)
            and override is not None
            and np.isfinite(float(override))
        ):
            return float(override)
        return self._segment_opening_angle(result, segment_idx, default_opening_angle)

    def _set_place_inhand_override(self, inhand_T, opening_angle):
        inhand_T = np.asarray(inhand_T, dtype=np.float64)
        if inhand_T.shape != (4, 4) or not np.all(np.isfinite(inhand_T)):
            return
        self._place_inhand_T_override = inhand_T.copy()
        self._place_opening_angle_override = float(opening_angle)
        self._log(
            "Place execution viewer will use the identified in-hand pose "
            "for the original gripper trajectory."
        )

    def _clear_place_inhand_override(self):
        self._place_inhand_T_override = None
        self._place_opening_angle_override = None

    def _place_inhand_override(self):
        inhand_T = getattr(self, "_place_inhand_T_override", None)
        if inhand_T is None:
            return None
        inhand_T = np.asarray(inhand_T, dtype=np.float64)
        if inhand_T.shape != (4, 4) or not np.all(np.isfinite(inhand_T)):
            return None
        return inhand_T

    def _should_use_place_inhand_override(self, result, segment_idx):
        if self._place_inhand_override() is None:
            return False
        for idx, mode in self._remaining_grasp_segments(result):
            if idx == segment_idx:
                return True
            if mode == "pick":
                return False
        return True

    def _segment_confirm_cb(self, mode):
        if mode == "pick":
            return self._confirm_pick_grasp
        if mode == "place":
            return self._confirm_place_grasp
        raise ValueError(f"Unexpected grasp control mode: {mode}")

    def _segment_pre_grasp_confirm_cb(self, mode):
        if mode == "pick":
            return self._confirm_pick_approach
        if mode == "place":
            return self._confirm_place_approach
        raise ValueError(f"Unexpected grasp control mode: {mode}")

    def _segment_phase_cb(self, mode, opening_angle_closed, inhand_T):
        def _phase_cb(phase):
            if mode == "pick":
                if phase in ("move", "approach"):
                    self._viz_target_mode = "pick_pose"
                    self._viz_opening_angle = 0.0
                elif phase in ("grasping", "lift", "retreat", "done"):
                    if inhand_T is not None:
                        self._viz_inhand_T = inhand_T
                    self._viz_target_mode = "in_hand"
                    self._viz_opening_angle = opening_angle_closed
                return

            if phase in ("move", "approach"):
                if inhand_T is not None:
                    self._viz_inhand_T = inhand_T
                self._viz_target_mode = "in_hand"
                self._viz_opening_angle = opening_angle_closed
            elif phase in ("grasping", "lift", "retreat", "done"):
                self._viz_target_mode = "place_pose"
                self._viz_opening_angle = 0.0

        return _phase_cb

    def _prepare_segment_viewer_state(
        self, result, segment_idx, mode, opening_angle_closed
    ):
        target_T = self._segment_target_pose_matrix(result, segment_idx)
        inhand_T = self._segment_inhand_pose(result, segment_idx, target_T)
        self._viz_inhand_T_init = None
        if mode == "pick":
            if target_T is not None:
                self._viz_pick_pose_T = target_T
            self._viz_target_mode = "pick_pose"
            self._viz_opening_angle = 0.0
        elif mode == "place":
            inhand_override = (
                self._place_inhand_override()
                if self._should_use_place_inhand_override(result, segment_idx)
                else None
            )
            if inhand_override is not None:
                inhand_T = inhand_override
                release_T = self._segment_release_target_pose(
                    result,
                    segment_idx,
                    inhand_T,
                    opening_angle_closed,
                )
                if release_T is not None:
                    target_T = release_T
            if target_T is not None:
                self._viz_place_pose_T = target_T
            if inhand_T is not None:
                self._viz_inhand_T = inhand_T
            self._viz_target_mode = "in_hand"
            self._viz_opening_angle = opening_angle_closed
        else:
            raise ValueError(f"Unexpected grasp control mode: {mode}")
        return inhand_T

    def _segment_target_pose_matrix(self, result, segment_idx):
        target_paths = list(getattr(result, "target_path_sequence", []) or [])
        if segment_idx >= len(target_paths) or len(target_paths[segment_idx]) == 0:
            return None
        return self._pose_to_matrix(target_paths[segment_idx][-1])

    def _segment_settled_or_release_target_pose_matrix(self, result, segment_idx):
        settled_paths = list(getattr(result, "settled_target_path_sequence", []) or [])
        if segment_idx < len(settled_paths):
            settled_path = settled_paths[segment_idx]
            if len(settled_path) > 0:
                return self._pose_to_matrix(settled_path[-1])
        return self._segment_target_pose_matrix(result, segment_idx)

    def _result_final_target_pose_matrix(self, result):
        target_paths = list(getattr(result, "target_path_sequence", []) or [])
        for path_sub in reversed(target_paths):
            if len(path_sub) == 0:
                continue
            return self._pose_to_matrix(path_sub[-1])
        return None

    def _pose_matrix_or_none(self, pose):
        try:
            T = np.asarray(self._pose_to_matrix(pose), dtype=np.float64)
        except Exception:
            return None
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return None
        return T

    def _settled_release_pose_matrices(self, result):
        target_paths = list(getattr(result, "target_path_sequence", []) or [])
        settled_paths = list(getattr(result, "settled_target_path_sequence", []) or [])
        release_path = []
        settled_path = []
        for segment_idx, settled_subpath in enumerate(settled_paths):
            if len(settled_subpath) == 0:
                continue
            release_T = None
            if segment_idx < len(target_paths) and len(target_paths[segment_idx]) > 0:
                release_T = self._pose_matrix_or_none(target_paths[segment_idx][-1])
            for settled_pose in settled_subpath:
                settled_T = self._pose_matrix_or_none(settled_pose)
                if settled_T is None:
                    continue
                if release_T is not None:
                    release_path.append(release_T)
                settled_path.append(settled_T)
        return release_path, settled_path

    def _settled_release_overlay_entries(self, target_id, result, max_markers=80):
        release_path, settled_path = self._settled_release_pose_matrices(result)
        entries = []
        entries += [
            (self._stone_geometry(target_id, T), C_BASIN_RELEASE, "wireframe")
            for T in release_path
        ]
        entries += [
            (self._stone_geometry(target_id, T), C_STABLE_POSE, "wireframe")
            for T in settled_path
        ]
        entries += self._target_trajectory_markers(
            release_path, C_BASIN_RELEASE, max_markers=max_markers
        )
        entries += self._target_trajectory_markers(
            settled_path, C_STABLE_POSE, max_markers=max_markers
        )
        return entries

    def _pose_to_matrix(self, pose):
        if hasattr(pose, "as_matrix"):
            return pose.as_matrix()
        return np.asarray(pose, dtype=np.float64)

    def _pose_array_to_matrix(self, pose):
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape[0] != 7 or not np.all(np.isfinite(pose)):
            return None

        quat = pose[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-8:
            return None
        x, y, z, w = quat / quat_norm
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        T = np.eye(4)
        T[:3, :3] = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )
        T[:3, 3] = pose[:3]
        return T

    def _pose_array_from_matrix(self, matrix):
        T = np.asarray(matrix, dtype=np.float64)
        R = T[:3, :3]
        trace = float(np.trace(R))
        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            idx = int(np.argmax(np.diag(R)))
            if idx == 0:
                s = 2.0 * np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0))
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif idx == 1:
                s = 2.0 * np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0))
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0))
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
        quat = np.array([qx, qy, qz, qw], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm > 1e-8:
            quat /= norm
        return np.concatenate([T[:3, 3].copy(), quat])

    def _segment_inhand_pose(self, result, segment_idx, target_T):
        if target_T is None:
            return None
        grasps = list(getattr(result, "grasp_sequence", []) or [])
        if len(grasps) == 0:
            return None
        grasp_idx = min(segment_idx // 2, len(grasps) - 1)
        grasp_T = grasps[grasp_idx].pose.as_matrix()
        return np.linalg.inv(grasp_T) @ target_T

    def _segment_release_joint_state(self, result, segment_idx):
        q_paths = list(getattr(result, "q_path_sequence", []) or [])
        if segment_idx >= len(q_paths):
            return None
        path = q_paths[segment_idx]
        approach_path = path[self.n_move :]
        for q in reversed(approach_path or path):
            q = np.asarray(q, dtype=np.float64)
            if q.ndim == 1 and q.shape[0] >= 6 and np.all(np.isfinite(q[:6])):
                return q[:6].copy()
        return None

    def _segment_release_target_pose(
        self, result, segment_idx, inhand_T, opening_angle
    ):
        q_release = self._segment_release_joint_state(result, segment_idx)
        if q_release is None:
            return None
        return self._target_pose_from_gripper_inhand(
            q_release,
            inhand_T,
            opening_angle,
        )

    def _split_place_paths(self, result):
        return self.split_regrasp_place_paths(
            result, self.n_move, self.n_grasp, q_home=self.q_home
        )

    def _place_release_joint_state(self, result):
        try:
            _, path_approach, _, _ = self._split_place_paths(result)
        except Exception:
            path_approach = []
        for q in reversed(path_approach or []):
            q = np.asarray(q, dtype=np.float64)
            if q.ndim == 1 and q.shape[0] >= 6 and np.all(np.isfinite(q[:6])):
                return q[:6].copy()

        for path in reversed(list(getattr(result, "q_path_sequence", []) or [])):
            for q in reversed(path or []):
                q = np.asarray(q, dtype=np.float64)
                if q.ndim == 1 and q.shape[0] >= 6 and np.all(np.isfinite(q[:6])):
                    return q[:6].copy()
        return np.asarray(self.q_home, dtype=np.float64).copy()

    def _target_pose_from_gripper_inhand(self, q, inhand_T, opening_angle):
        inhand_T = np.asarray(inhand_T, dtype=np.float64)
        if inhand_T.shape != (4, 4) or not np.all(np.isfinite(inhand_T)):
            return None
        grip_T = self._grip_body_world(
            self._q_with_opening(q, opening_angle=opening_angle)
        )
        target_T = grip_T @ inhand_T
        if target_T.shape != (4, 4) or not np.all(np.isfinite(target_T)):
            return None
        return target_T
