"""Execution perception: in-hand scans, field recovery, and pose-ID replanning."""

from .execution_common import *


class ExecutionPerceptionMixin:
    def _run_inhand_perception_phase(
        self,
        n_step,
        target_id,
        pick_config,
        place_config,
        result,
        place_body_id,
    ):
        if not PERCEPTION_ON:
            return None

        self._set_status(phase="Pose identification review")
        decision = self._wait_for_decision(
            "pose_identification",
            {"step": n_step + 1, "stone_id": target_id},
        )
        if decision == "skip":
            self._log(
                "Skipping in-hand pose identification; using original place trajectory."
            )
            self._publish_phase(self.Phase.PLACE, count=1000)
            return None
        if decision == "abort":
            return "abort"

        self._set_status(phase="In-hand scan")
        step_log_dir = f"{self.base_log_dir}/step_{n_step + 1}"

        inhand_T, pick_T, opening_angle = self._current_inhand_pose(result, pick_config)

        self._viz_target_mode = "in_hand"
        self._viz_opening_angle = opening_angle
        q_inhand_scan_8d = np.concatenate(
            [self.Q_SCAN_INHAND, [opening_angle, opening_angle]]
        )
        self._on_display_geometries(
            [(self.ground_mesh, C_GROUND)]
            + self._excavator_at(q_inhand_scan_8d)
            + self._other_stone_entries()
            + self._live_target_stone_entries(q_inhand_scan_8d)
            + self._ghost_entries()
            + self._target_structure_entries()
            + [self.origin_frame],
            f"Step {n_step + 1}: in-hand scan",
        )

        self._log("Requesting in-hand scan on nuc...")
        self._start_requested_scan_pcd_transfer(
            os.path.join(step_log_dir, "inhand_scan_pcd"),
            "inhand_scan",
        )
        inhand_scan_done = False
        try:
            grasp_success, inhand_T_opt, opening_angle_opt, recovered_pose = (
                self._get_inhand_pose_with_live_viewer(
                    inhand_T,
                    pick_T,
                    target_id,
                    opening_angle,
                    step_log_dir,
                )
            )
            inhand_scan_done = True
        finally:
            self._finish_requested_scan_pcd_transfer(inhand_scan_done)

        if not grasp_success:
            if recovered_pose is None:
                self._log(
                    "Grasp failure recovery did not return a pose for the "
                    "current target; aborting to avoid replanning with a "
                    "stale or mismatched field pose."
                )
                return "abort"
            self._recover_failed_grasp(
                target_id, recovered_pose, place_body_id=place_body_id
            )
            self._set_regrasp_start_pose_from_live_joint_state()
            return "retry"

        self._log("In-hand pose optimization done")
        self._publish_phase(self.Phase.PLACE, count=1000)
        opening_angle = self._updated_inhand_opening_angle(
            opening_angle,
            opening_angle_opt,
        )
        self._log_inhand_delta(inhand_T_opt, inhand_T)
        self._viz_inhand_T = inhand_T_opt
        self._set_place_inhand_override(inhand_T_opt, opening_angle)
        self._manual_place_inhand_T = inhand_T_opt
        self._manual_place_opening_angle = opening_angle
        self._show_inhand_pose_delta_preview(
            target_id, result, inhand_T, inhand_T_opt, opening_angle
        )

        if not self._needs_inhand_replan(inhand_T_opt, inhand_T):
            self._log(
                "In-hand pose is within tolerance; using the original place trajectory."
            )
            return None

        self._set_status(phase="In-hand replan review")
        self._manual_place.clear()
        decision = self._wait_for_decision(
            "replan",
            {"step": n_step + 1, "stone_id": target_id},
        )
        if decision in {"skip", "manual_place"}:
            if decision == "manual_place":
                self._manual_place.clear()
                return "manual_place"
            self._log("Skipping in-hand replan; using the original place trajectory.")
            return None
        if decision == "abort":
            return "abort"
        selected_replan_mode = "direct" if decision == "continue" else "regrasp"
        self._log(f"Selected in-hand replan mode: {selected_replan_mode}")

        return self._run_inhand_replan_phase(
            n_step,
            target_id,
            pick_config,
            place_config,
            result,
            inhand_T,
            inhand_T_opt,
            opening_angle,
            step_log_dir,
            place_body_id,
            selected_replan_mode,
        )

    def _current_inhand_pose(self, result, pick_config):
        gripper_T = result.grasp_sequence[0].pose.as_matrix()
        pick_T = pick_config.pose.as_matrix()
        inhand_T = np.linalg.inv(gripper_T) @ pick_T
        opening_angle = result.grasp_sequence[0].opening_angle
        return inhand_T, pick_T, opening_angle

    def _get_inhand_pose_with_live_viewer(
        self, inhand_T, pick_T, target_id, opening_angle, step_log_dir
    ):
        def publish_inhand_request():
            self._publish_log_and_phase(step_log_dir, self.Phase.INHANDSCAN, count=3)

        with self._pause_live_joint_polling():
            return self.integrated_planner.get_inhand_pose(
                inhand_T,
                pick_T,
                target_id,
                opening_angle,
                live_joint_node=self._live_joint_node_for_viewer(),
                live_state_cb=self._live_joint_state_cb(),
                spin_lock=self._ros_spin_lock,
                recovery_status_cb=self._field_recovery_status_cb(
                    target_id,
                    pick_T,
                ),
                request_publish_cb=publish_inhand_request,
                pcd_spin_cb=self._spin_requested_scan_pcd_transfer,
                return_opening_angle=True,
            )

    def _updated_inhand_opening_angle(self, planned_angle, identified_angle):
        planned_angle = float(planned_angle)
        if identified_angle is None:
            self._log(
                "No identified in-hand opening angle received; using planned "
                f"grasp opening angle {planned_angle:.4f} rad."
            )
            return planned_angle
        identified_angle = float(identified_angle)
        if not np.isfinite(identified_angle):
            self._log(
                "Identified in-hand opening angle was non-finite; using planned "
                f"grasp opening angle {planned_angle:.4f} rad."
            )
            return planned_angle
        self._log(
            "Updated in-hand opening angle from pose identification: "
            f"{planned_angle:.4f} -> {identified_angle:.4f} rad"
        )
        return identified_angle

    def _field_recovery_status_cb(self, target_id, field_T):
        def _callback(status, stone_id, detail):
            if int(stone_id) != int(target_id):
                return
            detail_text = f" ({detail})" if detail else ""
            self._log(
                "Field recovery status from nuc: "
                f"{status} for stone {stone_id}{detail_text}"
            )
            if status == "started":
                self._show_field_recovery_started_preview(target_id, field_T)

        return _callback

    def _show_field_recovery_started_preview(self, target_id, field_T):
        self._viz_target_id = target_id
        self._viz_target_mode = "pick_pose"
        self._viz_pick_pose_T = np.asarray(field_T, dtype=np.float64).copy()
        self._viz_place_pose_T = None
        self._viz_inhand_T = None
        self._viz_inhand_T_init = None
        self._viz_opening_angle = 0.0
        scan_qs = self._field_scan_preview_qs(field_T)
        q_preview = scan_qs[0] if scan_qs else self.q_home
        scan_line = self._trajectory_lineset(
            [self._grip_body_world(q)[:3, 3] for q in scan_qs],
            C_SCENEID_GROUND,
        )
        scan_entries = [] if scan_line is None else [scan_line]

        self._on_display_geometries(
            self._base_scene(q_preview, opening_angle=0.0)
            + scan_entries
            + [self.origin_frame],
            "Field recovery scan",
        )

    def _field_scan_preview_qs(self, field_T):
        field_T = np.asarray(field_T, dtype=np.float64)
        if field_T.shape != (4, 4):
            return [np.asarray(self.q_home, dtype=np.float64).copy()]
        pos = field_T[:3, 3]
        q0 = float(np.arctan2(pos[1], pos[0]))
        tail = np.array([np.pi / 4, -np.pi / 2, 0.70, 0.0, 0.0], dtype=np.float64)
        offsets = np.linspace(-np.pi / 12, np.pi / 12, 3)
        return [
            np.concatenate([[q0 + float(offset)], tail]).astype(np.float64)
            for offset in offsets
        ]

    def _recover_failed_grasp(self, target_id, recovered_pose, place_body_id):
        self._log(
            "Grasp failure reported by nuc. Updating field pose and retrying "
            "step with re-identified stone pose."
        )
        pos_new, quat_new = recovered_pose
        self._save_pose_data_snapshot(
            target_id,
            pos_new,
            quat_new,
            reason="grasp_failure_field_recovery",
        )
        self._log(
            "Recovered pick pose: "
            f"pos {np.asarray(pos_new).tolist()}, quat {np.asarray(quat_new).tolist()}"
        )
        self.stone_configs[target_id].pose.setPosition(pos_new)
        self.stone_configs[target_id].pose.setOrientation(quat_new)

        if place_body_id is not None:
            self.context.remove_body(place_body_id)
            self.place_body_ids_by_stone.pop(target_id, None)
        self.pick_ids[target_id] = self.context.add_pick_body(
            self.stone_configs[target_id]
        )
        self._update_scene_mesh_from_config(target_id)
        self._viz_target_mode = "pick_pose"
        self._viz_pick_pose_T = self.stone_configs[target_id].pose.as_matrix()
        self._viz_inhand_T = None
        self._viz_inhand_T_init = None
        self._viz_opening_angle = 0.0

    def _reset_regrasp_start_pose(self):
        self._regrasp_q_start = None
        self._regrasp_q_mid = None
        self._regrasp_q_end = None

    def _current_regrasp_start_poses(self):
        if self._regrasp_q_start is None:
            q = np.asarray(self.q_home, dtype=np.float64)
            return q, q, q
        return self._regrasp_q_start, self._regrasp_q_mid, self._regrasp_q_end

    def _set_regrasp_start_pose_from_live_joint_state(self):
        q_start = self._current_desktop_joint_state()
        if q_start is None:
            self._log(
                "Live joint state was not available on the desktop; "
                "retry planning will use q_home."
            )
            self._reset_regrasp_start_pose()
            return

        q_start[1:] = np.asarray(self.q_home, dtype=np.float64)[1:]
        q_mid = np.asarray(
            self.planning_params.get("field_scan_regrasp_q_mid", q_start),
            dtype=np.float64,
        )
        q_end = np.asarray(
            self.planning_params.get("field_scan_regrasp_q_end", q_mid),
            dtype=np.float64,
        )
        self._regrasp_q_start = q_start
        self._regrasp_q_mid = q_mid
        self._regrasp_q_end = q_end
        self._log(
            "Retry regrasp start poses from desktop live joint state: "
            f"q_start={q_start.tolist()}, q_mid={q_mid.tolist()}, "
            f"q_end={q_end.tolist()}"
        )

    def _current_desktop_joint_state(self):
        node = getattr(self, "joint_node_sub", None)
        if node is None:
            return None
        try:
            with self._ros_spin_lock:
                self.rclpy.spin_once(node, timeout_sec=0.1)
        except Exception:
            pass
        if not getattr(node, "get_flag", lambda: False)():
            return None
        q = np.asarray(node.pos.copy(), dtype=np.float64)
        node.reset_get_flag()
        if q.ndim != 1 or q.shape[0] < 6 or not np.all(np.isfinite(q[:6])):
            return None
        return q[:6].copy()

    def _run_inhand_replan_phase(
        self,
        n_step,
        target_id,
        pick_config,
        place_config,
        initial_result,
        inhand_T,
        inhand_T_opt,
        opening_angle,
        step_log_dir,
        place_body_id,
        inhand_replan_mode,
    ):
        if inhand_replan_mode == "direct":
            mode_order = ["direct"]
        else:
            mode_order = ["regrasp", "direct"]

        while True:
            retry_requested = False
            failed_modes = []
            for mode_idx, mode in enumerate(mode_order):
                if self._manual_place_requested():
                    return "manual_place"
                self._log("Replanning with the identified inhand pose " f"({mode}).")
                self._log(
                    "Identified in-hand pose used for replan "
                    "(target in gripper frame):\n"
                    + np.array2string(inhand_T_opt, precision=6, suppress_small=True)
                )
                replan_place_config = self._make_inhand_replan_place_config(
                    place_config
                )
                q_start = getattr(self, "q_joint", self.q_home)
                self._log_inhand_replan_request(
                    target_id,
                    mode,
                    q_start,
                    replan_place_config,
                    opening_angle,
                    enable_near_ik=True,
                )
                # Remove the target's own place body around the replan solve.
                # It was committed to the "place" scene at the original pose
                # before pick/perception, so leaving it in makes the grasped
                # motion plan collision-check the held stone against its own
                # placed copy (~full-stone self-overlap at the place pose), which
                # the ALM then shoves the placement far away to escape. Re-added
                # at the original pose right after, so downstream (review /
                # recover / place control) is unchanged.
                had_place_body = place_body_id is not None
                if had_place_body:
                    self.context.remove_body(place_body_id)
                    self.place_body_ids_by_stone.pop(target_id, None)
                    place_body_id = None
                replan_result = self.solve_inhand_grasp_planning(
                    self.context,
                    q_start,
                    self.diffsim.Pose().from_matrix(inhand_T_opt),
                    replan_place_config,
                    opening_angle,
                    self.n_move,
                    self.n_grasp,
                    True,
                    regrasp_xy_pos=self.regrasp_xy_pos,
                    max_num_solutions=self.max_num_regrasp_solutions,
                    inhand_replan_mode=mode,
                )
                if had_place_body:
                    place_body_id = self._commit_place_body(
                        None, target_id, place_config
                    )
                if self._manual_place_requested():
                    return "manual_place"
                if self._planning_result_is_valid(replan_result):
                    reviewed = self._review_inhand_replan_result(
                        n_step,
                        target_id,
                        replan_result,
                        opening_angle,
                        initial_result,
                    )
                    if reviewed == "retry":
                        self._log("Retrying in-hand replan...")
                        retry_requested = True
                        break
                    if reviewed == "intermediate":
                        return self._recover_from_failed_inhand_replan(
                            target_id,
                            pick_config,
                            inhand_T,
                            inhand_T_opt,
                            opening_angle,
                            step_log_dir,
                            place_body_id,
                        )
                    return reviewed

                failed_modes.append(mode)
                self._log(f"In-hand replan failed for mode: {mode}")
                if mode_idx + 1 < len(mode_order):
                    self._log(
                        f"Trying alternate in-hand replan mode: {mode_order[mode_idx + 1]}"
                    )

            if retry_requested:
                continue

            decision = self._choose_failed_inhand_replan_fallback(
                n_step,
                target_id,
                initial_result,
                opening_angle,
                failed_modes,
                inhand_T_opt,
            )
            if decision in {"continue", "manual_place"}:
                if decision == "manual_place":
                    self._manual_place.clear()
                    return "manual_place"
                self._log(
                    "Using the original place trajectory from before pose "
                    "identification."
                )
                return None
            if decision == "abort":
                return "abort"

            return self._recover_from_failed_inhand_replan(
                target_id,
                pick_config,
                inhand_T,
                inhand_T_opt,
                opening_angle,
                step_log_dir,
                place_body_id,
            )

    def _manual_place_requested(self):
        if not self._manual_place.is_set():
            return False
        self._log("Manual place interrupt accepted.")
        self._manual_place.clear()
        return True

    def _choose_failed_inhand_replan_fallback(
        self,
        n_step,
        target_id,
        initial_result,
        opening_angle,
        failed_modes,
        inhand_T_opt=None,
    ):
        self._show_original_place_trajectory(
            n_step,
            target_id,
            initial_result,
            opening_angle,
            inhand_T_opt=inhand_T_opt,
        )
        return self._wait_for_decision(
            "inhand_replan_failed",
            {
                "step": n_step + 1,
                "stone_id": target_id,
                "modes": failed_modes,
            },
        )

    def _review_inhand_replan_result(
        self, n_step, target_id, replan_result, opening_angle, initial_result
    ):
        num_candidates = self._num_regrasp_candidates(replan_result)
        for candidate_idx in range(num_candidates):
            candidate = self._select_regrasp_candidate(replan_result, candidate_idx)
            score = self._regrasp_candidate_score(replan_result, candidate_idx)
            self._log(
                f"Reviewing in-hand replan candidate {candidate_idx + 1}/"
                f"{num_candidates}"
                + (f" (score: {score:.3f})" if score is not None else "")
            )
            self._show_replanned_trajectory(
                n_step, target_id, candidate, opening_angle, initial_result
            )
            decision = self._wait_for_decision(
                "replan_review",
                {
                    "step": n_step + 1,
                    "stone_id": target_id,
                    "candidate": candidate_idx + 1,
                    "num_candidates": num_candidates,
                },
            )
            if decision == "continue":
                self._log("Replanned trajectory accepted.")
                return candidate
            if decision in {"skip", "manual_place"}:
                if decision == "manual_place":
                    self._manual_place.clear()
                    return "manual_place"
                self._log(
                    "Replanned trajectory rejected; using intermediate "
                    "put-down recovery."
                )
                return "intermediate"
            if decision == "abort":
                return "abort"
            if candidate_idx + 1 < num_candidates:
                self._log("Showing next in-hand replan candidate...")
                continue
            self._log(
                "Generated in-hand replan candidates exhausted; "
                "retrying in-hand replan..."
            )
            return "retry"

        self._log("No valid in-hand replan candidates were available.")
        return "retry"

    def _show_original_place_trajectory(
        self,
        n_step,
        target_id,
        result,
        opening_angle,
        inhand_T_opt=None,
    ):
        target_path = self._place_target_path_matrices(result)
        trajectory_markers = self._target_trajectory_markers(
            target_path, C_INITIAL_TRAJECTORY
        )
        settled_release_entries = self._settled_release_overlay_entries(
            target_id, result
        )
        trajectory_lines = []
        target_line = self._trajectory_lineset(
            [pose_t[:3, 3] for pose_t in target_path],
            C_INITIAL_TRAJECTORY,
        )
        if target_line is not None:
            trajectory_lines.append(target_line)

        q_points = []
        try:
            q_segments = self._split_place_paths(result)
            q_points = [q_t for segment in q_segments for q_t in segment]
            gripper_line = self._trajectory_lineset(
                [self._grip_body_world(q_t)[:3, 3] for q_t in q_points],
                C_GRIPPER,
            )
            if gripper_line is not None:
                trajectory_lines.append(gripper_line)
        except Exception:
            pass

        q_release = self._place_release_joint_state(result)
        expected_place_T = None
        result_place_T = target_path[-1] if target_path else None
        if inhand_T_opt is not None:
            expected_place_T = self._target_pose_from_gripper_inhand(
                q_release,
                inhand_T_opt,
                opening_angle,
            )

        terminal_pose_entries = []
        if result_place_T is not None:
            terminal_pose_entries.append(
                (
                    self._stone_geometry(target_id, result_place_T),
                    C_INITIAL_TRAJECTORY,
                    {"name": "original_intended_place_pose"},
                    "wireframe",
                )
            )
        if expected_place_T is not None:
            terminal_pose_entries.append(
                (
                    self._stone_geometry(target_id, expected_place_T),
                    C_REPLANNED_TRAJECTORY,
                    {"name": "updated_inhand_expected_place_pose", "alpha": 0.90},
                )
            )
            self._log(
                "Original fallback preview: blue wireframe is the intended "
                "place pose; red solid stone is gB_gripper_release * inhand_new."
            )
            if result_place_T is not None:
                delta_line = self._trajectory_lineset(
                    [result_place_T[:3, 3], expected_place_T[:3, 3]],
                    C_SCENEID_DELTA,
                )
                if delta_line is not None:
                    terminal_pose_entries.append(delta_line)
                translation, rotation, _ = self._pose_delta(
                    result_place_T,
                    expected_place_T,
                )
                self._log(
                    "Original fallback terminal pose comparison: "
                    f"intended vs updated-in-hand expected delta "
                    f"{translation:.3f} m, {rotation:.1f} deg"
                )
            self._log(
                "Original fallback updated-in-hand expected pose:\n"
                + np.array2string(expected_place_T, precision=6, suppress_small=True)
            )

        self._viz_target_id = target_id
        self._viz_target_mode = "place_pose"
        if expected_place_T is not None:
            self._viz_place_pose_T = expected_place_T
        elif result_place_T is not None:
            self._viz_place_pose_T = result_place_T
        self._viz_opening_angle = opening_angle
        self._viz_overlay_entries = (
            trajectory_lines
            + trajectory_markers
            + settled_release_entries
            + terminal_pose_entries
        )
        base_scene = (
            [(self.ground_mesh, C_GROUND)]
            + self._excavator_at(q_release, opening_angle=opening_angle)
            + self._other_stone_entries()
            + self._target_structure_entries()
        )
        self._on_display_geometries(
            base_scene + [self.origin_frame] + self._viz_overlay_entries,
            f"Step {n_step + 1}: original place trajectory",
        )

    def _make_inhand_replan_place_config(self, place_config):
        replan_place_config = copy.deepcopy(place_config)
        pos = np.array(replan_place_config.pose.position(), dtype=np.float64, copy=True)
        pos[2] += self.inhand_replan_place_z_offset
        replan_place_config.pose.setPosition(pos)
        self._log(
            "Applying z-offset for in-hand place replan: "
            f"{self.inhand_replan_place_z_offset:.3f} m"
        )
        return replan_place_config

    def _recover_from_failed_inhand_replan(
        self,
        target_id,
        pick_config,
        inhand_T,
        inhand_T_opt,
        opening_angle,
        step_log_dir,
        place_body_id,
    ):
        self._log(
            "Replanning with identified inhand pose failed; putting the stone "
            "down at the intermediate regrasp position and recovering its field pose."
        )
        if place_body_id is not None:
            self.context.remove_body(place_body_id)
            self.place_body_ids_by_stone.pop(target_id, None)

        inhand_candidates = [("identified in-hand pose", inhand_T_opt)]
        if inhand_T is not None and not np.allclose(inhand_T, inhand_T_opt):
            inhand_candidates.append(("original planned in-hand pose", inhand_T))

        y_offset_increment = 0.05
        intermediate_result = None
        intermediate_config = None
        for attempt in range(INTERMEDIATE_PUTDOWN_MAX_ATTEMPTS):
            y_offset = attempt * y_offset_increment
            intermediate_config = self._make_intermediate_config(pick_config)
            pos = np.array(
                intermediate_config.pose.position(), dtype=np.float64, copy=True
            )
            pos[1] += y_offset
            intermediate_config.pose.setPosition(pos)
            self._log_ground_drop_place_config(
                target_id,
                attempt,
                y_offset,
                intermediate_config,
            )
            self._viz_place_pose_T = intermediate_config.pose.as_matrix()

            for candidate_label, candidate_inhand_T in inhand_candidates:
                intermediate_result = self.solve_inhand_grasp_planning(
                    self.context,
                    self.q_joint,
                    self.diffsim.Pose().from_matrix(candidate_inhand_T),
                    intermediate_config,
                    opening_angle,
                    self.n_move,
                    self.n_grasp,
                    True,
                )
                if self._planning_result_is_valid(intermediate_result):
                    if candidate_label != inhand_candidates[0][0]:
                        self._log(
                            "Intermediate put-down fallback succeeded using "
                            f"{candidate_label} at y-offset {y_offset:.2f}."
                        )
                    break

                self._log(
                    "Intermediate put-down planning failed using "
                    f"{candidate_label} for y-offset {y_offset:.2f}; "
                    "trying fallback or another offset."
                )
                self._log(
                    "The length of the generated sequence was {}.".format(
                        len(intermediate_result.q_path_sequence)
                    )
                )
                self._log(
                    "The is_feasible flag was {}.".format(
                        intermediate_result.is_feasible
                    )
                )
            else:
                continue

            if self._planning_result_is_valid(intermediate_result):
                break
        else:
            self._log(
                "Intermediate put-down planning failed after {} attempts; "
                "aborting this execution.".format(INTERMEDIATE_PUTDOWN_MAX_ATTEMPTS)
            )
            return "abort"

        self._set_scene_stone_pose(target_id, intermediate_config)

        self._publish_phase(self.Phase.PLACE, count=1000)
        self._execute_place_solution(
            intermediate_result, opening_angle, "Intermediate put-down"
        )

        self._log(
            "Intermediate put-down complete. Requesting field pose "
            "re-identification at the intermediate position."
        )
        self._recover_field_pose_after_intermediate_putdown(
            target_id,
            intermediate_config,
            step_log_dir,
            reason="intermediate_field_recovery",
        )
        self._log("Recovered field pose. Retrying this step with regrasp planning.")
        return "retry"

    def _recover_field_pose_after_intermediate_putdown(
        self,
        target_id,
        intermediate_config,
        step_log_dir,
        reason,
        return_q=None,
    ):
        self._publish_log_and_phase(step_log_dir, self.Phase.FIELDSCAN, count=3)
        self._show_field_recovery_started_preview(
            target_id,
            intermediate_config.pose.as_matrix(),
        )
        self._start_requested_scan_pcd_transfer(
            os.path.join(step_log_dir, "field_scan_pcd"),
            "field_scan",
        )
        field_scan_done = False
        live_was_paused = bool(self._live_joint_pause.is_set())
        if live_was_paused:
            self._live_joint_pause.clear()
        try:
            recovered_pose = self.integrated_planner.request_field_pose_recovery(
                intermediate_config.pose.as_matrix(),
                target_id,
                spin_lock=self._ros_spin_lock,
                pcd_spin_cb=self._spin_requested_scan_pcd_transfer,
            )
            field_scan_done = True
        finally:
            if live_was_paused:
                self._live_joint_pause.set()
            self._finish_requested_scan_pcd_transfer(field_scan_done)

        if recovered_pose is None:
            self._log(
                "Field re-identification did not return a recovered pose; "
                "falling back to the generated intermediate pose."
            )
            pos_new = np.array(intermediate_config.pose.position(), dtype=np.float64)
            quat_new = np.array(
                intermediate_config.pose.orientation(), dtype=np.float64
            )
        else:
            pos_new, quat_new = recovered_pose
            self._show_field_pose_recovery_preview(
                target_id, intermediate_config, recovered_pose
            )

        self._return_to_intermediate_regrasp_pose_after_field_scan(return_q)
        return self._apply_intermediate_putdown_pose(
            target_id,
            pos_new,
            quat_new,
            reason,
            return_q,
            "Recovered pick pose after intermediate put-down",
        )

    def _use_generated_intermediate_pose_after_putdown(
        self,
        target_id,
        intermediate_config,
        reason,
        return_q=None,
    ):
        pos = np.array(intermediate_config.pose.position(), dtype=np.float64)
        quat = np.array(intermediate_config.pose.orientation(), dtype=np.float64)
        return self._apply_intermediate_putdown_pose(
            target_id,
            pos,
            quat,
            reason,
            return_q,
            "Generated intermediate pick pose after put-down",
        )

    def _apply_intermediate_putdown_pose(
        self,
        target_id,
        pos_new,
        quat_new,
        reason,
        return_q,
        log_label,
    ):
        self._save_pose_data_snapshot(
            target_id,
            pos_new,
            quat_new,
            reason=reason,
        )
        self._log(
            f"{log_label}: "
            f"pos {np.asarray(pos_new).tolist()}, quat {np.asarray(quat_new).tolist()}"
        )
        self.stone_configs[target_id].pose.setPosition(pos_new)
        self.stone_configs[target_id].pose.setOrientation(quat_new)

        old_pick_id = self.pick_ids.pop(target_id, None)
        if old_pick_id is not None:
            try:
                self.context.remove_body(old_pick_id)
            except Exception:
                pass
        self.pick_ids[target_id] = self.context.add_pick_body(
            self.stone_configs[target_id]
        )
        self._update_scene_mesh_from_config(target_id)

        self._viz_target_mode = "pick_pose"
        self._viz_pick_pose_T = self.stone_configs[target_id].pose.as_matrix()
        self._viz_place_pose_T = None
        self._viz_inhand_T = None
        self._viz_inhand_T_init = None
        self._viz_opening_angle = 0.0
        if return_q is None:
            self._set_regrasp_start_pose_from_live_joint_state()
        else:
            return_q = np.asarray(return_q, dtype=np.float64).copy()[:6]
            self._regrasp_q_start = return_q.copy()
            self._regrasp_q_mid = return_q.copy()
            self._regrasp_q_end = return_q.copy()
            self.q_joint = return_q.copy()
            self._log(
                "Retry regrasp start pose set to planned intermediate pose: "
                f"{return_q.tolist()}"
            )
        return pos_new, quat_new

    def _return_to_intermediate_regrasp_pose_after_field_scan(self, return_q):
        if return_q is None or not ROS_CONTROL_ON:
            return
        return_q = np.asarray(return_q, dtype=np.float64).copy()[:6]
        if return_q.shape[0] != 6 or not np.all(np.isfinite(return_q)):
            self._log(
                "Skipping return to intermediate regrasp pose: invalid joint pose."
            )
            return
        self._set_status(phase="Return to intermediate regrasp pose")
        self._log(
            "Returning excavator to the planned intermediate regrasp pose "
            "after field scan."
        )
        v = np.zeros_like(return_q)
        try:
            self.position_control(
                return_q,
                v,
                self.joint_node_pub,
                self.joint_node_sub,
                error_tol=self.move_control_error_tol,
                time_limit=self.move_control_convergence_time_limit,
                grab="open",
                state_cb=self._live_joint_state_cb(),
                spin_lock=self._ros_spin_lock,
            )
            self.q_joint = return_q.copy()
            if LIVE_JOINT_VIEWER_ON:
                self._live_state_cb(return_q)
        except Exception as exc:
            self._log(
                "Failed to return to intermediate regrasp pose after field scan: "
                f"{exc}"
            )

    def _log_inhand_replan_request(
        self,
        target_id: int,
        mode: str,
        q_start,
        place_config,
        opening_angle: float,
        enable_near_ik: bool,
    ) -> None:
        pos = np.asarray(place_config.pose.position(), dtype=np.float64)
        quat = np.asarray(place_config.pose.orientation(), dtype=np.float64)
        q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)
        regrasp_xy = (
            np.asarray(self.regrasp_xy_pos, dtype=np.float64)
            if str(mode).strip().lower() == "regrasp"
            else None
        )
        regrasp_text = (
            "None"
            if regrasp_xy is None
            else np.array2string(regrasp_xy, precision=6, suppress_small=True)
        )
        self._log(
            "In-hand motion planning request: "
            f"stone={int(target_id)}, mode={mode}, "
            f"enable_near_ik={bool(enable_near_ik)}, "
            f"place_pos={np.array2string(pos, precision=6, suppress_small=True)}, "
            f"place_quat={np.array2string(quat, precision=6, suppress_small=True)}, "
            f"opening_angle={float(opening_angle):.6f}, "
            f"q_start={np.array2string(q_start[:6], precision=6, suppress_small=True)}, "
            f"regrasp_xy={regrasp_text}, "
            f"n_move={int(self.n_move)}, n_grasp={int(self.n_grasp)}, "
            f"max_num_solutions={int(self.max_num_regrasp_solutions)}"
        )

    def _log_ground_drop_place_config(
        self,
        target_id: int,
        attempt: int,
        y_offset: float,
        config,
    ) -> None:
        pos = np.asarray(config.pose.position(), dtype=np.float64)
        quat = np.asarray(config.pose.orientation(), dtype=np.float64)
        pose_T = np.asarray(config.pose.as_matrix(), dtype=np.float64)
        self._log(
            "Ground-drop generated place config "
            f"attempt {attempt + 1}/{INTERMEDIATE_PUTDOWN_MAX_ATTEMPTS} "
            f"for stone {int(target_id)}: "
            f"y_offset={float(y_offset):.3f} m, "
            f"pos={np.array2string(pos, precision=6, suppress_small=True)}, "
            f"quat={np.array2string(quat, precision=6, suppress_small=True)}\n"
            + np.array2string(pose_T, precision=6, suppress_small=True)
        )

    def _show_replanned_trajectory(
        self, n_step, target_id, result, opening_angle, initial_result=None
    ):
        q_path, target_path = self.generate_path_with_opening_angle(
            result, self.n_opening_angle
        )

        if VISUALIZATION_ON:
            save_path = os.path.join(self.video_dir, f"step_{n_step + 1}.mp4")
            self.trajectory_visualization_with_target(
                q_path,
                target_path,
                self.excavator_model,
                self.excavator_meshes,
                self.scene_meshes,
                self._copy_base_stone_mesh(target_id),
                save_path,
                self.camera_center,
                self.camera_position,
            )

        replanned_trajectory = self._target_trajectory_markers(
            target_path, C_REPLANNED_TRAJECTORY
        )
        settled_release_entries = self._settled_release_overlay_entries(
            target_id, result
        )
        initial_trajectory = []
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
        if initial_result is not None:
            initial_target_path = self._place_target_path_matrices(initial_result)
            initial_trajectory = self._target_trajectory_markers(
                initial_target_path, C_INITIAL_TRAJECTORY
            )
            initial_line = self._trajectory_lineset(
                [pose_t[:3, 3] for pose_t in initial_target_path],
                C_INITIAL_TRAJECTORY,
            )
            if initial_line is not None:
                trajectory_lines.append(initial_line)
        q_terminal = q_path[-1] if q_path else result.q_path_sequence[-1][-1]
        result_place_T = self._result_final_target_pose_matrix(result)
        self._viz_target_mode = "place_pose"
        if result_place_T is not None:
            self._viz_place_pose_T = result_place_T
        self._viz_opening_angle = opening_angle
        self._viz_overlay_entries = (
            trajectory_lines
            + initial_trajectory
            + replanned_trajectory
            + settled_release_entries
        )
        self._on_display_geometries(
            self._base_scene(q_terminal)
            + [self.origin_frame]
            + self._viz_overlay_entries,
            f"Step {n_step + 1}: replanned trajectory",
        )

    def _show_inhand_pose_delta_preview(
        self, target_id, result, inhand_T, inhand_T_opt, opening_angle
    ):
        q_home_8d = np.concatenate(
            [np.asarray(self.q_home, dtype=np.float64), [opening_angle, opening_angle]]
        )
        grip_T = self._grip_body_world(q_home_8d)
        grip_T[:3, 3] += INHAND_PREVIEW_RIGHT_OFFSET

        original_geom = self._stone_geometry(target_id, grip_T @ inhand_T)
        optimized_geom = self._stone_geometry(target_id, grip_T @ inhand_T_opt)

        q_gripper = np.ones(2) * opening_angle
        gripper_meshes_t = self.update_urdf_mesh(
            self.gripper_model, self.gripper_meshes, q_gripper
        )
        for mesh in gripper_meshes_t.values():
            mesh.transform(grip_T)

        self._viz_target_id = target_id
        self._viz_target_mode = "in_hand"
        self._viz_inhand_T_init = inhand_T.copy()
        self._viz_inhand_T = inhand_T_opt
        self._viz_opening_angle = opening_angle

        viz = (
            [(self.ground_mesh, C_GROUND)]
            + self._excavator_at(self.q_home, opening_angle=opening_angle)
            + self._other_stone_entries()
            + self._ghost_entries()
            + self._target_structure_entries()
            + [(m, C_GRIPPER) for m in gripper_meshes_t.values()]
            + [(original_geom, C_INHAND_INIT, "wireframe"), (optimized_geom, C_TARGET)]
            + [self.origin_frame]
        )
        self._on_display_geometries(viz, "Identified in-hand pose delta")

    def _show_field_pose_recovery_preview(
        self, target_id, field_init_config, recovered_pose
    ):
        recovered_config = copy.deepcopy(field_init_config)
        pos_new, quat_new = recovered_pose
        recovered_config.pose.setPosition(pos_new)
        recovered_config.pose.setOrientation(quat_new)

        initial_geom = self._stone_geometry(
            target_id, field_init_config.pose.as_matrix()
        )
        recovered_geom = self._stone_geometry(
            target_id, recovered_config.pose.as_matrix()
        )

        self._viz_target_id = target_id
        self._viz_target_mode = "place_pose"
        self._viz_pick_pose_T = None
        self._viz_place_pose_T = recovered_config.pose.as_matrix()
        self._viz_inhand_T = None
        self._viz_inhand_T_init = None
        self._viz_opening_angle = 0.0

        scan_pcd_entries = [
            entry
            for entry in self._viz_overlay_entries
            if (
                isinstance(entry, tuple)
                and len(entry) >= 3
                and isinstance(entry[2], dict)
                and str(entry[2].get("name", "")).endswith("_requested_scan_pcd")
            )
        ]

        viz = (
            [(self.ground_mesh, C_GROUND)]
            + self._excavator_at(self.q_home, opening_angle=0.0)
            + self._other_stone_entries()
            + self._target_structure_entries()
            + scan_pcd_entries
            + [(initial_geom, C_INHAND_INIT, "wireframe"), (recovered_geom, C_TARGET)]
            + [self.origin_frame]
        )
        self._on_display_geometries(viz, "Recovered field pose")
