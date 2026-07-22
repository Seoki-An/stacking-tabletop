"""Per-step motion planning, saved motion reconstruction, and pick-control helpers."""

from .execution_common import *


class ExecutionStepMixin:
    def _execute_step(self, n_step, action, total):
        target_id = int(action["stone_id"])
        self._reset_regrasp_start_pose()
        self._begin_step_status(n_step, total, target_id)

        place_body_id = None
        while True:  # retry on detected grasp failure
            pick_config, place_config = self._build_pick_and_place_configs(
                n_step, target_id, action
            )

            result = self._plan_step_with_review(
                n_step, target_id, pick_config, place_config
            )
            if result == "abort":
                return False

            place_body_id = self._commit_place_body(
                place_body_id, target_id, place_config
            )
            if n_step < self.start_step - 1:
                return True

            self._publish_phase(self.Phase.GRASP, count=10)
            opening_angle_pick = self._prepare_inhand_viewer_pose(result, pick_config)
            self._run_pick_control(result, opening_angle_pick)

            perception_outcome = self._run_inhand_perception_phase(
                n_step,
                target_id,
                pick_config,
                place_config,
                result,
                place_body_id,
            )
            if perception_outcome == "abort":
                return False
            if perception_outcome == "retry":
                place_body_id = None
                self.place_body_ids_by_stone.pop(target_id, None)
                continue
            if perception_outcome == "manual_place":
                if (
                    self._wait_for_decision("press_enter", {"phase": "manual_place"})
                    == "abort"
                ):
                    return False
                manual_ok = self._run_manual_place_control(
                    target_id,
                    place_body_id,
                    self._manual_place_opening_angle,
                )
                if not manual_ok:
                    return False
                self._finish_step(n_step)
                self._reset_regrasp_start_pose()
                return True
            if perception_outcome is not None:
                result = perception_outcome

            reviewed = self._review_place_control_before_execution(
                n_step,
                target_id,
                pick_config,
                place_config,
                result,
                place_body_id,
                opening_angle_pick,
            )
            if reviewed == "abort":
                return False
            result, place_config, place_body_id = reviewed

            place_outcome = self._run_place_control(
                result,
                opening_angle_pick,
                n_step=n_step,
                target_id=target_id,
                place_body_id=place_body_id,
                place_config=place_config,
            )
            if place_outcome == "retry":
                place_body_id = None
                self.place_body_ids_by_stone.pop(target_id, None)
                continue
            if place_outcome == "abort":
                return False
            self._finish_step(n_step)
            self._reset_regrasp_start_pose()
            return True

    def _begin_step_status(self, n_step, total, target_id):
        self._current_step_index = n_step
        self._current_target_id = target_id
        self._set_status(
            step=n_step + 1, total=total, stone_id=target_id, phase="Planning"
        )
        self._log("=" * 40)
        self._log(f"Step {n_step + 1}/{total} (stone {target_id})")
        self._log("=" * 40)

    def _build_pick_and_place_configs(self, n_step, target_id, action):
        place_pose = self._place_pose_for_action(action, n_step)

        pick_config = copy.deepcopy(self.stone_configs[target_id])
        place_config = copy.deepcopy(pick_config)
        place_config.pose.setPosition(place_pose[:3])
        place_config.pose.setOrientation(place_pose[3:])
        return pick_config, place_config

    def _apply_place_retry_z_offset(self, place_config, base_z: float):
        offset = float(self._place_retry_z_offset)
        if not np.isfinite(offset):
            offset = 0.0
        pos = np.array(place_config.pose.position(), dtype=np.float64, copy=True)
        pos[2] = float(base_z) + offset
        place_config.pose.setPosition(pos)
        if abs(offset) > 1e-9 and offset != self._last_logged_place_retry_z_offset:
            self._log(
                "Applying motion retry place Z offset: "
                f"{offset:+.3f} m (place z={pos[2]:.3f})"
            )
            self._last_logged_place_retry_z_offset = offset

    def _place_pose_for_action(self, action, n_step: int):
        target_id = int(action["stone_id"])
        pose = None
        if n_step < self.start_step - 1:
            pose = self.resume_scene_poses.get(target_id)
        if pose is None:
            pose = action["pose"]
        place_pose = np.asarray(pose, dtype=float).copy()
        if place_pose.ndim != 1 or place_pose.shape[0] < 7:
            raise ValueError(f"Invalid pose for stone {target_id}: {pose!r}")
        place_pose = place_pose[:7]
        place_pose[:2] += self.target_structure_offset[:2]
        return place_pose

    def _saved_motion_result_for_step(self, n_step, pick_config, place_config):
        result = self._saved_motion_result_metadata_for_step(n_step)
        if result is not None:
            return result

        if n_step >= len(getattr(self, "motion_sequence", []) or []):
            return None
        motions = self.motion_sequence[n_step]
        paths = self._coerce_saved_motion_paths(motions)
        if paths is None:
            self._log(
                f"Ignoring saved motion for step {n_step + 1}: "
                "expected four finite joint-path chunks."
            )
            return None
        try:
            return self._make_saved_motion_result(paths, pick_config, place_config)
        except Exception as exc:
            self._log(f"Could not replay saved motion for step {n_step + 1}: {exc}")
            return None

    def _saved_motion_result_metadata_for_step(self, n_step):
        if n_step >= len(getattr(self, "motion_result_sequence", []) or []):
            return None
        data = self.motion_result_sequence[n_step]
        if not isinstance(data, dict):
            return None
        try:
            return self._make_saved_motion_result_from_metadata(data)
        except Exception as exc:
            self._log(
                f"Could not load saved motion result metadata for step "
                f"{n_step + 1}: {exc}"
            )
            return None

    def _make_saved_motion_result_from_metadata(self, data: dict):
        q_path_sequence = self._coerce_q_path_sequence(data.get("q_path_sequence", []))
        target_path_sequence = self._coerce_pose_path_sequence(
            data.get("target_path_sequence", [])
        )
        settled_target_path_sequence = self._coerce_pose_path_sequence(
            data.get("settled_target_path_sequence", [])
        )
        grasp_sequence = self._coerce_grasp_sequence(data.get("grasp_sequence", []))
        if not q_path_sequence:
            raise ValueError("empty q_path_sequence")
        if not target_path_sequence:
            raise ValueError("empty target_path_sequence")
        if not grasp_sequence:
            raise ValueError("empty grasp_sequence")
        return SimpleNamespace(
            is_feasible=bool(data.get("is_feasible", True)),
            q_path_sequence=q_path_sequence,
            target_path_sequence=target_path_sequence,
            settled_target_path_sequence=settled_target_path_sequence,
            grasp_sequence=grasp_sequence,
            q_path_sequences=[q_path_sequence],
            target_path_sequences=[target_path_sequence],
            settled_target_path_sequences=[settled_target_path_sequence],
            grasp_sequences=[grasp_sequence],
            is_feasible_sequence=[bool(data.get("is_feasible", True))],
            scores=[float(score) for score in data.get("scores", []) or []],
            failure_stage=str(data.get("failure_stage", "") or ""),
            failure_detail=str(data.get("failure_detail", "") or ""),
            failure_stage_sequence=[str(data.get("failure_stage", "") or "")],
            replayed_saved_motion=True,
        )

    @staticmethod
    def _coerce_q_path_sequence(path_sequence):
        out = []
        for path in path_sequence or []:
            q_path = []
            for q in path or []:
                arr = np.asarray(q, dtype=np.float64)
                if arr.ndim == 1 and arr.shape[0] >= 6 and np.all(np.isfinite(arr[:6])):
                    q_path.append(arr[:6].copy())
            out.append(q_path)
        return out

    def _coerce_pose_path_sequence(self, path_sequence):
        out = []
        for path in path_sequence or []:
            pose_path = []
            for matrix in path or []:
                pose_path.append(self._pose_from_matrix(matrix))
            out.append(pose_path)
        return out

    def _coerce_grasp_sequence(self, grasp_sequence):
        out = []
        for item in grasp_sequence or []:
            if not isinstance(item, dict):
                continue
            grasp = self.planner.Grasp()
            grasp.pose = self._pose_from_matrix(item.get("pose"))
            grasp.opening_angle = float(item.get("opening_angle", 0.0))
            out.append(grasp)
        return out

    @staticmethod
    def _coerce_saved_motion_paths(motions):
        if not isinstance(motions, (list, tuple)) or len(motions) != 4:
            return None
        paths = []
        for path in motions:
            if not isinstance(path, (list, tuple)) or len(path) == 0:
                return None
            q_path = []
            for q in path:
                arr = np.asarray(q, dtype=np.float64)
                if arr.ndim != 1 or arr.shape[0] < 6 or not np.all(np.isfinite(arr)):
                    return None
                q_path.append(arr[:6].copy())
            paths.append(q_path)
        return paths

    def _make_saved_motion_result(self, paths, pick_config, place_config):
        q_path_sequence = [paths[0] + paths[1], paths[2] + paths[3]]
        opening_angle = float(
            self.planning_params.get("saved_motion_opening_angle", -0.01)
        )
        q_at_grasp = q_path_sequence[0][-1]
        grip_T = self._grip_body_world(
            self._q_with_opening(q_at_grasp, opening_angle=opening_angle)
        )

        grasp = self.planner.Grasp()
        grasp.pose = self._pose_from_matrix(grip_T)
        grasp.opening_angle = opening_angle

        pick_T = pick_config.pose.as_matrix()
        place_T = place_config.pose.as_matrix()
        inhand_T = np.linalg.inv(grip_T) @ pick_T
        target_path_sequence = [
            [self._pose_from_matrix(pick_T) for _ in q_path_sequence[0]],
            [
                self._pose_from_matrix(
                    self._saved_motion_target_pose(q, inhand_T, opening_angle)
                )
                for q in q_path_sequence[1]
            ],
        ]
        if target_path_sequence[1]:
            target_path_sequence[1][-1] = self._pose_from_matrix(place_T)

        return SimpleNamespace(
            is_feasible=True,
            q_path_sequence=q_path_sequence,
            target_path_sequence=target_path_sequence,
            settled_target_path_sequence=[[], []],
            grasp_sequence=[grasp],
            q_path_sequences=[q_path_sequence],
            target_path_sequences=[target_path_sequence],
            settled_target_path_sequences=[[[], []]],
            grasp_sequences=[[grasp]],
            is_feasible_sequence=[True],
            scores=[0.0],
            failure_stage="",
            failure_detail="",
            failure_stage_sequence=[""],
            replayed_saved_motion=True,
        )

    def _saved_motion_target_pose(self, q, inhand_T, opening_angle):
        grip_T = self._grip_body_world(self._q_with_opening(q, opening_angle))
        return grip_T @ inhand_T

    def _pose_from_matrix(self, matrix):
        return self.diffsim.Pose().from_matrix(np.asarray(matrix, dtype=np.float64))

    def _plan_step_with_review(self, n_step, target_id, pick_config, place_config):
        self._log("Start motion planning...")
        self.context.remove_body(self.pick_ids[target_id])
        reviewed_saved_motion = False
        base_place_z = float(place_config.pose.position()[2])
        self._last_logged_place_retry_z_offset = None

        while True:
            if n_step < self.start_step - 1:
                self._log(
                    f"Skipping motion planning for step {n_step + 1} "
                    f"(before start_step {self.start_step})."
                )
                return None
            if self._abort.is_set():
                return "abort"

            saved_result = None
            if not reviewed_saved_motion:
                saved_result = self._saved_motion_result_for_step(
                    n_step, pick_config, place_config
                )
            if saved_result is not None:
                reviewed_saved_motion = True
                if n_step in getattr(
                    self,
                    "_online_auto_accept_saved_motion_steps",
                    set(),
                ):
                    self._log("Using previously accepted planned seed motion.")
                    self._set_last_motion_joint_state(saved_result)
                    return saved_result
                self._log("Reviewing motion generated by generate_sequence.py.")
                with self._pause_live_joint_polling():
                    self._show_planning_preview(
                        n_step, target_id, pick_config, place_config, saved_result
                    )
                    decision = self._wait_for_decision(
                        "plan_review",
                        {"step": n_step + 1, "stone_id": target_id},
                    )
                if decision == "continue":
                    self._log("Saved generated motion accepted.")
                    self._set_last_motion_joint_state(saved_result)
                    return saved_result
                if decision == "abort":
                    return "abort"
                self._log("Saved generated motion rejected; replanning motion...")

            self._apply_place_retry_z_offset(place_config, base_place_z)
            q_start, q_mid, q_end = self._current_regrasp_start_poses()
            self.q_joint = q_end
            # Single regrasp_xy position (multi-position retry removed).
            result = self.regrasp_planning(
                self.context,
                pick_config,
                place_config,
                q_start,
                self.regrasp_xy_pos,
                self.n_move,
                self.n_grasp,
                self.max_num_regrasp_solutions,
                q_mid=q_mid,
                q_end=q_end,
            )
            if result.is_feasible:
                if len(result.q_path_sequence) == 2:
                    self._log("Direct motion planning succeeded.")
                else:
                    self._log("Fallback regrasp planning succeeded.")
            self._log(f"Motion planning result: {self.motion_result_summary(result)}")
            if len(result.q_path_sequence) == 0 or result.is_feasible is False:
                self._log("Planning failed; retrying...")
                continue

            num_candidates = self._num_regrasp_candidates(result)
            for candidate_idx in range(num_candidates):
                candidate = self._select_regrasp_candidate(result, candidate_idx)
                score = self._regrasp_candidate_score(result, candidate_idx)
                self._log(
                    f"Reviewing motion candidate {candidate_idx + 1}/"
                    f"{num_candidates}"
                    + (f" (score: {score:.3f})" if score is not None else "")
                )

                with self._pause_live_joint_polling():
                    self._show_planning_preview(
                        n_step, target_id, pick_config, place_config, candidate
                    )
                    decision = self._wait_for_decision(
                        "plan_review",
                        {"step": n_step + 1, "stone_id": target_id},
                    )
                if decision == "continue":
                    self._log("Plan accepted.")
                    self._set_last_motion_joint_state(candidate)
                    return candidate
                if decision == "retry":
                    if candidate_idx + 1 < num_candidates:
                        self._log("Showing next generated motion candidate...")
                    else:
                        self._log(
                            "Generated motion candidates exhausted; "
                            "searching for another motion plan..."
                        )
                        self._viz_overlay_entries = []
                        self._on_display_geometries(
                            self._base_scene(self.q_home) + [self.origin_frame],
                            f"Step {n_step + 1}: replanning...",
                        )
                    continue
                if decision == "abort":
                    return "abort"

    def _num_regrasp_candidates(self, result):
        return self.num_regrasp_candidates(result)

    def _select_regrasp_candidate(self, result, candidate_idx):
        return self.select_regrasp_candidate(result, candidate_idx)

    def _regrasp_candidate_score(self, result, candidate_idx):
        return self.regrasp_candidate_score(result, candidate_idx)

    def _set_last_motion_joint_state(self, result) -> None:
        for paths_attr in ("q_path_sequence", "q_path_sequences"):
            paths = getattr(result, paths_attr, None)
            if not paths:
                continue
            if paths_attr == "q_path_sequences":
                paths = paths[0] if paths else []
            for path in reversed(paths):
                if not path:
                    continue
                q = np.asarray(path[-1], dtype=np.float64)
                if q.ndim == 1 and q.shape[0] >= 6 and np.all(np.isfinite(q[:6])):
                    self.q_joint = q[:6].copy()
                    return
        self.q_joint = np.asarray(self.q_home, dtype=np.float64).copy()

    def _show_planning_preview(
        self, n_step, target_id, pick_config, place_config, result
    ):
        q_path, target_path = self.generate_path_with_opening_angle(
            result, self.n_opening_angle
        )

        if VISUALIZATION_ON:
            save_path = os.path.join(self.video_dir, f"step_{n_step + 1}.mp4")
            scene_meshes_for_viz = {
                k: v for k, v in self.scene_meshes.items() if k != target_id
            }
            self.trajectory_visualization_with_target(
                q_path,
                target_path,
                self.excavator_model,
                self.excavator_meshes,
                scene_meshes_for_viz,
                self._copy_base_stone_mesh(target_id),
                save_path,
                self.camera_center,
                self.camera_position,
            )

        self._log(
            "Planning preview path samples: "
            f"q={len(q_path)}, target={len(target_path)}, "
            f"sequence_length={len(result.q_path_sequence)}"
        )
        target_trajectory = self._target_trajectory_markers(
            target_path,
            C_REPLANNED_TRAJECTORY,
            max_markers=PLANNING_PREVIEW_MAX_MARKERS,
        )
        settled_release_entries = self._settled_release_overlay_entries(
            target_id,
            result,
            max_markers=PLANNING_PREVIEW_MAX_MARKERS,
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
        q_gripper = np.ones(2) * result.grasp_sequence[0].opening_angle
        gripper_meshes_t = self.update_urdf_mesh(
            self.gripper_model, self.gripper_meshes, q_gripper
        )
        for mesh in gripper_meshes_t.values():
            mesh.transform(result.grasp_sequence[0].pose.as_matrix())

        q_at_grasp = result.q_path_sequence[0][-1]
        opening_angle_pick = result.grasp_sequence[0].opening_angle

        self._viz_target_id = target_id
        self._viz_target_mode = "pick_pose"
        self._viz_pick_pose_T = pick_config.pose.as_matrix()
        self._viz_place_pose_T = place_config.pose.as_matrix()
        self._viz_opening_angle = opening_angle_pick
        self._viz_overlay_entries = (
            [(m, C_GRIPPER) for m in gripper_meshes_t.values()]
            + trajectory_lines
            + target_trajectory
            + settled_release_entries
        )

        viz = (
            self._base_scene(q_at_grasp, opening_angle=opening_angle_pick)
            + [self.origin_frame]
            + self._viz_overlay_entries
        )
        self._log(f"Planning preview display geometries: {len(viz)}")
        self._on_display_geometries(viz, f"Step {n_step + 1}: planning preview")

    def _commit_place_body(self, place_body_id, target_id, place_config):
        if place_body_id is not None:
            self.context.remove_body(place_body_id)
        place_body_id = self.context.add_place_body(place_config)
        self.place_body_ids_by_stone[int(target_id)] = place_body_id

        self._set_scene_stone_pose(target_id, place_config)
        return place_body_id

    def _prepare_inhand_viewer_pose(self, result, pick_config):
        self._clear_place_inhand_override()
        grip_T_plan = result.grasp_sequence[0].pose.as_matrix()
        pick_T = pick_config.pose.as_matrix()
        self._viz_inhand_T = np.linalg.inv(grip_T_plan) @ pick_T
        self._viz_inhand_T_init = self._viz_inhand_T.copy()
        return result.grasp_sequence[0].opening_angle

    def _run_pick_control(self, result, opening_angle_pick):
        def _phase_cb_pick(phase):
            if phase in ("move", "approach"):
                self._viz_target_mode = "pick_pose"
                self._viz_opening_angle = 0.0
            elif phase in ("grasping", "lift", "retreat", "done"):
                self._viz_target_mode = "in_hand"
                self._viz_opening_angle = opening_angle_pick

        if not ROS_CONTROL_ON:
            return

        self._set_status(phase="Pick control")
        self._viz_overlay_entries = []
        path1 = result.q_path_sequence[0][: self.n_move]
        path2 = result.q_path_sequence[0][self.n_move :]
        path3 = result.q_path_sequence[1][: self.n_grasp]
        path4 = result.q_path_sequence[1][self.n_grasp :]
        with self._pause_live_joint_polling():
            self.sequential_grasp_control(
                path1,
                path2,
                path3,
                path4,
                self.joint_node_pub,
                self.joint_node_sub,
                "pick",
                confirm_cb=self._confirm_pick_grasp,
                pre_grasp_confirm_cb=self._confirm_pick_approach,
                state_cb=self._live_joint_state_cb(),
                phase_cb=_phase_cb_pick,
                spin_lock=self._ros_spin_lock,
                move_error_tol=self.move_control_error_tol,
                move_convergence_time_limit=(
                    self.move_control_convergence_time_limit
                ),
            )

    def _review_place_control_before_execution(
        self,
        n_step,
        target_id,
        pick_config,
        place_config,
        result,
        place_body_id,
        opening_angle_pick,
    ):
        while True:
            decision = self._wait_for_decision(
                "place_control_review",
                {"step": n_step + 1, "stone_id": target_id},
            )
            if decision == "continue":
                return result, place_config, place_body_id
            if decision == "abort":
                return "abort"

            # Remove the target's own place body during the replan. It was
            # committed to the "place" scene before pick/perception, so leaving
            # it in makes the grasped motion plan collision-check the held stone
            # against its own placed copy at the place pose -- a full-stone
            # penetration no relaxation or sub-~30cm lift can clear. Restored at
            # the original pose on abandon; re-committed at the retry pose on
            # success.
            if place_body_id is not None:
                self.context.remove_body(place_body_id)
                self.place_body_ids_by_stone.pop(target_id, None)
                place_body_id = None

            retried = self._run_pre_place_motion_retry(
                n_step,
                target_id,
                pick_config,
                place_config,
                result,
                opening_angle_pick,
            )
            if retried == "abort":
                self._commit_place_body(None, target_id, place_config)
                return "abort"
            if retried == "retry":
                place_body_id = self._commit_place_body(
                    None, target_id, place_config
                )
                continue
            if retried is None:
                self._log("Using the original place trajectory.")
                place_body_id = self._commit_place_body(
                    None, target_id, place_config
                )
                return result, place_config, place_body_id

            retry_result, retry_place_config = retried
            place_body_id = self._commit_place_body(
                None,
                target_id,
                retry_place_config,
            )
            return retry_result, retry_place_config, place_body_id

    def _run_pre_place_motion_retry(
        self,
        n_step,
        target_id,
        pick_config,
        place_config,
        initial_result,
        opening_angle_pick,
    ):
        base_place_config = self._place_config_from_result_or_current(
            initial_result,
            place_config,
        )
        while True:
            retry_place_config = self._make_place_control_retry_config(
                base_place_config
            )
            inhand_T = self._pre_place_retry_inhand_pose(
                initial_result,
                pick_config,
            )
            if inhand_T is None:
                self._log(
                    "Pre-place motion retry skipped: no held-stone in-hand pose "
                    "is available."
                )
                return "retry"
            opening_angle = self._pre_place_retry_opening_angle(opening_angle_pick)
            q_start = self._current_desktop_joint_state()
            if q_start is None:
                q_start = np.asarray(getattr(self, "q_joint", self.q_home))
            q_start = np.asarray(q_start, dtype=np.float64).copy()[:6]
            self.q_joint = q_start.copy()
            self._log("Retrying held-stone place motion before place control.")
            self._log_inhand_replan_request(
                target_id,
                "direct",
                q_start,
                retry_place_config,
                opening_angle,
                enable_near_ik=True,
            )
            retry_result = self.solve_inhand_grasp_planning(
                self.context,
                q_start,
                self.diffsim.Pose().from_matrix(inhand_T),
                retry_place_config,
                opening_angle,
                self.n_move,
                self.n_grasp,
                True,
                regrasp_xy_pos=None,
                max_num_solutions=self.max_num_regrasp_solutions,
                inhand_replan_mode="direct",
            )
            if not self._planning_result_is_valid(retry_result):
                self._log(
                    "Pre-place motion retry failed; adjust the place Z offset "
                    "or execute the original trajectory."
                )
                return "retry"

            reviewed = self._review_pre_place_retry_result(
                n_step,
                target_id,
                retry_result,
                opening_angle,
                initial_result,
            )
            if reviewed == "retry":
                continue
            if reviewed == "abort":
                return "abort"
            if reviewed is None:
                return None
            return reviewed, retry_place_config

    def _place_config_from_result_or_current(self, result, place_config):
        current = copy.deepcopy(place_config)
        try:
            place_T = self._result_final_target_pose_matrix(result)
        except Exception:
            place_T = None
        if place_T is not None:
            pose = self._pose_array_from_matrix(place_T)
            current.pose.setPosition(pose[:3])
            current.pose.setOrientation(pose[3:7])
        return current

    def _make_place_control_retry_config(self, place_config):
        retry_config = copy.deepcopy(place_config)
        offset = float(self._place_retry_z_offset)
        if not np.isfinite(offset):
            offset = 0.0
        pos = np.array(retry_config.pose.position(), dtype=np.float64, copy=True)
        pos[2] += offset
        retry_config.pose.setPosition(pos)
        self._log(
            "Pre-place motion retry target: "
            f"place Z offset {offset:+.3f} m, "
            f"pos={np.array2string(pos, precision=6, suppress_small=True)}"
        )
        return retry_config

    def _pre_place_retry_inhand_pose(self, result, pick_config):
        inhand_T = self._place_inhand_override()
        if inhand_T is not None:
            return inhand_T
        inhand_T = getattr(self, "_viz_inhand_T", None)
        if inhand_T is not None:
            inhand_T = np.asarray(inhand_T, dtype=np.float64)
            if inhand_T.shape == (4, 4) and np.all(np.isfinite(inhand_T)):
                return inhand_T.copy()
        try:
            gripper_T = result.grasp_sequence[0].pose.as_matrix()
            pick_T = pick_config.pose.as_matrix()
            return np.linalg.inv(gripper_T) @ pick_T
        except Exception:
            return None

    def _pre_place_retry_opening_angle(self, opening_angle_pick):
        override = getattr(self, "_place_opening_angle_override", None)
        if override is not None and np.isfinite(float(override)):
            return float(override)
        return float(opening_angle_pick)

    def _review_pre_place_retry_result(
        self,
        n_step,
        target_id,
        retry_result,
        opening_angle,
        initial_result,
    ):
        num_candidates = self._num_regrasp_candidates(retry_result)
        for candidate_idx in range(num_candidates):
            candidate = self._select_regrasp_candidate(retry_result, candidate_idx)
            score = self._regrasp_candidate_score(retry_result, candidate_idx)
            self._log(
                f"Reviewing pre-place retry candidate {candidate_idx + 1}/"
                f"{num_candidates}"
                + (f" (score: {score:.3f})" if score is not None else "")
            )
            self._show_replanned_trajectory(
                n_step,
                target_id,
                candidate,
                opening_angle,
                initial_result,
            )
            decision = self._wait_for_decision(
                "place_retry_review",
                {
                    "step": n_step + 1,
                    "stone_id": target_id,
                    "candidate": candidate_idx + 1,
                    "num_candidates": num_candidates,
                },
            )
            if decision == "continue":
                self._log("Pre-place retried trajectory accepted.")
                return candidate
            if decision == "skip":
                return None
            if decision == "abort":
                return "abort"
            return "retry"
        return "retry"

    def _confirm_pick_approach(self, q_pregrasp, grab):
        return self._confirm_adjustable_grasp("pick_approach", q_pregrasp, grab)

    def _confirm_pick_grasp(self, q_pregrasp, grab):
        return self._confirm_adjustable_grasp("pick", q_pregrasp, grab)

    def _confirm_place_grasp(self, q_pregrasp, grab):
        self._pick_adjust_base_q = np.asarray(q_pregrasp, dtype=np.float64).copy()
        self._pick_adjust_q = self._pick_adjust_base_q.copy()
        self._pick_adjust_grab = grab
        self._pick_adjust_phase = "place"
        release_count = 0
        while True:
            decision = self._wait_for_decision("grasp_confirm", {"phase": "place"})
            adjusted_q = self._pick_adjust_q.copy()
            if decision == "abort":
                self._clear_place_release_prompt(adjusted_q)
                return "abort"
            if decision == "skip":
                self._clear_place_release_prompt(adjusted_q)
                self._log(
                    "Place release finished after "
                    f"{release_count} manual release step(s)."
                )
                return "skip_grasp_publish"
            release_count += 1
            # continue = small release step; retry = full opening in one press.
            publish_count = (
                MANUAL_PLACE_OPEN_PUBLISH_COUNT
                if decision == "retry"
                else PLACE_RELEASE_STEP_PUBLISH_COUNT
            )
            self._publish_grasp_release_step(
                adjusted_q, release_count, publish_count=publish_count
            )
            self._pick_adjust_grab = "open"

    def _confirm_place_approach(self, q_pregrasp, grab):
        return self._confirm_adjustable_grasp("place_approach", q_pregrasp, grab)

    def _publish_grasp_release_step(
        self, q_release, release_count, publish_count=PLACE_RELEASE_STEP_PUBLISH_COUNT
    ):
        q_release = np.asarray(q_release, dtype=np.float64).reshape(-1)[:6].copy()
        v = np.zeros_like(q_release)
        for _ in range(int(publish_count)):
            self.joint_node_pub.publish(q_release, v, "open")
        self.q_joint = q_release.copy()
        self._update_place_release_pose(q_release)
        self._viz_target_mode = "place_pose"
        self._viz_opening_angle = 0.0
        if LIVE_JOINT_VIEWER_ON:
            self._live_state_cb(q_release)
        self._clear_subgoal_pcd_overlay(q_release, "Place Release")
        self._log(
            "Published place release step "
            f"{release_count} ({int(publish_count)} open messages)."
        )

    def _update_place_release_pose(self, q_release):
        if self._viz_inhand_T is None:
            return
        release_T = self._target_pose_from_gripper_inhand(
            q_release,
            self._viz_inhand_T,
            self._viz_opening_angle,
        )
        if release_T is not None:
            self._viz_place_pose_T = release_T

    def _clear_place_release_prompt(self, q_release):
        self._update_place_release_pose(q_release)
        self._clear_subgoal_pcd_overlay(q_release, "Place Release")
        self._pick_adjust_base_q = None
        self._pick_adjust_q = None

    def _confirm_adjustable_grasp(self, phase, q_pregrasp, grab):
        self._pick_adjust_base_q = np.asarray(q_pregrasp, dtype=np.float64).copy()
        self._pick_adjust_q = self._pick_adjust_base_q.copy()
        self._pick_adjust_grab = grab
        self._pick_adjust_phase = phase
        while True:
            decision = self._wait_for_decision("grasp_confirm", {"phase": phase})
            if phase == "manual_release" and decision == "retry":
                # Small open step; "Open" (continue) still publishes the full
                # opening from the caller.
                self._publish_manual_joint(
                    self._pick_adjust_q.copy(),
                    "open",
                    PLACE_RELEASE_STEP_PUBLISH_COUNT,
                )
                self._pick_adjust_grab = "open"
                self._log(
                    "Published manual open step "
                    f"({int(PLACE_RELEASE_STEP_PUBLISH_COUNT)} open messages)."
                )
                continue
            break
        adjusted_q = self._pick_adjust_q.copy()
        self._pick_adjust_base_q = None
        self._pick_adjust_q = None
        if phase == "place" and self._viz_inhand_T is not None:
            release_T = self._target_pose_from_gripper_inhand(
                adjusted_q,
                self._viz_inhand_T,
                self._viz_opening_angle,
            )
            if release_T is not None:
                self._viz_place_pose_T = release_T
        self._clear_subgoal_pcd_overlay(
            adjusted_q,
            f"{phase.replace('_', ' ').title()}",
        )
        if decision == "abort":
            return "abort"
        return adjusted_q

    def _publish_pick_adjusted_target(self):
        if self._pick_adjust_q is None:
            return
        q = self._pick_adjust_q.copy()
        v = np.zeros_like(q)
        for _ in range(100):
            self.joint_node_pub.publish(q, v, self._pick_adjust_grab)
        if LIVE_JOINT_VIEWER_ON:
            self._live_state_cb(q)
        offsets = {
            "swing": np.rad2deg(q[0] - self._pick_adjust_base_q[0]),
            "boom": np.rad2deg(q[1] - self._pick_adjust_base_q[1]),
            "arm": np.rad2deg(q[2] - self._pick_adjust_base_q[2]),
            "bucket": np.rad2deg(q[3] - self._pick_adjust_base_q[3]),
            "rotate": np.rad2deg(q[5] - self._pick_adjust_base_q[5]),
            "tilt": np.rad2deg(q[4] - self._pick_adjust_base_q[4]),
        }
        phase_name = self._pick_adjust_phase.replace("_", " ").title()
        self._log(
            f"{phase_name} grasp offset: "
            + ", ".join(f"{name} {value:.1f} deg" for name, value in offsets.items())
        )
