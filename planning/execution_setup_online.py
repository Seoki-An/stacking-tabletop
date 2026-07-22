"""Runtime initialization, scene setup, test mode, and online planning checkpoints."""

from agent.config_views import support_config

from .execution_common import *


class ExecutionSetupOnlineMixin:
    def _initialize_ros_and_models(self):
        cfg_path = os.path.join(self.plan_dir, "config.yml")
        if not os.path.exists(cfg_path):
            cfg_path = "agent/configs/config.yml"
        self._log(f"Execution config: {cfg_path}")
        cfg = self.OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.asset_dir = cfg.environment.data.load_dir
        old_env_ground_height = self.environment_ground_height(cfg.environment)
        self.set_environment_ground_height(cfg.environment, self.place_plane_height)
        if abs(old_env_ground_height - self.place_plane_height) > 1e-9:
            self._log(
                "Updating execution environment support ground_z: "
                f"{old_env_ground_height:.4f} -> {self.place_plane_height:.4f}"
            )

        self.rclpy.init()
        self.integrated_planner = self.IntegratedPlanner(cfg)
        self.joint_node_pub = self.CtrlJointPublisher()
        self.joint_node_sub = self.CtrlJointSubscriber()
        self.phase_node_pub = self.PhasePublisher()
        self.log_dir_pub = self.LogDirPublisher()
        self.scene_pose_init_pub = self.PoseArrayPublisher(
            name="scene_poseid_init_publisher_execution",
            topic="/scene_poseid_init",
        )
        self.scene_scan_done_sub = self.SceneScanDoneSubscriber(
            name="scene_scan_done_subscriber_execution",
            topic="/scene_scan_done",
        )
        self.diagnostic_pcd_request_pub = self.DiagnosticPcdRequestPublisher(
            name="diagnostic_pcd_request_publisher_execution",
            topic="/diagnostic_pcd_request",
        )
        self.diagnostic_pcd_done_sub = self.SceneScanDoneSubscriber(
            name="diagnostic_pcd_done_subscriber_execution",
            topic="/diagnostic_pcd_done",
        )
        self.scene_pcd_sub = self.SceneIdentifierSubscriber(
            name="scene_pcd_subscriber_execution",
            topic="/scene_pcd",
        )
        self.place_body_ids_by_stone: dict[int, int] = {}

        date_str = datetime.datetime.now().strftime("%y%m%d")
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        plan_log_dir = os.path.join("logs", date_str, f"plan_{self.plan_id}")
        os.makedirs(plan_log_dir, exist_ok=True)
        self.base_log_dir = unique_suffixed_dir(
            plan_log_dir, prefix=f"exec_{timestamp}", suffix=DESKTOP_LOG_SUFFIX
        )
        os.makedirs(self.base_log_dir, exist_ok=True)
        self._log(f"Logging directory: {self.base_log_dir}")

        self.context, _ = self.get_planner(
            self.pick_plane_height,
            self.place_plane_height,
            n_threads=self.resolve_thread_count(20, cfg),
        )
        self.posegen = self._create_posegen()
        self.excavator_model, self.excavator_meshes = self.get_excavator_model()
        (
            self.stone_dsf_meshes,
            self.stone_configs,
            self.stone_pcds,
            self.stone_meshes,
        ) = self.get_stone_model(self.asset_dir)
        self.gripper_model, self.gripper_meshes = self.get_gripper_model()

    def _initialize_online_debug_state(self):
        seq = self.sequence_runtime
        debug_path = os.path.join(self.plan_dir, "state.pkl")
        self._online_debug_data = seq._init_debug_data(self.integrated_planner.env)
        seq._refresh_debug_target_wall(
            self._online_debug_data,
            self.integrated_planner.env,
            reason="online session",
        )
        seq._backfill_debug_stone_meshes(
            self._online_debug_data,
            self.stone_dsf_meshes,
        )
        seq._set_debug_coordinate_metadata(
            self._online_debug_data,
            self.target_structure_offset,
        )
        seq._save_debug_data(self._online_debug_data, debug_path)
        self._log(f"Online planning state log: {debug_path}")

    def _create_posegen(self):
        posegen = self.get_posegen(ground_height=self.pick_plane_height)
        posegen.config().obj.set_k_wrench(np.zeros([6, 6]))
        posegen.config().obj.k_comp = 0.0
        posegen.config().tr.max_iter = 100
        posegen.config().tr.tol_eps = 1e-12
        return posegen

    def _initialize_scene(self):
        self.pick_ids = {}
        self.scene_meshes = {}
        self.scene_configs = {}
        skipped_missing_pose = []
        for sid in self.stone_dsf_meshes.keys():
            pose = self._initial_pose_for_stone(sid)
            if pose is None:
                skipped_missing_pose.append(int(sid))
                continue
            pos, quat = pose[:3], pose[3:7]
            self.stone_configs[sid].pose.setPosition(pos)
            self.stone_configs[sid].pose.setOrientation(quat)
            pose_T = self.stone_configs[sid].pose.as_matrix()
            mesh = self._copy_base_stone_mesh(sid)
            mesh.transform(pose_T)
            self.scene_meshes[sid] = mesh
            self.scene_configs[sid] = self.stone_configs[sid]
            self.pick_ids[sid] = self.context.add_pick_body(self.stone_configs[sid])

        if skipped_missing_pose:
            self._log(
                "Warning: ignored stones without field pose or resumed scene pose: "
                f"{skipped_missing_pose}"
            )

        self.target_structure_configs = []
        for action_i, action in enumerate(self.action_sequence):
            sid = int(action["stone_id"])
            place_pose = self._place_pose_for_action(action, action_i)
            tmp_config = copy.deepcopy(self.stone_configs[sid])
            tmp_config.pose.setPosition(place_pose[:3])
            tmp_config.pose.setOrientation(place_pose[3:])
            self.target_structure_configs.append((sid, tmp_config))

        self.origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self.lidar_frame_geometry = self._make_lidar_frame_geometry(size=0.35)
        self.ground_mesh = o3d.geometry.TriangleMesh.create_box(30.0, 30.0, 0.001)
        self.ground_mesh.translate((-15, -15, self.pick_plane_height - 0.001))
        self.place_height_box_mesh = self._make_place_height_box_mesh()

    def _initial_pose_for_stone(self, sid: int) -> np.ndarray | None:
        sid = int(sid)
        pose = self.poses.get(sid)
        if pose is None:
            pose = self.resume_scene_poses.get(sid)
        if pose is None:
            return None
        return self._pose_array(pose, stone_id=sid)

    def _show_initial_scene(self):
        self._on_display_geometries(
            self._base_scene(self.q_home) + [self.origin_frame],
            "Initial scene",
        )

    def _home_robot(self):
        if not ROS_CONTROL_ON:
            return

        self._set_status(phase="Homing")
        with self._pause_live_joint_polling():
            self.position_control(
                self.q_home,
                np.zeros_like(self.q_home),
                self.joint_node_pub,
                self.joint_node_sub,
                error_tol=0.1,
                time_limit=10.0,
                state_cb=self._live_joint_state_cb(),
                spin_lock=self._ros_spin_lock,
            )

    def _execute_action_sequence(self):
        total = len(self.action_sequence)
        for n_step, action in enumerate(self.action_sequence):
            if self._abort.is_set():
                self._log("Aborted by user.")
                return
            if not self._execute_step(n_step, action, total):
                return

    def _execute_online_action_sequence(self):
        self._restore_online_seed_prefix_scene()
        total = int(self.integrated_planner.env.cfg.n_stone)
        while len(self.action_sequence) < total:
            if self._abort.is_set():
                self._log("Aborted by user.")
                return
            n_step = len(self.action_sequence)
            action = self._plan_online_step(n_step, total)
            if action is None:
                return
            if not self._execute_step(n_step, action, total):
                return
            if self._online_last_node_done:
                return

    def _restore_online_seed_prefix_scene(self) -> None:
        if self._online_seed_prefix_restored:
            return
        self._online_seed_prefix_restored = True
        prefix_actions = list(getattr(self, "action_sequence", []) or [])
        if not prefix_actions:
            return

        self._log(
            "Restoring seeded online prefix before execution: "
            f"{len(prefix_actions)} step(s)."
        )
        state = self.integrated_planner.env.get_state()
        state = self._prepare_online_seed_prefix_state(state, prefix_actions)
        seed_prefix_pose_by_id = self._online_seed_pose_by_id_for_prefix(
            len(prefix_actions)
        )
        if seed_prefix_pose_by_id:
            self._log(
                "Using seed state poses for already stacked online prefix stones: "
                f"{sorted(seed_prefix_pose_by_id)}"
            )
        restored_ids = []
        for action_i, action in enumerate(prefix_actions):
            try:
                target_id = int(action["stone_id"])
            except (TypeError, ValueError, KeyError):
                self._log(
                    f"Skipping invalid seeded prefix action {action_i + 1}: "
                    f"{action!r}"
                )
                continue
            if target_id not in self.stone_configs:
                self._log(
                    f"Skipping seeded prefix stone {target_id}: "
                    "no loaded stone config."
                )
                continue

            local_pose = seed_prefix_pose_by_id.get(target_id)
            if local_pose is None:
                local_pose = self._local_pose_from_action(action, target_id)
            else:
                local_pose = local_pose.copy()
            place_pose = local_pose.copy()
            place_pose[:2] += self.target_structure_offset[:2]
            place_config = copy.deepcopy(self.stone_configs[target_id])
            place_config.pose.setPosition(place_pose[:3])
            place_config.pose.setOrientation(place_pose[3:7])

            pick_id = self.pick_ids.pop(target_id, None)
            if pick_id is not None:
                self.context.remove_body(pick_id)
            old_place_id = self.place_body_ids_by_stone.get(target_id)
            self._commit_place_body(old_place_id, target_id, place_config)

            action_for_restore = copy.deepcopy(action)
            action_for_restore["pose"] = local_pose.copy()
            seed_action = self._make_online_seed_action(
                state,
                action_for_restore,
                solve=False,
            )
            if seed_action is None:
                continue
            state, _obs, _done, _reward, _info = (
                self._advance_online_state_with_seed_action(state, seed_action)
            )
            self.integrated_planner.env.update_from_state(state)
            self.resume_scene_poses[target_id] = local_pose.copy()
            restored_ids.append(target_id)

        if self._online_debug_data is not None:
            self._online_debug_data["resume_state"] = copy.deepcopy(state)
            self._online_debug_data["resume_step"] = len(prefix_actions)
            self.sequence_runtime._save_debug_data(
                self._online_debug_data,
                os.path.join(self.plan_dir, "state.pkl"),
            )
        self.planning_params["resume_scene_poses"] = {
            int(stone_id): np.asarray(pose, dtype=np.float64).copy()
            for stone_id, pose in state.stone_poses.items()
            if int(stone_id) in set(restored_ids)
        }
        with open(os.path.join(self.plan_dir, "planning_params.pkl"), "wb") as f:
            pickle.dump(self.planning_params, f)
        self._log(f"Seeded prefix restored stone ids: {restored_ids}")

    def _prepare_online_seed_prefix_state(self, state, prefix_actions):
        prefix_ids = []
        for action in prefix_actions:
            if not isinstance(action, dict) or "stone_id" not in action:
                continue
            try:
                stone_id = int(action["stone_id"])
            except (TypeError, ValueError):
                continue
            if stone_id not in prefix_ids:
                prefix_ids.append(stone_id)

        seed_stone_set = self._online_seed_stone_set_for_prefix(len(prefix_actions))
        current_stone_set = [
            int(stone_id)
            for stone_id in np.asarray(getattr(state, "stone_set", [])).reshape(-1)
        ]
        ordered_stone_set = []
        for stone_id in [*prefix_ids, *seed_stone_set, *current_stone_set]:
            if stone_id not in ordered_stone_set:
                ordered_stone_set.append(int(stone_id))

        missing_prefix = [
            stone_id for stone_id in prefix_ids if stone_id not in current_stone_set
        ]
        if missing_prefix:
            self._log(
                "Online seed prefix extends planner stone_set with "
                f"{len(missing_prefix)} already placed stone id(s): {missing_prefix}"
            )

        state = copy.deepcopy(state)
        state.stone_set = np.asarray(ordered_stone_set, dtype=int)
        state.stone_seq = []
        state.stone_poses = {}
        state.action_history = []
        state.terminated = False
        state.failed = False
        if hasattr(state, "contact_points"):
            state.contact_points = []
        if hasattr(state, "pose_identified_stone_ids"):
            state.pose_identified_stone_ids = set()
        self.integrated_planner.env.update_from_state(state)
        return state

    def _online_seed_state_for_prefix(self, prefix_len: int):
        data = getattr(self, "_online_seed_debug_data", {}) or {}
        candidates = []
        if isinstance(data, dict):
            for step in data.get("steps", []) or []:
                if not isinstance(step, dict):
                    continue
                for key in ("raw_state", "resume_state"):
                    state = step.get(key)
                    if state is not None:
                        candidates.append(state)
            state = data.get("resume_state")
            if state is not None:
                candidates.append(state)

        best_state = None
        best_delta = None
        for candidate in candidates:
            stone_set = getattr(candidate, "stone_set", None)
            stone_seq = getattr(candidate, "stone_seq", None)
            if stone_set is None:
                continue
            seq_len = len(stone_seq) if stone_seq is not None else prefix_len
            delta = abs(int(seq_len) - int(prefix_len))
            if best_delta is None or delta < best_delta:
                best_state = candidate
                best_delta = delta
                if delta == 0:
                    break
        return best_state

    def _online_seed_stone_set_for_prefix(self, prefix_len: int) -> list[int]:
        best_state = self._online_seed_state_for_prefix(prefix_len)
        if best_state is None:
            return []
        return [
            int(stone_id)
            for stone_id in np.asarray(best_state.stone_set).reshape(-1)
        ]

    def _online_seed_pose_by_id_for_prefix(self, prefix_len: int) -> dict[int, np.ndarray]:
        best_state = self._online_seed_state_for_prefix(prefix_len)
        if best_state is None:
            return {}
        stone_poses = getattr(best_state, "stone_poses", {}) or {}
        out = {}
        for stone_id, pose in stone_poses.items():
            try:
                pose = np.asarray(pose, dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
                continue
            out[int(stone_id)] = pose[:7].copy()
        return out

    def _execute_test_mode(self):
        if self.online_mode:
            raise ValueError("Execution test modes are only supported for saved plans.")
        if self.test_mode == "scene_scan":
            self._execute_scene_scan_test()
            return
        raise ValueError(f"Unknown execution test mode: {self.test_mode!r}")

    def _execute_scene_scan_test(self):
        if not self.action_sequence:
            raise ValueError("Cannot run scene_scan test mode without actions.")
        n_step = self._scene_scan_test_step_index()
        total = len(self.action_sequence)
        target_id = int(self.action_sequence[n_step]["stone_id"])

        self._set_status(
            step=n_step + 1,
            total=total,
            stone_id=target_id,
            phase="Scene scan test",
        )
        self._log("=" * 40)
        self._log(
            f"Scene scan test mode: step {n_step + 1}/{total} " f"(stone {target_id})"
        )
        self._log("=" * 40)
        self._log(
            "Initialized q_home first; skipping motion planning, pick control, "
            "in-hand scan, and place control."
        )

        placed_ids = self._reconstruct_scene_prefix_for_scene_scan_test(n_step)
        self._log(
            "Scene scan test reconstructed placed prefix stone ids: " f"{placed_ids}"
        )
        self._on_display_geometries(
            self._base_scene(self.Q_SCAN) + [self.origin_frame],
            f"Step {n_step + 1}: scene scan test setup",
        )

        if not PERCEPTION_ON:
            self._log("Scene scan test skipped because PERCEPTION_ON is False.")
            return

        with self._suppress_live_layout_updates():
            self._reset_viewer_state()
            self._run_scene_scan(n_step, allow_planned_scene_prior=True)

    def _scene_scan_test_step_index(self) -> int:
        total = len(self.action_sequence)
        step = int(self.start_step)
        if step > total:
            self._log(
                f"Scene scan test step {step} is after the last action "
                f"({total}); using step {total}."
            )
            step = total
        return max(1, step) - 1

    def _reconstruct_scene_prefix_for_scene_scan_test(self, n_step: int) -> list[int]:
        placed_ids = []
        for action_i, action in enumerate(self.action_sequence[: n_step + 1]):
            try:
                target_id = int(action["stone_id"])
            except (TypeError, ValueError, KeyError):
                self._log(f"Skipping invalid action at index {action_i}: {action!r}")
                continue
            if target_id not in self.stone_configs:
                self._log(
                    f"Skipping stone {target_id} in scene scan test prefix: "
                    "no loaded stone config."
                )
                continue

            place_pose = self._place_pose_for_action(action, action_i)
            place_config = copy.deepcopy(self.stone_configs[target_id])
            place_config.pose.setPosition(place_pose[:3])
            place_config.pose.setOrientation(place_pose[3:7])

            pick_id = self.pick_ids.pop(target_id, None)
            if pick_id is not None:
                self.context.remove_body(pick_id)
            old_place_id = self.place_body_ids_by_stone.get(target_id)
            self._commit_place_body(old_place_id, target_id, place_config)

            pose = self._pose_array_from_matrix(place_config.pose.as_matrix())
            self.resume_scene_poses[target_id] = pose.copy()
            self._sceneid_prior_poses[target_id] = pose.copy()
            placed_ids.append(target_id)

        return placed_ids

    def _plan_online_step(self, n_step: int, total: int):
        seq = self.sequence_runtime
        plan_attempt = 0
        execution_rejected_actions = []
        execution_rejected_stone_ids = set()
        motion_fail_counts_by_stone = {}
        self._online_last_node_done = False
        seed_actions_for_mcts = []
        if bool((self.online_options or {}).get("seed_plan_review", False)):
            seed_decision = self._review_online_seed_step(n_step, total)
            if seed_decision == "abort":
                return None
            if isinstance(seed_decision, dict):
                return seed_decision
            if seed_decision is not None:
                seed_actions_for_mcts = [seed_decision]

        while not self._abort.is_set():
            plan_attempt += 1
            self._set_status(
                step=n_step + 1,
                total=total,
                stone_id="-",
                phase="Online MCTS planning",
            )
            self._log("=" * 40)
            self._log(f"Online MCTS step {n_step + 1}/{total}")
            self._log("=" * 40)

            state = self.integrated_planner.env.get_state()
            actions, nodes, debug_nodes = self.integrated_planner.plan_one_step(
                state,
                use_policy=False,
                use_feasibility_score=True,
                execution_rejected_actions=execution_rejected_actions,
                execution_rejected_stone_ids=sorted(execution_rejected_stone_ids),
                seed_actions=seed_actions_for_mcts,
            )
            seed_actions_for_mcts = []
            if actions is None:
                failure_reason = seq._no_candidate_failure_reason(
                    state,
                    self.integrated_planner.env,
                    execution_rejected_stone_ids,
                )
                self._log(f"No MCTS action candidates returned: {failure_reason}")
                self._append_online_debug_step(
                    n_step + 1,
                    False,
                    state,
                    debug_nodes or [],
                    attempt=plan_attempt,
                    failure_reason=failure_reason,
                    rejected_stone_ids=sorted(execution_rejected_stone_ids),
                )
                return None

            for action_idx, (action, node) in enumerate(zip(actions, nodes), start=1):
                target_id = int(action["stone_id"])
                candidate_pose = copy.deepcopy(node.state.stone_poses[target_id])
                reject_stacked, ground_fill = seq._candidate_stacks_before_ground_fill(
                    state,
                    target_id,
                    candidate_pose,
                    self.stone_dsf_meshes,
                    self.cfg,
                )
                if reject_stacked:
                    self._reject_online_candidate_before_motion(
                        node,
                        execution_rejected_actions,
                        "stacked_before_ground_fill",
                        extra_info={
                            "ground_fill_occupancy": float(ground_fill["occupancy"]),
                            "ground_fill_required": float(ground_fill["required"]),
                        },
                    )
                    continue

                pick_config = copy.deepcopy(self.stone_configs[target_id])
                place_config = copy.deepcopy(pick_config)
                place_pose = candidate_pose.copy()
                place_pose[:2] += self.target_structure_offset[:2]
                place_config.pose.setPosition(place_pose[:3])
                place_config.pose.setOrientation(place_pose[3:])

                if node.info is None:
                    node.info = {}
                node.info.update(
                    seq._long_sim_motion_scene_check(
                        state,
                        node.state,
                        target_id,
                        place_config,
                        self.scene_configs,
                        self.target_structure_offset,
                    )
                )
                settled_local_pose, resettle_info = (
                    seq._resettle_place_config_fixed_scene(
                        self.cfg,
                        state,
                        target_id,
                        place_config,
                        self.scene_configs,
                        self.target_structure_offset,
                    )
                )
                node.info.update(resettle_info)
                if settled_local_pose is None:
                    reject_reason = str(
                        resettle_info.get(
                            "fixed_scene_resettle_reject_reason",
                            "invalid",
                        )
                    )
                    self._reject_online_candidate_before_motion(
                        node,
                        execution_rejected_actions,
                        f"fixed_scene_resettle_{reject_reason}",
                    )
                    continue

                reject_active_floor, active_floor_status = (
                    self._online_candidate_rejects_active_floor(
                        state,
                        target_id,
                        settled_local_pose,
                    )
                )
                node.info.update(active_floor_status)
                if reject_active_floor:
                    reject_reason = str(
                        active_floor_status.get(
                            "active_floor_reject_reason",
                            "invalid",
                        )
                    )
                    self._reject_online_candidate_before_motion(
                        node,
                        execution_rejected_actions,
                        f"active_floor_{reject_reason}",
                        extra_info=active_floor_status,
                    )
                    continue

                candidate_pose = settled_local_pose.copy()
                for stone_id, pose in getattr(state, "stone_poses", {}).items():
                    node.state.stone_poses[int(stone_id)] = np.asarray(
                        pose,
                        dtype=float,
                    ).copy()
                node.state.stone_poses[target_id] = candidate_pose.copy()
                node.info["fixed_scene_commit_preserves_parent_scene"] = True
                if getattr(node, "action", None) is not None:
                    node.action.pose = candidate_pose.copy()

                place_pose = candidate_pose.copy()
                place_pose[:2] += self.target_structure_offset[:2]
                if abs(float(self.place_z_offset)) > 1e-12:
                    place_pose[2] += float(self.place_z_offset)
                    node.info["place_z_offset"] = float(self.place_z_offset)
                    node.info["place_motion_target_position"] = place_pose[:3].copy()
                    self._log(
                        "Online re-settled target stone position with "
                        f"place_z_offset={self.place_z_offset:.4f}: "
                        f"{place_pose[:3]}"
                    )
                else:
                    self._log(
                        "Online re-settled target stone position: "
                        f"{place_pose[:3]}"
                    )
                place_config.pose.setPosition(place_pose[:3])
                place_config.pose.setOrientation(place_pose[3:7])

                gap_info, gap_reject_reason = self._online_place_gap_check(
                    target_id,
                    place_config,
                )
                node.info.update(gap_info)
                if gap_reject_reason is not None:
                    self._reject_online_candidate_before_motion(
                        node,
                        execution_rejected_actions,
                        gap_reject_reason,
                        extra_info=gap_info,
                    )
                    continue

                self._log(
                    "Online candidate "
                    f"{action_idx}: stone {target_id}, "
                    f"target position {place_config.pose.position()}"
                )
                result = self._plan_online_candidate_motion(
                    target_id,
                    pick_config,
                    place_config,
                )
                place_pick_transform = place_config.pose.as_matrix() @ np.linalg.inv(
                    pick_config.pose.as_matrix()
                )
                node._failed_grasps = seq._serializable_failed_grasps(
                    result,
                    place_pick_transform,
                )
                failure_stage = self.motion_failure_stage(result)
                if failure_stage == "interrupted":
                    raise KeyboardInterrupt
                self._log(
                    f"Motion planning result: {self.motion_result_summary(result)}"
                )
                if len(result.q_path_sequence) == 0 or not result.is_feasible:
                    self._log(
                        "No feasible motion for this MCTS candidate; "
                        "trying another candidate."
                    )
                    node._motion_failed = True
                    node.info["motion_planning_failed"] = True
                    node.info["motion_attempt"] = int(action_idx)
                    node.info["motion_failure_stage"] = failure_stage
                    node.info["motion_failure_detail"] = self.motion_failure_detail(
                        result
                    )
                    node.info["motion_regrasp_xy"] = self.regrasp_xy_pos.copy()
                    if getattr(node, "action", None) is not None:
                        execution_rejected_actions.append(node.action.copy())
                    motion_fail_counts_by_stone[target_id] = (
                        motion_fail_counts_by_stone.get(target_id, 0) + 1
                    )
                    if (
                        motion_fail_counts_by_stone[target_id]
                        >= seq.STONE_REJECTION_THRESHOLD
                    ):
                        execution_rejected_stone_ids.add(target_id)
                    continue

                return self._commit_online_planned_step(
                    n_step,
                    action,
                    node,
                    result,
                    state,
                    debug_nodes or nodes,
                    action_idx,
                    plan_attempt,
                )

            self._log(
                "All returned MCTS actions failed motion planning; retrying "
                f"MCTS with {len(execution_rejected_actions)} rejected actions "
                f"and {len(execution_rejected_stone_ids)} rejected stones."
            )
            self._append_online_debug_step(
                n_step + 1,
                False,
                state,
                debug_nodes or nodes,
                resume_state=state,
                attempt=plan_attempt,
                failure_reason="motion_planning_failed",
                rejected_stone_ids=sorted(execution_rejected_stone_ids),
            )

        return None

    def _review_online_seed_step(self, n_step: int, total: int):
        seed_actions = getattr(self, "_online_seed_action_sequence", []) or []
        if n_step >= len(seed_actions):
            return None
        seed_record = seed_actions[n_step]
        if not isinstance(seed_record, dict):
            self._log(f"Seed step {n_step + 1} is not an action record; replanning.")
            return None
        try:
            target_id = int(seed_record["stone_id"])
        except (TypeError, ValueError, KeyError):
            self._log(f"Seed step {n_step + 1} has no valid stone id; replanning.")
            return None
        if target_id not in self.stone_configs:
            self._log(
                f"Seed step {n_step + 1} uses stone {target_id}, "
                "but that stone is not loaded; replanning."
            )
            return None

        state = self.integrated_planner.env.get_state()
        seed_action = self._make_online_seed_action(state, seed_record, solve=False)
        if seed_action is None:
            self._log(f"Could not build seed action for step {n_step + 1}; replanning.")
            return None

        pick_config = copy.deepcopy(self.stone_configs[target_id])
        place_config = copy.deepcopy(pick_config)
        place_pose = self._world_place_pose_from_seed_action(seed_record, target_id)
        place_config.pose.setPosition(place_pose[:3])
        place_config.pose.setOrientation(place_pose[3:7])
        seed_result = self._saved_seed_motion_result_for_step(
            n_step,
            pick_config,
            place_config,
        )
        if seed_result is None:
            self._log(
                f"No replayable saved seed motion for step {n_step + 1}; "
                "running online MCTS."
            )
            return self._make_online_seed_action(state, seed_record, solve=True)

        self._set_status(
            step=n_step + 1,
            total=total,
            stone_id=target_id,
            phase="Planned seed review",
        )
        self._log("=" * 40)
        self._log(
            f"Planned seed step {n_step + 1}/{total}: stone {target_id}. "
            "Choose Use Planned to execute it, or Replan to run online MCTS "
            "seeded by this placement."
        )
        self._log("=" * 40)
        with self._pause_live_joint_polling():
            self._show_planning_preview(
                n_step,
                target_id,
                pick_config,
                place_config,
                seed_result,
            )
            decision = self._wait_for_decision(
                "online_seed_review",
                {"step": n_step + 1, "stone_id": target_id},
            )
        if decision == "continue":
            seed_node = self._make_online_seed_node(
                state,
                seed_action,
                simulate=False,
            )
            if seed_node is None:
                self._log(
                    f"Planned seed step {n_step + 1} could not be committed; "
                    "running online MCTS instead."
                )
                return self._make_online_seed_action(state, seed_record, solve=True)
            self._online_auto_accept_saved_motion_steps.add(n_step)
            self._log("Planned seed step accepted.")
            return self._commit_online_planned_step(
                n_step,
                seed_record,
                seed_node,
                seed_result,
                state,
                [seed_node],
                1,
                0,
            )
        if decision == "abort":
            return "abort"

        self._log("Planned seed step rejected; running online MCTS with it as seed.")
        return self._make_online_seed_action(state, seed_record, solve=True)

    def _saved_seed_motion_result_for_step(self, n_step, pick_config, place_config):
        motion_results = getattr(self, "_online_seed_motion_result_sequence", []) or []
        if n_step < len(motion_results) and isinstance(motion_results[n_step], dict):
            try:
                return self._make_saved_motion_result_from_metadata(
                    motion_results[n_step]
                )
            except Exception as exc:
                self._log(
                    f"Could not load seed motion result metadata for step "
                    f"{n_step + 1}: {exc}"
                )

        motions = getattr(self, "_online_seed_motion_sequence", []) or []
        if n_step >= len(motions):
            return None
        paths = self._coerce_saved_motion_paths(motions[n_step])
        if paths is None:
            return None
        try:
            return self._make_saved_motion_result(paths, pick_config, place_config)
        except Exception as exc:
            self._log(f"Could not replay seed motion for step {n_step + 1}: {exc}")
            return None

    def _make_online_seed_action(self, state, seed_record, solve: bool):
        try:
            target_id = int(seed_record["stone_id"])
            local_pose = self._local_pose_from_action(seed_record, target_id)
        except Exception as exc:
            self._log(f"Invalid online seed action: {exc}")
            return None
        stone_idx = self._stone_idx_for_id(state, target_id)
        if stone_idx is None:
            self._log(f"Seed stone {target_id} is not in the planner stone set.")
            return None
        if int(stone_idx) in self._state_stone_seq_indices(state):
            self._log(f"Seed stone {target_id} is already placed; ignoring seed.")
            return None

        if solve:
            try:
                self.integrated_planner.env.update_from_state(state)
                action = self.integrated_planner.env.get_action(
                    state,
                    local_pose.copy(),
                    int(stone_idx),
                )
            except Exception as exc:
                self._log(
                    f"Could not solve planned seed action for stone {target_id}: {exc}"
                )
                return None
            finally:
                try:
                    self.integrated_planner.env.update_from_state(state)
                except Exception:
                    pass
        else:
            action = self.Action(
                stone_idx=int(stone_idx),
                stone_id=int(target_id),
                pose=local_pose.copy(),
                init_pose=local_pose.copy(),
            )
        action.diagnostics["online_seed_plan"] = True
        return action

    def _make_online_seed_node(self, state, action, simulate: bool):
        try:
            if simulate:
                self.integrated_planner.env.update_from_state(state)
                next_state, obs, done, reward, info = self.integrated_planner.env.step(
                    action,
                    simulate=True,
                )
            else:
                next_state, obs, done, reward, info = (
                    self._advance_online_state_with_seed_action(state, action)
                )
        except Exception as exc:
            self._log(f"Could not simulate planned seed action: {exc}")
            try:
                self.integrated_planner.env.update_from_state(state)
            except Exception:
                pass
            return None
        node = self.MCTS_Node(self.cfg.algorithm.mcts, action=action.copy())
        info = dict(info or {})
        info.update(
            {
                "online_seed_plan": True,
                "online_seed_decision": "accepted",
                "online_seed_simulate": bool(simulate),
            }
        )
        node.update_state(
            next_state,
            obs,
            float(reward),
            bool(done),
            bool(getattr(next_state, "failed", False)),
            info=info,
        )
        node.set_is_simulated(True)
        node.visits = 1
        node.q_value = float(reward) if np.isfinite(float(reward)) else 0.0
        return node

    def _advance_online_state_with_seed_action(self, state, action):
        next_state = copy.deepcopy(state)
        stone_idx = int(action.stone_idx)
        stone_id = int(action.stone_id)
        stone_seq = list(np.asarray(next_state.stone_seq).reshape(-1))
        if stone_idx not in [int(idx) for idx in stone_seq]:
            next_state.stone_seq = stone_seq + [stone_idx]
        next_state.stone_poses = dict(next_state.stone_poses)
        next_state.stone_poses[stone_id] = np.asarray(
            action.pose,
            dtype=np.float64,
        ).copy()
        next_state.action_history = list(next_state.action_history) + [action.copy()]
        next_state.terminated = len(next_state.stone_seq) >= len(next_state.stone_set)
        next_state.failed = False
        self.integrated_planner.env.update_from_state(next_state)
        obs = self.integrated_planner.env.get_observation()
        reward, info = self.integrated_planner.env.reward_fn.compute(
            self.integrated_planner.env.inventory,
            next_state,
        )
        done = self.integrated_planner.env.simulator.is_done(next_state)
        return next_state, obs, done, reward, info

    def _stone_idx_for_id(self, state, target_id: int) -> int | None:
        stone_set = getattr(state, "stone_set", None)
        if stone_set is None:
            return None
        for idx, stone_id in enumerate(np.asarray(stone_set).reshape(-1)):
            if int(stone_id) == int(target_id):
                return int(idx)
        return None

    @staticmethod
    def _state_stone_seq_indices(state) -> set[int]:
        stone_seq = getattr(state, "stone_seq", None)
        if stone_seq is None:
            return set()
        return {int(idx) for idx in np.asarray(stone_seq).reshape(-1)}

    def _local_pose_from_action(self, action, target_id: int) -> np.ndarray:
        pose = np.asarray(action["pose"], dtype=np.float64).copy()
        if pose.ndim != 1 or pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
            raise ValueError(f"invalid pose for stone {target_id}: {action!r}")
        return pose[:7].copy()

    def _world_place_pose_from_seed_action(self, action, target_id: int) -> np.ndarray:
        pose = self._local_pose_from_action(action, target_id)
        pose[:2] += self.target_structure_offset[:2]
        return pose

    def _reject_online_candidate_before_motion(
        self,
        node,
        execution_rejected_actions: list,
        reason: str,
        extra_info: dict | None = None,
    ) -> None:
        node._motion_failed = True
        if node.info is None:
            node.info = {}
        node.info["execution_rejected"] = True
        node.info["execution_reject_reason"] = reason
        if extra_info:
            node.info.update(extra_info)
        if getattr(node, "action", None) is not None:
            execution_rejected_actions.append(node.action.copy())
        self._log(f"Rejecting MCTS candidate before motion planning: {reason}.")

    def _plan_online_candidate_motion(self, target_id, pick_config, place_config):
        context, pick_ids = self._make_online_motion_context()
        context.remove_body(pick_ids[int(target_id)])
        return self.regrasp_planning(
            context,
            pick_config,
            place_config,
            self.q_home,
            self.regrasp_xy_pos,
            self.n_move,
            self.n_grasp,
            self.online_motion_max_num_regrasp_solutions,
        )

    def _online_place_gap_check(self, target_id, place_config):
        threshold = -support_config(self.integrated_planner.env).contact_gap_tolerance

        target_points = self._stone_config_surface_points(target_id, place_config)
        if len(target_points) == 0:
            return {}, None

        plane_gap = float(np.min(target_points[:, 2] - float(self.place_plane_height)))
        self._log(
            "Online place gap check: "
            f"stone={int(target_id)}, plane_gap={plane_gap:.6f}, "
            f"threshold={threshold:.6f}"
        )

        info = {
            "online_place_plane_gap": plane_gap,
            "online_place_gap_threshold": threshold,
        }
        if plane_gap < threshold:
            return info, "place_plane_gap"
        return info, None

    def _online_candidate_rejects_active_floor(self, state, target_id, candidate_pose):
        floor_fill_cfg = self.integrated_planner.env.cfg.action.planar.get(
            "floor_fill",
            {},
        )
        if not lower_floor_fill_reject_stacked(floor_fill_cfg):
            return False, {}

        layer = active_floor_context(self.integrated_planner.env.inventory, state)
        if layer is None:
            return False, {}

        stone_set = np.asarray(
            self.integrated_planner.env.inventory.stone_set,
            dtype=int,
        ).reshape(-1)
        matches = np.flatnonzero(stone_set == int(target_id))
        if len(matches) == 0:
            return False, {}

        metrics = active_layer_fill_metrics(
            self.integrated_planner.env.inventory,
            layer,
            int(matches[0]),
            candidate_pose,
        )
        min_contact = max(
            int(floor_fill_cfg.get("u_shape_min_frontier_contact_cells", 1)),
            0,
        )
        status = {
            "fixed_scene_active_layer_fill_score": float(metrics["fill_score"]),
            "fixed_scene_active_layer_above": bool(metrics["above_active_layer"]),
            "fixed_scene_active_layer_contact_cells": int(metrics["contact_cells"]),
            "fixed_scene_active_layer_min_contact_cells": int(min_contact),
            "fixed_scene_active_layer_unfilled_cells": int(metrics["unfilled_cells"]),
            "fixed_scene_active_layer_overlap_cells": int(metrics["overlap_cells"]),
            "fixed_scene_active_layer_occupancy": float(layer["occupancy"]),
        }
        if bool(metrics["above_active_layer"]):
            status["active_floor_reject_reason"] = "above_active_layer"
            return True, status
        if (
            min_contact > 0
            and float(metrics["fill_score"]) > 0.0
            and int(metrics["contact_cells"]) < min_contact
        ):
            status["active_floor_reject_reason"] = "disconnected_active_layer"
            return True, status
        return False, status

    def _stone_config_surface_points(self, stone_id, config):
        mesh = copy.deepcopy(self.stone_dsf_meshes[int(stone_id)])
        mesh.transform(config.pose.as_matrix())
        return self._mesh_surface_points(mesh)

    def _mesh_surface_points(self, mesh):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] < 3 or len(vertices) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        vertices = vertices[:, :3]
        points = [vertices]
        triangles = np.asarray(mesh.triangles, dtype=np.int64)
        if triangles.ndim == 2 and triangles.shape[1] == 3 and len(triangles) > 0:
            valid = np.all((triangles >= 0) & (triangles < len(vertices)), axis=1)
            triangles = triangles[valid]
            if len(triangles) > 0:
                tri_vertices = vertices[triangles]
                points.append(tri_vertices.mean(axis=1))
                points.append(0.5 * (tri_vertices[:, 0] + tri_vertices[:, 1]))
                points.append(0.5 * (tri_vertices[:, 1] + tri_vertices[:, 2]))
                points.append(0.5 * (tri_vertices[:, 2] + tri_vertices[:, 0]))
        return np.concatenate(points, axis=0)

    def _make_online_motion_context(self):
        context, _ = self.get_planner(
            self.pick_plane_height,
            self.place_plane_height,
            n_threads=self.resolve_thread_count(20, self.cfg),
        )
        placed_ids = self._online_placed_stone_ids()
        pick_ids = {}
        for stone_id, config in self.stone_configs.items():
            sid = int(stone_id)
            if sid in placed_ids:
                scene_config = self.scene_configs.get(sid, config)
                context.add_place_body(copy.deepcopy(scene_config))
            else:
                pick_ids[sid] = context.add_pick_body(copy.deepcopy(config))
        return context, pick_ids

    def _online_placed_stone_ids(self) -> set[int]:
        return {int(action["stone_id"]) for action in self.action_sequence}

    def _commit_online_planned_step(
        self,
        n_step: int,
        action,
        node,
        result,
        state,
        debug_nodes,
        action_idx: int,
        plan_attempt: int,
    ):
        seq = self.sequence_runtime
        target_id = int(action["stone_id"])
        self.integrated_planner.env.update_from_state(node.state)
        action_record = {
            "stone_id": target_id,
            "pose": np.asarray(node.state.stone_poses[target_id], dtype=float).copy(),
            "place_z_offset": float(self.place_z_offset),
            "regrasp_xy_pos": self.regrasp_xy_pos.copy(),
            "selected_regrasp_xy_pos": self.regrasp_xy_pos.copy(),
            "motion_mode": "direct" if len(result.q_path_sequence) == 2 else "regrasp",
        }
        node_info = getattr(node, "info", None) or {}
        if node_info.get("online_seed_plan"):
            action_record["source"] = "planned_seed"
        self.action_sequence.append(action_record)
        path1, path2, path3, path4 = self.split_regrasp_place_paths(
            result,
            self.n_move,
            self.n_grasp,
            q_home=self.q_home,
        )
        self.motion_sequence.append([path1, path2, path3, path4])
        self.motion_result_sequence.append(seq._serializable_motion_result(result))
        self._append_online_target_structure(target_id, action_record["pose"])
        self._append_online_debug_step(
            n_step + 1,
            True,
            state,
            debug_nodes,
            resume_state=node.state,
            selected_node=node,
            selected_action_idx=action_idx,
            selected_regrasp_xy=self.regrasp_xy_pos,
            selected_motion_mode=action_record["motion_mode"],
            attempt=plan_attempt,
        )
        self._save_online_checkpoint()
        self._online_last_node_done = bool(getattr(node, "done", False))
        self._log(
            f"Saved online MCTS step {n_step + 1} to {self.plan_dir}; "
            "showing trajectory in the live viewer."
        )
        return action_record

    def _append_online_target_structure(self, target_id: int, local_pose):
        place_pose = np.asarray(local_pose, dtype=np.float64).copy()
        place_pose[:2] += self.target_structure_offset[:2]
        tmp_config = copy.deepcopy(self.stone_configs[int(target_id)])
        tmp_config.pose.setPosition(place_pose[:3])
        tmp_config.pose.setOrientation(place_pose[3:7])
        self.target_structure_configs.append((int(target_id), tmp_config))

    def _append_online_debug_step(self, step, succeeded, state, nodes, **kwargs):
        if self._online_debug_data is None:
            return
        self.sequence_runtime._append_debug_step(
            self._online_debug_data,
            step,
            succeeded,
            state,
            nodes,
            self.integrated_planner.env,
            **kwargs,
        )
        self.sequence_runtime._save_debug_data(
            self._online_debug_data,
            os.path.join(self.plan_dir, "state.pkl"),
        )

    def _save_online_checkpoint(self):
        if not getattr(self, "plan_dir", None):
            return
        with open(os.path.join(self.plan_dir, "action_sequence.pkl"), "wb") as f:
            pickle.dump(getattr(self, "action_sequence", []), f)
        with open(os.path.join(self.plan_dir, "motion_sequence.pkl"), "wb") as f:
            pickle.dump(getattr(self, "motion_sequence", []), f)
        with open(
            os.path.join(self.plan_dir, "motion_result_sequence.pkl"),
            "wb",
        ) as f:
            pickle.dump(getattr(self, "motion_result_sequence", []), f)
