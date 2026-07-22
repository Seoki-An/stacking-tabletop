"""Plan/session loading, resume-state reconciliation, and SceneID prior helpers."""

from .execution_common import *


class ExecutionPlanStateMixin:
    def _load_plan_files(self):
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cur_dir = os.path.join(repo_dir, "scripts", "desktop")
        self.plan_dir = self._resolve_plan_dir(repo_dir, self.plan_id)
        plan_name = os.path.basename(os.path.normpath(self.plan_dir))
        if plan_name.startswith("plan_"):
            self.plan_id = plan_name[len("plan_") :]
        else:
            self.plan_id = plan_name
        action_path = os.path.join(self.plan_dir, "action_sequence.pkl")
        motion_path = os.path.join(self.plan_dir, "motion_sequence.pkl")
        motion_result_path = os.path.join(self.plan_dir, "motion_result_sequence.pkl")
        planning_params_path = os.path.join(self.plan_dir, "planning_params.pkl")

        if VISUALIZATION_ON:
            self.video_dir = os.path.join(self.plan_dir, "videos")
            self.video_dir = self.get_unique_dir(self.video_dir, prefix="execution")
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)
        else:
            self.video_dir = None

        self.camera_center = [0, 0, 0]
        self.camera_position = [8.0, 5.0, 4.0]

        with open(action_path, "rb") as f:
            self.action_sequence = pickle.load(f)
        self.motion_sequence = []
        if os.path.exists(motion_path):
            with open(motion_path, "rb") as f:
                self.motion_sequence = pickle.load(f)
        self.motion_result_sequence = []
        if os.path.exists(motion_result_path):
            with open(motion_result_path, "rb") as f:
                self.motion_result_sequence = pickle.load(f)
        with open(planning_params_path, "rb") as f:
            self.planning_params = pickle.load(f)
        self.start_step = self._resolve_execution_start_step(
            total_actions=len(self.action_sequence)
        )
        self.resume_scene_pose_order = []
        self.resume_scene_poses = self._load_resume_scene_poses()

        self.q_home = self.planning_params["q_home"]
        self.q_joint = np.asarray(self.q_home, dtype=np.float64).copy()
        self.target_structure_offset = self.planning_params["target_structure_offset"]
        self.n_move = self.planning_params["n_move"]
        self.n_grasp = self.planning_params["n_grasp"]
        self.n_opening_angle = self.planning_params["n_opening_angle"]
        self.regrasp_xy_pos = self.planning_params["regrasp_xy_pos"]
        self.regrasp_xy_candidates = self.planning_params.get(
            "regrasp_xy_candidates",
            None,
        )
        if self.regrasp_xy_candidates is None:
            self.regrasp_xy_candidates = self.regrasp_position_candidates(
                self.regrasp_xy_pos
            )
        planned_max_num_regrasp_solutions = int(
            self.planning_params.get("max_num_regrasp_solutions", 3)
        )
        self.max_num_regrasp_solutions = EXECUTION_MAX_NUM_REGRASP_SOLUTIONS
        if planned_max_num_regrasp_solutions != self.max_num_regrasp_solutions:
            self._log(
                "Overriding max_num_regrasp_solutions for execution: "
                f"{planned_max_num_regrasp_solutions} -> "
                f"{self.max_num_regrasp_solutions}"
            )
        self.poses = self.planning_params["poses"]
        self.inhand_replan_translation_threshold = self.planning_params.get(
            "inhand_replan_translation_threshold",
            INHAND_REPLAN_TRANSLATION_THRESHOLD,
        )
        self.inhand_replan_rotation_threshold_deg = self.planning_params.get(
            "inhand_replan_rotation_threshold_deg",
            INHAND_REPLAN_ROTATION_THRESHOLD_DEG,
        )
        self.inhand_replan_place_z_offset = self.planning_params.get(
            "inhand_replan_place_z_offset",
            Z_OFFSET_PLACE,
        )
        self.place_z_offset = float(
            self.planning_params.get("place_z_offset", ONLINE_PLACE_Z_OFFSET)
        )
        self.inhand_replan_mode = (
            str(self.planning_params.get("inhand_replan_mode", "regrasp"))
            .strip()
            .lower()
        )
        if self.inhand_replan_mode not in {"direct", "regrasp"}:
            raise ValueError(
                "planning_params['inhand_replan_mode'] must be either "
                "'direct' or 'regrasp'"
            )
        self.pick_plane_height = float(
            self.planning_params.get(
                "pick_plane_height",
                self.planning_params.get(
                    "plane_pick_height",
                    self.planning_params.get("ground_height", 0.0),
                ),
            )
        )
        self.place_plane_height = float(
            self.planning_params.get(
                "place_plane_height",
                self.planning_params.get(
                    "plane_place_height",
                    self.planning_params.get("ground_height", self.pick_plane_height),
                ),
            )
        )
        self._reconcile_action_sequence_with_resume_scene_poses()

        stone_ids = [int(a["stone_id"]) for a in self.action_sequence]
        self._log(f"Plan directory: {self.plan_dir}")
        self._log(f"Stone ids in the action sequence: {stone_ids}")
        if self.motion_sequence:
            self._log(
                "Loaded generated motion sequence: "
                f"{len(self.motion_sequence)} step(s)."
            )
        if self.motion_result_sequence:
            self._log(
                "Loaded generated motion result metadata: "
                f"{len(self.motion_result_sequence)} step(s)."
            )
        self._log(f"Execution start step: {self.start_step}")
        if self.resume_scene_poses:
            self._log(
                "Loaded resumed scene poses for placed stone ids: "
                f"{sorted(self.resume_scene_poses)}"
            )
            self._sceneid_prior_poses = {
                int(stone_id): self._pose_as_world_scene_frame(
                    int(stone_id),
                    pose,
                )
                for stone_id, pose in self.resume_scene_poses.items()
            }
        self._log(f"q_home: {self.q_home.tolist()}")
        self._log(f"In-hand replan mode: {self.inhand_replan_mode}")
        if (
            abs(self.pick_plane_height) > 1e-9
            or abs(self.place_plane_height) > 1e-9
            or abs(self.pick_plane_height - self.place_plane_height) > 1e-9
        ):
            self._log(
                "Planner plane heights: "
                f"pick={self.pick_plane_height:.4f}, "
                f"place={self.place_plane_height:.4f}"
            )
        if LIVE_JOINT_VIEWER_ON:
            self._log(
                "Live joint 3D updates are enabled "
                f"({self._viewer_min_interval:.2f}s min interval)."
            )
        else:
            self._log(
                "Live joint 3D updates are disabled during execution. "
                "Run with --live_joint_viewer to enable them."
            )

    def _load_online_session_files(self):
        seq = self.sequence_runtime
        options = self.online_options or {}
        config_overrides = list(options.get("config_override") or [])
        seed_plan_dir = options.get("seed_plan_dir", None)

        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cur_dir = os.path.join(repo_dir, "scripts", "desktop")
        if seed_plan_dir is None:
            self.plan_dir = seq._new_plan_dir("sessions", suffix="online")
        else:
            self.plan_dir = os.path.abspath(
                os.path.normpath(os.path.expanduser(str(seed_plan_dir)))
            )
            if not os.path.isdir(self.plan_dir):
                raise FileNotFoundError(f"Seed plan directory not found: {self.plan_dir}")
        os.makedirs(self.plan_dir, exist_ok=True)
        plan_name = os.path.basename(self.plan_dir)
        if plan_name.startswith("plan_"):
            self.plan_id = plan_name[len("plan_") :]
        else:
            self.plan_id = plan_name

        seed_actions = []
        seed_motions = []
        seed_motion_results = []
        seed_planning_params = {}
        seed_debug_data = {}
        if seed_plan_dir is not None:
            seed_actions = self._load_pickle_or_default("action_sequence.pkl", [])
            seed_motions = self._load_pickle_or_default("motion_sequence.pkl", [])
            seed_motion_results = self._load_pickle_or_default(
                "motion_result_sequence.pkl",
                [],
            )
            seed_debug_data = self._load_pickle_or_default("state.pkl", {})
            seed_planning_params = self._load_pickle_or_default(
                "planning_params.pkl",
                {},
            )
            if not isinstance(seed_planning_params, dict):
                seed_planning_params = {}
            for source_name, seed_name in (
                ("action_sequence.pkl", "seed_action_sequence.pkl"),
                ("motion_sequence.pkl", "seed_motion_sequence.pkl"),
                ("motion_result_sequence.pkl", "seed_motion_result_sequence.pkl"),
                ("planning_params.pkl", "seed_planning_params.pkl"),
                ("state.pkl", "seed_state.pkl"),
            ):
                self._preserve_seed_plan_file(source_name, seed_name)

        self.camera_center = [0, 0, 0]
        self.camera_position = [8.0, 5.0, 4.0]
        self.video_dir = None

        def _param(name, default=None):
            return seed_planning_params.get(name, default)

        self.q_home = np.asarray(_param("q_home", seq.Q_HOME), dtype=np.float64).copy()
        self.target_structure_offset = np.asarray(
            options.get("target_structure_offset")
            or _param("target_structure_offset", seq.TARGET_STRUCTURE_OFFSET),
            dtype=np.float64,
        ).reshape(2)
        self.n_move = int(_param("n_move", seq.N_MOVE))
        self.n_grasp = int(_param("n_grasp", seq.N_GRASP))
        self.n_opening_angle = int(_param("n_opening_angle", seq.N_OPENING_ANGLE))
        self.regrasp_xy_pos = np.asarray(
            options.get("regrasp_xy_pos")
            or _param("regrasp_xy_pos", seq.REGRASP_XY_POS),
            dtype=np.float64,
        ).reshape(2)
        seed_regrasp_candidates = None
        if options.get("regrasp_xy_pos") is None:
            seed_regrasp_candidates = _param("regrasp_xy_candidates", None)
        if seed_regrasp_candidates is None:
            self.regrasp_xy_candidates = self.regrasp_position_candidates(
                self.regrasp_xy_pos
            )
        else:
            self.regrasp_xy_candidates = [
                np.asarray(candidate, dtype=np.float64).reshape(2)
                for candidate in seed_regrasp_candidates
            ]
        self.online_motion_max_num_regrasp_solutions = 1
        self.max_num_regrasp_solutions = EXECUTION_MAX_NUM_REGRASP_SOLUTIONS
        self.pick_plane_height = float(_param("pick_plane_height", seq.PICK_PLANE_HEIGHT))
        self.place_plane_height = float(
            _param(
                "place_plane_height",
                _param("plane_place_height", _param("ground_height", seq.PLACE_PLANE_HEIGHT)),
            )
        )
        self.inhand_replan_translation_threshold = _param(
            "inhand_replan_translation_threshold",
            INHAND_REPLAN_TRANSLATION_THRESHOLD,
        )
        self.inhand_replan_rotation_threshold_deg = _param(
            "inhand_replan_rotation_threshold_deg",
            INHAND_REPLAN_ROTATION_THRESHOLD_DEG,
        )
        self.inhand_replan_place_z_offset = _param(
            "inhand_replan_place_z_offset",
            Z_OFFSET_PLACE,
        )
        self.place_z_offset = float(_param("place_z_offset", ONLINE_PLACE_Z_OFFSET))
        self.inhand_replan_mode = str(
            _param("inhand_replan_mode", seq.INHAND_REPLAN_MODE)
        ).strip().lower()
        if self.inhand_replan_mode not in {"direct", "regrasp"}:
            raise ValueError("INHAND_REPLAN_MODE must be either 'direct' or 'regrasp'")

        cfg_path = options.get("config")
        if cfg_path is None:
            seed_cfg_path = os.path.join(self.plan_dir, "config.yml")
            cfg_path = (
                seed_cfg_path
                if seed_plan_dir is not None and os.path.exists(seed_cfg_path)
                else os.path.join("agent", "configs", "config.yml")
            )
        else:
            cfg_path = seq._resolve_config_arg(str(cfg_path))
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config path does not exist: {cfg_path}")
        cfg = self.OmegaConf.load(cfg_path)
        cfg = seq._apply_config_overrides(cfg, config_overrides)
        seq._set_action_score_excavator_xy(cfg, self.target_structure_offset)
        old_env_ground_height = self.environment_ground_height(cfg.environment)
        self.set_environment_ground_height(cfg.environment, self.place_plane_height)
        if abs(old_env_ground_height - self.place_plane_height) > 1e-9:
            self._log(
                "Updating online planning support ground_z: "
                f"{old_env_ground_height:.4f} -> {self.place_plane_height:.4f}"
            )

        pose_data_refresh_from_config = (
            options.get("config") is not None
            or seq._config_overrides_key(
                config_overrides,
                "environment.action.pose_data_path",
            )
        )
        local_pose_data_path = os.path.join(self.plan_dir, "pose_data.pkl")
        if pose_data_refresh_from_config:
            pose_data_load_path = seq._cfg_pose_data_path(cfg)
            pose_data_source = "config override"
        else:
            pose_data_load_path = (
                local_pose_data_path
                if seed_plan_dir is not None and os.path.exists(local_pose_data_path)
                else (
                    _param("pose_data_path", None)
                    or _param("pose_data_source_path", None)
                    or seq._cfg_pose_data_path(cfg)
                )
            )
            pose_data_source = (
                "seed branch"
                if pose_data_load_path == local_pose_data_path
                else "planning params"
            )
        if _param("poses", None) is not None and not pose_data_refresh_from_config:
            self.poses = {
                int(stone_id): self._pose_array(pose, stone_id=stone_id)
                for stone_id, pose in _param("poses", {}).items()
            }
        else:
            self.poses = seq._load_pose_data(pose_data_load_path)
        self._log(
            "Online pose data source: "
            f"{pose_data_load_path} ({pose_data_source})"
        )
        pose_data_save_path = os.path.join(self.plan_dir, "pose_data.pkl")
        pose_data_runtime_path = seq._write_pose_data_copy(
            self.poses,
            pose_data_load_path,
            pose_data_save_path,
        )
        cfg.environment.action.pose_data_path = pose_data_runtime_path

        self._online_seed_action_sequence = []
        self._online_seed_motion_sequence = []
        self._online_seed_motion_result_sequence = []
        self._online_seed_debug_data = seed_debug_data if isinstance(seed_debug_data, dict) else {}
        self._online_seed_plan_dir = self.plan_dir if seed_plan_dir is not None else None
        self.resume_scene_pose_order = []
        self.resume_scene_poses = {}
        self._sceneid_prior_poses = {}
        self.start_step = 1

        if seed_plan_dir is not None:
            self.planning_params = seed_planning_params
            self.action_sequence = copy.deepcopy(seed_actions)
            self.motion_sequence = copy.deepcopy(seed_motions)
            self.motion_result_sequence = copy.deepcopy(seed_motion_results)
            self.start_step = self._resolve_execution_start_step(
                total_actions=len(self.action_sequence)
            )
            self.resume_scene_poses = self._load_resume_scene_poses()
            self._reconcile_action_sequence_with_resume_scene_poses()
            self._online_seed_action_sequence = copy.deepcopy(self.action_sequence)
            self._online_seed_motion_sequence = copy.deepcopy(self.motion_sequence)
            self._online_seed_motion_result_sequence = copy.deepcopy(
                self.motion_result_sequence
            )
            prefix_len = min(max(int(self.start_step) - 1, 0), len(self.action_sequence))
            if prefix_len != max(int(self.start_step) - 1, 0):
                self._log(
                    f"Requested/metadata start_step {self.start_step} is after "
                    f"the seed plan length ({len(self.action_sequence)}); "
                    f"starting online execution at step {prefix_len + 1}."
                )
                self.start_step = prefix_len + 1
            self.action_sequence = copy.deepcopy(
                self._online_seed_action_sequence[:prefix_len]
            )
            self.motion_sequence = copy.deepcopy(
                self._online_seed_motion_sequence[:prefix_len]
            )
            self.motion_result_sequence = copy.deepcopy(
                self._online_seed_motion_result_sequence[:prefix_len]
            )
            prefix_ids = {
                int(action["stone_id"])
                for action in self.action_sequence
                if isinstance(action, dict) and "stone_id" in action
            }
            self._sceneid_prior_poses = {
                int(stone_id): self._pose_as_world_scene_frame(int(stone_id), pose)
                for stone_id, pose in self.resume_scene_poses.items()
                if int(stone_id) in prefix_ids
            }
        else:
            self.action_sequence = []
            self.motion_sequence = []
            self.motion_result_sequence = []
        self.planning_params = {
            "online_mcts": True,
            "online_seed_plan_dir": self._online_seed_plan_dir,
            "online_seed_steps": len(self._online_seed_action_sequence),
            "q_home": self.q_home,
            "target_structure_offset": self.target_structure_offset,
            "n_move": self.n_move,
            "n_grasp": self.n_grasp,
            "n_opening_angle": self.n_opening_angle,
            "regrasp_xy_pos": self.regrasp_xy_pos,
            "regrasp_xy_candidates": self.regrasp_xy_candidates,
            "max_num_regrasp_solutions": self.online_motion_max_num_regrasp_solutions,
            "execution_max_num_regrasp_solutions": self.max_num_regrasp_solutions,
            "pick_plane_height": self.pick_plane_height,
            "place_plane_height": self.place_plane_height,
            "place_z_offset": float(self.place_z_offset),
            "inhand_replan_mode": self.inhand_replan_mode,
            "poses": self.poses,
            "pose_data_path": pose_data_runtime_path,
            "pose_data_source_path": pose_data_load_path,
            "config_path": os.path.abspath(cfg_path),
            "config_overrides": config_overrides,
            "execution_start_step": self.start_step,
        }
        if seed_planning_params.get("branched_from") is not None:
            self.planning_params["branched_from"] = seed_planning_params["branched_from"]
        if self.resume_scene_poses:
            self.planning_params["resume_scene_poses"] = {
                int(stone_id): np.asarray(pose, dtype=np.float64).copy()
                for stone_id, pose in self.resume_scene_poses.items()
            }
        self.q_joint = self.q_home.copy()
        with open(os.path.join(self.plan_dir, "planning_params.pkl"), "wb") as f:
            pickle.dump(self.planning_params, f)
        self.OmegaConf.save(cfg, os.path.join(self.plan_dir, "config.yml"))
        with open(os.path.join(self.plan_dir, "pose_data.pkl"), "wb") as f:
            pickle.dump(self.poses, f)
        self._save_online_checkpoint()
        self._log(f"Online MCTS session directory: {self.plan_dir}")
        if self._online_seed_action_sequence:
            self._log(
                "Loaded seed plan for branch-and-online-replan: "
                f"{len(self._online_seed_action_sequence)} planned step(s), "
                f"prefix restored before execution: {len(self.action_sequence)}."
            )
            self._log(
                "Seed plan artifacts are preserved as seed_* files in the "
                "branch directory."
            )
        self._log(
            "Online planning will plan, review, execute, scan, and then plan "
            "the next step from the updated scene."
        )
        self._log(f"q_home: {self.q_home.tolist()}")
        self._log(f"In-hand replan mode: {self.inhand_replan_mode}")
        self._log(
            "Planner plane heights: "
            f"pick={self.pick_plane_height:.4f}, "
            f"place={self.place_plane_height:.4f}"
        )
        if LIVE_JOINT_VIEWER_ON:
            self._log(
                "Live joint 3D updates are enabled "
                f"({self._viewer_min_interval:.2f}s min interval)."
            )
        else:
            self._log("Live joint 3D updates are disabled during execution.")

    def _resolve_execution_start_step(self, total_actions: int) -> int:
        if self._requested_start_step is not None:
            start_step = int(self._requested_start_step)
            self._log(f"Using command-line start_step: {start_step}")
        else:
            start_step = self._planning_params_execution_start_step()
            source = "planning_params.pkl"
            if start_step is None:
                start_step = self._state_pkl_execution_start_step()
                source = "state.pkl"
            if start_step is None:
                start_step = 1
                source = "default"
            self._log(f"Auto-selected start_step {start_step} from {source}.")

        if start_step < 1:
            raise ValueError(f"start_step must be >= 1, got {start_step}")
        if total_actions > 0 and start_step > total_actions:
            self._log(
                f"start_step {start_step} is after the last action "
                f"({total_actions}); execution will only reconstruct saved steps."
            )
        return start_step

    def _planning_params_execution_start_step(self) -> int | None:
        for key in ("execution_start_step", "start_step"):
            value = self.planning_params.get(key, None)
            if value is None:
                continue
            try:
                start_step = int(value)
            except (TypeError, ValueError):
                self._log(f"Ignoring non-integer planning_params[{key!r}]: {value!r}")
                continue
            if start_step >= 1:
                return start_step
            self._log(f"Ignoring invalid planning_params[{key!r}]: {value!r}")
        return None

    def _state_pkl_execution_start_step(self) -> int | None:
        state_path = os.path.join(self.plan_dir, "state.pkl")
        if not os.path.exists(state_path):
            return None
        try:
            with open(state_path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            self._log(f"Could not inspect {state_path} for start_step: {exc}")
            return None
        if not isinstance(data, dict):
            return None

        steps = data.get("steps", [])
        if isinstance(steps, list) and steps:
            step_values = []
            for item in steps:
                if not isinstance(item, dict):
                    continue
                try:
                    step_values.append(int(item.get("step")))
                except (TypeError, ValueError):
                    continue
            if step_values:
                return max(1, min(step_values))

        resume_step = data.get("resume_step", None)
        if resume_step is not None:
            try:
                resume_step = int(resume_step)
            except (TypeError, ValueError):
                return None
            if resume_step >= 0:
                return resume_step + 1
        return None

    def _load_resume_scene_poses(self) -> dict[int, np.ndarray]:
        self.resume_scene_pose_order = []
        if self.start_step <= 1:
            return {}
        from_params = self._coerce_pose_dict(
            self.planning_params.get("resume_scene_poses", {})
        )
        if from_params:
            self.resume_scene_pose_order = list(from_params.keys())
            return from_params

        data = self._load_state_debug_data()
        if not isinstance(data, dict):
            return {}

        from_step = self._resume_scene_poses_from_debug_steps(data)
        if from_step:
            return from_step

        return self._resume_scene_poses_from_state_obj(data.get("resume_state", None))

    def _load_state_debug_data(self):
        state_path = os.path.join(self.plan_dir, "state.pkl")
        if not os.path.exists(state_path):
            return None
        try:
            with open(state_path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            self._log(f"Could not inspect {state_path} for resumed scene poses: {exc}")
            return None

    def _resume_scene_poses_from_debug_steps(self, data: dict) -> dict[int, np.ndarray]:
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            return {}

        selected = None
        selected_step = None
        for item in steps:
            if not isinstance(item, dict):
                continue
            try:
                step = int(item.get("step"))
            except (TypeError, ValueError):
                continue
            if step < self.start_step:
                continue
            if selected is None or step < selected_step:
                selected = item
                selected_step = step
        if selected is None:
            return {}

        scene = selected.get("scene", {})
        if not isinstance(scene, dict):
            return {}
        poses = self._coerce_pose_dict(scene.get("stone_poses", {}))
        order = []
        for stone_id in scene.get("stone_seq", []) or []:
            try:
                sid = int(stone_id)
            except (TypeError, ValueError):
                continue
            if sid in poses:
                order.append(sid)
        self.resume_scene_pose_order = order or list(poses.keys())
        return {sid: poses[sid] for sid in self.resume_scene_pose_order}

    def _resume_scene_poses_from_state_obj(self, state) -> dict[int, np.ndarray]:
        if state is None:
            return {}
        stone_poses = getattr(state, "stone_poses", None)
        if not isinstance(stone_poses, dict):
            return {}
        placed_ids = set()
        stone_set = getattr(state, "stone_set", None)
        stone_seq = getattr(state, "stone_seq", None)
        if stone_set is not None and stone_seq is not None:
            for idx in stone_seq:
                try:
                    placed_ids.add(int(stone_set[int(idx)]))
                except (TypeError, ValueError, IndexError):
                    continue
        poses = self._coerce_pose_dict(stone_poses)
        if placed_ids:
            poses = {sid: pose for sid, pose in poses.items() if sid in placed_ids}
        order = []
        if stone_set is not None and stone_seq is not None:
            for idx in stone_seq:
                try:
                    sid = int(stone_set[int(idx)])
                except (TypeError, ValueError, IndexError):
                    continue
                if sid in poses:
                    order.append(sid)
        self.resume_scene_pose_order = order or list(poses.keys())
        return poses

    def _reconcile_action_sequence_with_resume_scene_poses(self) -> None:
        prefix_len = max(0, int(self.start_step) - 1)
        if prefix_len <= 0 or not self.resume_scene_poses:
            return

        order = list(getattr(self, "resume_scene_pose_order", []) or [])
        if not order:
            order = list(self.resume_scene_poses.keys())
        expected_ids = [
            int(stone_id)
            for stone_id in order
            if int(stone_id) in self.resume_scene_poses
        ][:prefix_len]
        if len(expected_ids) < prefix_len:
            self._log(
                "Warning: resume_scene_poses has fewer ordered poses than the "
                f"skipped execution prefix ({len(expected_ids)} < {prefix_len}); "
                "using action_sequence for the missing prefix entries."
            )
            return

        old_prefix_ids = []
        for action in self.action_sequence[:prefix_len]:
            try:
                old_prefix_ids.append(int(action["stone_id"]))
            except (TypeError, ValueError, KeyError):
                old_prefix_ids.append(None)

        max_len = max(
            len(self.action_sequence),
            len(self.motion_sequence),
            len(self.motion_result_sequence),
        )
        records = []
        for idx in range(max_len):
            action = (
                copy.deepcopy(self.action_sequence[idx])
                if idx < len(self.action_sequence)
                else None
            )
            stone_id = None
            if isinstance(action, dict) and "stone_id" in action:
                try:
                    stone_id = int(action["stone_id"])
                except (TypeError, ValueError):
                    stone_id = None
            records.append(
                {
                    "index": idx,
                    "stone_id": stone_id,
                    "action": action,
                    "motion": (
                        self.motion_sequence[idx]
                        if idx < len(self.motion_sequence)
                        else None
                    ),
                    "motion_result": (
                        self.motion_result_sequence[idx]
                        if idx < len(self.motion_result_sequence)
                        else None
                    ),
                }
            )

        used = set()

        def take_record(stone_id: int):
            for rec in records:
                if rec["index"] in used:
                    continue
                if rec["stone_id"] == int(stone_id):
                    used.add(rec["index"])
                    return rec
            return None

        new_actions = []
        new_motions = []
        new_motion_results = []
        inserted_ids = []
        for stone_id in expected_ids:
            rec = take_record(stone_id)
            if rec is None or not isinstance(rec["action"], dict):
                action = {"stone_id": int(stone_id)}
                motion = None
                motion_result = None
                inserted_ids.append(int(stone_id))
            else:
                action = copy.deepcopy(rec["action"])
                motion = rec["motion"]
                motion_result = rec["motion_result"]
            action["stone_id"] = int(stone_id)
            action["pose"] = self.resume_scene_poses[int(stone_id)].copy()
            new_actions.append(action)
            new_motions.append(motion)
            new_motion_results.append(motion_result)

        prefix_id_set = set(expected_ids)
        skipped_duplicates = []
        for rec in records:
            if rec["index"] in used or not isinstance(rec["action"], dict):
                continue
            if rec["stone_id"] in prefix_id_set:
                skipped_duplicates.append(int(rec["stone_id"]))
                continue
            new_actions.append(copy.deepcopy(rec["action"]))
            new_motions.append(rec["motion"])
            new_motion_results.append(rec["motion_result"])

        changed = (
            old_prefix_ids != expected_ids
            or len(new_actions) != len(self.action_sequence)
            or inserted_ids
            or skipped_duplicates
        )
        if not changed:
            return

        self._log(
            "Warning: action_sequence skipped prefix differed from "
            "resume_scene_poses; reconstructing the in-memory prefix from "
            "resume state. "
            f"old_tail={old_prefix_ids[-5:]}, "
            f"new_tail={expected_ids[-5:]}, "
            f"inserted={inserted_ids}, "
            f"skipped_duplicates={skipped_duplicates}"
        )
        self.action_sequence = new_actions
        self.motion_sequence = new_motions
        self.motion_result_sequence = new_motion_results

    @staticmethod
    def _pose_array(pose, stone_id=None) -> np.ndarray:
        if isinstance(pose, (tuple, list)) and len(pose) >= 2:
            pos = np.asarray(pose[0], dtype=float).reshape(-1)
            quat = np.asarray(pose[1], dtype=float).reshape(-1)
            if pos.shape[0] >= 3 and quat.shape[0] >= 4:
                arr = np.concatenate([pos[:3], quat[:4]])
            else:
                arr = np.asarray(pose, dtype=float).reshape(-1)
        else:
            arr = np.asarray(pose, dtype=float).reshape(-1)

        if arr.shape[0] < 7 or not np.all(np.isfinite(arr[:7])):
            label = f" for stone {stone_id}" if stone_id is not None else ""
            raise ValueError(f"Invalid pose{label}: {pose!r}")
        return arr[:7].copy()

    @staticmethod
    def _coerce_pose_dict(stone_poses) -> dict[int, np.ndarray]:
        if not isinstance(stone_poses, dict):
            return {}
        poses = {}
        for stone_id, pose in stone_poses.items():
            try:
                sid = int(stone_id)
            except (TypeError, ValueError):
                continue
            try:
                poses[sid] = ExecutionPlanStateMixin._pose_array(
                    pose,
                    stone_id=sid,
                )
            except (TypeError, ValueError):
                continue
        return poses

    def _scene_config_pose_array(self, stone_id: int) -> np.ndarray | None:
        config = getattr(self, "scene_configs", {}).get(int(stone_id))
        if config is None:
            return None
        try:
            return self._pose_array_from_matrix(config.pose.as_matrix())
        except Exception:
            return None

    def _action_local_pose_for_stone(self, stone_id: int) -> np.ndarray | None:
        for action in self.action_sequence:
            try:
                action_stone_id = int(action["stone_id"])
            except (TypeError, ValueError, KeyError):
                continue
            if action_stone_id != int(stone_id):
                continue
            try:
                return self._pose_array(action["pose"], stone_id=stone_id)
            except (TypeError, ValueError, KeyError):
                return None
        return None

    def _pose_as_world_scene_frame(self, stone_id: int, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64).copy().reshape(-1)
        if pose.shape[0] < 7:
            raise ValueError(f"Invalid state pose for stone {stone_id}: {pose!r}")
        pose = pose[:7]

        offset = np.asarray(self.target_structure_offset[:2], dtype=np.float64)
        if np.linalg.norm(offset) <= 1e-9:
            return pose

        shifted = pose.copy()
        shifted[:2] += offset
        references = []
        action_pose = self._action_local_pose_for_stone(stone_id)
        if action_pose is not None:
            action_world = action_pose.copy()
            action_world[:2] += offset
            references.append(action_world)
        scene_pose = self._scene_config_pose_array(stone_id)
        if scene_pose is not None:
            references.append(scene_pose)
        if not references:
            return shifted

        raw_error = min(
            float(np.linalg.norm(pose[:2] - reference[:2])) for reference in references
        )
        shifted_error = min(
            float(np.linalg.norm(shifted[:2] - reference[:2]))
            for reference in references
        )
        if raw_error + 0.20 < shifted_error:
            return pose
        return shifted

    def _state_pose_as_world_scene_pose(
        self, stone_id: int, pose: np.ndarray
    ) -> np.ndarray:
        return self._pose_as_world_scene_frame(stone_id, pose)

    def _scene_pose_prior_entries_from_state(
        self, n_step: int
    ) -> list[tuple[int, np.ndarray]]:
        step_index = n_step + 1
        data = self._load_state_debug_data()
        if not isinstance(data, dict):
            self._log("Scene pose init unavailable: state.pkl was not loaded.")
            return []

        steps = data.get("steps", [])
        if not isinstance(steps, list):
            steps = []

        candidates = []
        for item in steps:
            if not isinstance(item, dict):
                continue
            try:
                item_step = int(item.get("step"))
            except (TypeError, ValueError):
                continue
            if item_step != step_index:
                continue
            candidates.extend([item.get("resume_state"), item.get("raw_state")])
            break
        candidates.append(data.get("resume_state"))
        for item in reversed(steps):
            if isinstance(item, dict):
                candidates.extend([item.get("resume_state"), item.get("raw_state")])

        action_ids = [
            int(action["stone_id"]) for action in self.action_sequence[:step_index]
        ]
        state_poses: dict[int, np.ndarray] = {}
        for state in candidates:
            if state is None:
                continue
            poses = self._coerce_pose_dict(getattr(state, "stone_poses", None))
            if not poses:
                continue

            state_ids = set()
            stone_set = getattr(state, "stone_set", None)
            stone_seq = getattr(state, "stone_seq", None)
            if stone_set is not None and stone_seq is not None:
                for stone_idx in stone_seq:
                    try:
                        state_ids.add(int(stone_set[int(stone_idx)]))
                    except (TypeError, ValueError, IndexError):
                        continue
            else:
                state_ids = set(poses)

            for stone_id in action_ids:
                if stone_id not in state_ids or stone_id not in poses:
                    continue
                state_poses[stone_id] = self._state_pose_as_world_scene_pose(
                    stone_id,
                    poses[stone_id],
                )
            if all(stone_id in state_poses for stone_id in action_ids):
                break

        entries = [
            (stone_id, state_poses[stone_id])
            for stone_id in action_ids
            if stone_id in state_poses
        ]
        missing = [stone_id for stone_id in action_ids if stone_id not in state_poses]
        if missing:
            self._log(
                "Scene pose init missing state.pkl poses for placed stone ids: "
                f"{missing}"
            )
        return entries

    def _scene_pose_prior_entries_from_current_scene(
        self, n_step: int
    ) -> list[tuple[int, np.ndarray]]:
        entries = []
        for action_i, action in enumerate(self.action_sequence[: n_step + 1]):
            try:
                stone_id = int(action["stone_id"])
            except (TypeError, ValueError, KeyError):
                continue
            pose = self._scene_config_pose_array(stone_id)
            if pose is None:
                try:
                    pose = self._place_pose_for_action(action, action_i)
                except Exception as exc:
                    self._log(
                        "Scene pose init fallback skipped stone " f"{stone_id}: {exc}"
                    )
                    continue
            entries.append((stone_id, np.asarray(pose, dtype=np.float64).copy()))
        return entries

    def _publish_scene_pose_prior(self, entries: list[tuple[int, np.ndarray]]) -> None:
        pose_with_id_list = [
            (pose[:3].copy(), pose[3:7].copy(), stone_id) for stone_id, pose in entries
        ]
        self.scene_pose_init_pub.publish(pose_with_id_list)

    def _resolve_plan_dir(self, repo_dir: str, plan_id: str):
        desktop_dir = os.path.join(repo_dir, "scripts", "desktop")

        def plan_date(plan_name: str):
            parts = plan_name.split("_")
            if len(parts) >= 3 and parts[0] == "plan" and parts[1].isdigit():
                return parts[1]
            return None

        def session_candidates(root_dir: str, plan_name: str):
            candidates = []
            date_str = plan_date(plan_name)
            if date_str is not None:
                candidates.append(
                    os.path.join(root_dir, "sessions", date_str, plan_name)
                )
            candidates.append(os.path.join(root_dir, "sessions", plan_name))
            return candidates

        candidates = []
        plan_ref = os.path.normpath(plan_id)
        plan_ref_name = os.path.basename(plan_ref)
        plan_ref_parent = os.path.dirname(plan_ref)
        if os.path.isdir(plan_ref):
            candidates.append(plan_ref)
        if plan_ref_name.startswith("plan_"):
            date_str = plan_date(plan_ref_name)
            if date_str is not None and plan_ref_parent:
                candidates.append(
                    os.path.join(plan_ref_parent, date_str, plan_ref_name)
                )
            candidates.extend(session_candidates(repo_dir, plan_ref_name))
            candidates.extend(session_candidates(desktop_dir, plan_ref_name))
        else:
            plan_name = f"plan_{plan_id}"
            candidates.extend(session_candidates(repo_dir, plan_name))
            candidates.extend(session_candidates(desktop_dir, plan_name))
        candidates.append(plan_ref)

        for path in candidates:
            path = os.path.normpath(path)
            if os.path.isdir(path):
                return path

        raise FileNotFoundError(
            "Plan directory not found. Tried:\n  "
            + "\n  ".join(os.path.normpath(path) for path in candidates)
        )
