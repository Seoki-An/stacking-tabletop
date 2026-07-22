"""Core worker lifecycle, GUI callbacks, runtime imports, and execution dispatch."""

from .execution_common import *


class ExecutionBaseMixin:
    def __init__(
        self,
        plan_id: str | None,
        start_step: int | None,
        online_options: dict | None = None,
        control_options: dict | None = None,
        test_mode: str | None = None,
        on_log=None,
        on_status=None,
        on_request_decision=None,
        on_display_geometries=None,
        on_live_joint_state=None,
        on_finished=None,
        on_failed=None,
    ):
        self.plan_id = plan_id
        self.online_options = online_options
        control_options = control_options or {}
        self.move_control_error_tol = float(
            control_options.get("move_control_error_tol", 0.02)
        )
        self.move_control_convergence_time_limit = float(
            control_options.get("move_control_convergence_time_limit", 50.0)
        )
        if (
            not np.isfinite(self.move_control_error_tol)
            or self.move_control_error_tol <= 0
        ):
            raise ValueError("move_control_error_tol must be a finite positive value")
        if (
            not np.isfinite(self.move_control_convergence_time_limit)
            or self.move_control_convergence_time_limit <= 0
        ):
            raise ValueError(
                "move_control_convergence_time_limit must be a finite positive value"
            )
        self.online_mode = online_options is not None
        self.test_mode = test_mode
        self._requested_start_step = start_step
        self.start_step = 1
        self._decision_event = threading.Event()
        self._decision = None
        self._abort = threading.Event()
        self._manual_place = threading.Event()
        self._manual_place_inhand_T = None
        self._manual_place_opening_angle = 0.0
        self._place_inhand_T_override = None
        self._place_opening_angle_override = None

        self._on_log = on_log or (lambda msg: print(msg))
        self._on_status = on_status or (lambda st: None)
        self._on_request_decision = on_request_decision or (lambda k, p: None)
        display_cb = on_display_geometries or (lambda g, t: None)

        def _display_geometries(geoms, title):
            self._last_live_layout_key = None
            display_cb(geoms, title)

        self._on_display_geometries = _display_geometries
        self._on_live_joint_state = on_live_joint_state or (lambda payload: None)
        self._on_finished = on_finished or (lambda: None)
        self._on_failed = on_failed or (lambda msg: None)
        self._log(
            "Move control limits: "
            f"error_tol={self.move_control_error_tol:.4f} rad, "
            "convergence_time_limit="
            f"{self.move_control_convergence_time_limit:.2f}s"
        )

        self._reset_viewer_state()
        self._last_viewer_emit = 0.0
        self._viewer_min_interval = LIVE_JOINT_VIEWER_MIN_INTERVAL
        self._last_live_q = None
        self._stone_pcd_mode = False
        self._stone_highpoly_mesh_mode = False
        self._lidar_frame_mode = False
        self._pick_adjust_base_q = None
        self._pick_adjust_q = None
        self._pick_adjust_grab = "open"
        self._pick_adjust_phase = "pick"
        self._place_retry_z_offset = 0.0
        self._last_logged_place_retry_z_offset = None
        self._regrasp_q_start = None
        self._regrasp_q_mid = None
        self._regrasp_q_end = None
        self._ros_spin_lock = threading.Lock()
        self._live_joint_stop = threading.Event()
        self._live_joint_pause = threading.Event()
        self._live_joint_thread = None
        self._live_layout_suppressed = False
        self._last_live_layout_key = None
        self._sceneid_prior_poses: dict[int, np.ndarray] = {}
        self._sceneid_ground_height: float | None = None
        self._sceneid_ground_plane: np.ndarray | None = None
        self._scene_pcd_transfer = None
        self._scene_scan_display_entry = None
        self._online_debug_data = None
        self._online_last_node_done = False
        self._online_auto_accept_saved_motion_steps: set[int] = set()
        self._online_seed_prefix_restored = False
        self._subgoal_pcd_scan_lock = threading.Lock()
        self._subgoal_pcd_scan_thread = None
        self._subgoal_pcd_scan_counter = 0
        self._subgoal_pcd_transfer = None
        self._requested_scan_pcd_transfer = None
        self._current_step_index = None
        self._current_target_id = None

    # --- Callbacks invoked by the GUI ----------------------------------------
    def on_decision(self, value: str):
        self._decision = value
        self._decision_event.set()

    def on_abort(self):
        self._abort.set()
        # Unblock any pending wait so the worker can return promptly.
        self._decision = "abort"
        self._decision_event.set()

    def on_manual_place(self):
        self._manual_place.set()
        self._decision = "manual_place"
        self._decision_event.set()
        self._log(
            "Manual place requested; in-hand replanning will switch to "
            "operator-driven place control."
        )

    def on_stone_pcd_mode_changed(self, enabled: bool):
        self._stone_pcd_mode = bool(enabled)
        mode = "point cloud" if self._stone_pcd_mode else "mesh"
        self._log(f"Stone visualization mode: {mode}")

    def on_stone_mesh_mode_changed(self, enabled: bool):
        self._stone_highpoly_mesh_mode = bool(enabled)
        mode = "high-poly mesh" if self._stone_highpoly_mesh_mode else "DSF mesh"
        self._log(f"Stone mesh visualization source: {mode}")

    def on_lidar_frame_mode_changed(self, enabled: bool):
        self._lidar_frame_mode = bool(enabled)
        mode = "shown" if self._lidar_frame_mode else "hidden"
        self._last_live_layout_key = None
        self._log(f"LiDAR frame visualization: {mode}")
        if self._last_live_q is not None:
            self._live_state_cb(self._last_live_q)

    def on_refresh_requested(self):
        self._last_live_layout_key = None
        q = None
        if ROS_CONTROL_ON and LIVE_JOINT_VIEWER_ON:
            node = getattr(self, "joint_node_sub", None)
            if node is not None:
                try:
                    with self._ros_spin_lock:
                        self.rclpy.spin_once(node, timeout_sec=0.10)
                    if node.get_flag():
                        q = node.pos.copy()
                        node.reset_get_flag()
                    elif getattr(node, "pos", None) is not None:
                        q = np.asarray(node.pos, dtype=np.float64).copy()
                except Exception as exc:
                    self._log(f"GUI refresh failed while reading joint state: {exc}")

        if q is None and self._last_live_q is not None:
            q = np.asarray(self._last_live_q, dtype=np.float64).copy()
        if q is None and hasattr(self, "q_joint"):
            q = np.asarray(self.q_joint, dtype=np.float64).copy()
        if q is None and hasattr(self, "q_home"):
            q = np.asarray(self.q_home, dtype=np.float64).copy()

        if q is None:
            self._log("GUI refresh requested, but no joint state is available.")
            return
        self._live_state_cb(q)
        self._log("GUI refresh requested: live joint/viewer state updated.")

    def on_pick_grasp_adjust(self, joint: str, delta_deg: float):
        if self._pick_adjust_q is None:
            return
        joint_indices = {
            "swing": 0,
            "boom": 1,
            "arm": 2,
            "bucket": 3,
            "tilt": 4,
            "rotate": 5,
        }
        if joint not in joint_indices:
            return
        idx = joint_indices[joint]
        self._pick_adjust_q[idx] += np.deg2rad(delta_deg)
        self._publish_pick_adjusted_target()

    def on_pick_grasp_adjust_reset(self):
        if self._pick_adjust_base_q is None:
            return
        self._pick_adjust_q = self._pick_adjust_base_q.copy()
        self._publish_pick_adjusted_target()

    def on_place_z_offset_changed(self, offset: float):
        try:
            offset = float(offset)
        except (TypeError, ValueError):
            return
        if not np.isfinite(offset):
            return
        self._place_retry_z_offset = offset
        self._log(f"Motion retry place Z offset set to {offset:+.3f} m")

    # --- helpers ------------------------------------------------------------
    def _log(self, msg: str):
        print(msg)
        self._on_log(msg)

    def _set_status(self, **kwargs):
        self._on_status(kwargs)

    def _load_pickle_or_default(self, name: str, default):
        path = os.path.join(self.plan_dir, name)
        if not os.path.exists(path):
            return copy.deepcopy(default)
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            self._log(f"Could not load {path}: {exc}")
            return copy.deepcopy(default)

    def _preserve_seed_plan_file(self, source_name: str, seed_name: str) -> None:
        source_path = os.path.join(self.plan_dir, source_name)
        seed_path = os.path.join(self.plan_dir, seed_name)
        if not os.path.exists(source_path) or os.path.exists(seed_path):
            return
        try:
            with open(source_path, "rb") as src, open(seed_path, "wb") as dst:
                dst.write(src.read())
        except Exception as exc:
            self._log(
                f"Warning: could not preserve seed plan file {source_name}: {exc}"
            )

    def _live_joint_state_cb(self):
        return self._live_state_cb if LIVE_JOINT_VIEWER_ON else None

    def _live_joint_node_for_viewer(self):
        if ROS_CONTROL_ON and LIVE_JOINT_VIEWER_ON:
            return self.joint_node_sub
        return None

    def _wait_for_decision(self, kind: str, payload=None) -> str:
        if self._abort.is_set():
            return "abort"
        self._decision = None
        self._decision_event.clear()
        self._on_request_decision(kind, payload or {})
        self._decision_event.wait()
        return self._decision or "abort"

    def _start_live_joint_polling(self):
        if (
            not ROS_CONTROL_ON
            or not LIVE_JOINT_VIEWER_ON
            or self._live_joint_thread is not None
        ):
            return

        self._live_joint_stop.clear()

        def _poll():
            while not self._live_joint_stop.is_set() and not self._abort.is_set():
                if self._live_joint_pause.is_set():
                    time.sleep(0.02)
                    continue
                try:
                    with self._ros_spin_lock:
                        self.rclpy.spin_once(self.joint_node_sub, timeout_sec=0.05)
                    if self.joint_node_sub.get_flag():
                        self._live_state_cb(self.joint_node_sub.pos.copy())
                        self.joint_node_sub.reset_get_flag()
                except Exception as exc:
                    self._log(f"Live joint GUI update stopped: {exc}")
                    return

        self._live_joint_thread = threading.Thread(target=_poll, daemon=True)
        self._live_joint_thread.start()

    def _stop_live_joint_polling(self):
        self._live_joint_stop.set()
        if self._live_joint_thread is not None:
            self._live_joint_thread.join(timeout=1.0)
            self._live_joint_thread = None

    @contextmanager
    def _pause_live_joint_polling(self):
        self._live_joint_pause.set()
        try:
            yield
        finally:
            self._live_joint_pause.clear()

    @contextmanager
    def _suppress_live_layout_updates(self):
        previous = self._live_layout_suppressed
        self._live_layout_suppressed = True
        try:
            yield
        finally:
            self._live_layout_suppressed = previous

    # --- entrypoint ---------------------------------------------------------
    def run(self):
        try:
            self._execute()
        except Exception as exc:
            tb = traceback.format_exc()
            self._on_failed(f"{exc}\n{tb}")
        else:
            self._on_finished()
        finally:
            self._stop_live_joint_polling()
            self._shutdown_runtime_dependencies()

    def _shutdown_runtime_dependencies(self):
        nodes = []
        for attr in (
            "joint_node_pub",
            "joint_node_sub",
            "phase_node_pub",
            "log_dir_pub",
            "scene_pose_init_pub",
            "scene_scan_done_sub",
            "scene_pcd_sub",
            "diagnostic_pcd_request_pub",
            "diagnostic_pcd_done_sub",
        ):
            node = getattr(self, attr, None)
            if node is not None:
                nodes.append(node)

        planner = getattr(self, "integrated_planner", None)
        for attr in (
            "inhand_pose_sub",
            "inhand_opening_angle_sub",
            "inhand_pose_pub",
            "field_pose_init_pub",
            "opening_angle_pub",
            "grasp_status_sub",
            "field_recovery_status_sub",
            "field_recovery_sub",
        ):
            node = getattr(planner, attr, None) if planner is not None else None
            if node is not None:
                nodes.append(node)

        seen = set()
        for node in nodes:
            if id(node) in seen:
                continue
            seen.add(id(node))
            try:
                node.destroy_node()
            except Exception:
                pass

        rclpy_mod = getattr(self, "rclpy", None)
        if rclpy_mod is not None:
            try:
                if rclpy_mod.ok():
                    rclpy_mod.shutdown()
            except Exception:
                pass

        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

    # --- main pipeline ------------------------------------------------------
    def _execute(self):
        self._load_runtime_dependencies()
        if self.online_mode:
            self._load_online_session_files()
        else:
            self._load_plan_files()
        self._initialize_ros_and_models()
        if self.online_mode:
            self._initialize_online_debug_state()
        self._initialize_scene()
        self._show_initial_scene()
        self._start_live_joint_polling()
        if self.test_mode is not None:
            if self.test_mode == "scene_scan":
                self._home_robot()
            self._execute_test_mode()
            return
        self._home_robot()
        if self.online_mode:
            self._execute_online_action_sequence()
        else:
            self._execute_action_sequence()

    def _load_runtime_dependencies(self):
        # Deferred imports keep Ray / CUDA / rclpy out of the main thread so
        # the Open3D GUI event loop starts cleanly before any heavy init.
        from omegaconf import OmegaConf
        import rclpy

        from agent import IntegratedPlanner
        from agent.env.components.action import Action
        from agent.mcts import MCTS_Node
        from agent.env.components.contexts import (
            environment_ground_height,
            get_posegen,
            set_environment_ground_height,
        )
        from model import (
            get_excavator_model,
            get_gripper_model,
            get_stone_model,
            update_urdf_mesh,
        )
        from planning import (
            Q_SCAN,
            Q_SCAN_INHAND,
            diffsim,
            generate_path_with_opening_angle,
            get_planner,
            motion_failure_can_retry_regrasp_xy,
            motion_failure_detail,
            motion_failure_summary,
            motion_failure_stage,
            motion_result_summary,
            num_regrasp_candidates,
            regrasp_candidate_score,
            regrasp_position_candidates,
            regrasp_planning,
            select_regrasp_candidate,
            normalize_joint_branches,
            split_regrasp_place_paths,
            solve_inhand_grasp_planning,
            trajectory_visualization_with_target,
            planner,
        )
        from ros2 import (
            CtrlJointPublisher,
            CtrlJointSubscriber,
            DiagnosticPcdRequestPublisher,
            LogDirPublisher,
            PhasePublisher,
            PoseArrayPublisher,
            SceneScanDoneSubscriber,
            position_control,
            sequential_grasp_control,
        )
        from ros2.sceneid_node import SceneIdentifierSubscriber
        from perception.sceneid_runtime import (
            correct_target_offset_frame_if_needed,
            identify_scene_dir_from_initial_poses,
            make_sceneid_runtime_args,
        )
        from scripts.desktop import generate_sequence as sequence_runtime
        from tp_msgs.msg import Phase
        from utils import get_unique_dir, resolve_thread_count

        self.Q_SCAN = Q_SCAN
        self.Q_SCAN_INHAND = Q_SCAN_INHAND
        self.OmegaConf = OmegaConf
        self.rclpy = rclpy
        self.IntegratedPlanner = IntegratedPlanner
        self.Action = Action
        self.MCTS_Node = MCTS_Node
        self.environment_ground_height = environment_ground_height
        self.get_posegen = get_posegen
        self.set_environment_ground_height = set_environment_ground_height
        self.get_excavator_model = get_excavator_model
        self.get_gripper_model = get_gripper_model
        self.get_stone_model = get_stone_model
        self.update_urdf_mesh = update_urdf_mesh
        self.diffsim = diffsim
        self.generate_path_with_opening_angle = generate_path_with_opening_angle
        self.planner = planner
        self.get_planner = get_planner
        self.regrasp_position_candidates = regrasp_position_candidates
        self.regrasp_planning = regrasp_planning
        self.motion_failure_stage = motion_failure_stage
        self.motion_failure_detail = motion_failure_detail
        self.motion_failure_can_retry_regrasp_xy = motion_failure_can_retry_regrasp_xy
        self.motion_failure_summary = motion_failure_summary
        self.motion_result_summary = motion_result_summary
        self.num_regrasp_candidates = num_regrasp_candidates
        self.select_regrasp_candidate = select_regrasp_candidate
        self.regrasp_candidate_score = regrasp_candidate_score
        self.normalize_joint_branches = normalize_joint_branches
        self.split_regrasp_place_paths = split_regrasp_place_paths
        self.solve_inhand_grasp_planning = solve_inhand_grasp_planning
        self.trajectory_visualization_with_target = trajectory_visualization_with_target
        self.CtrlJointPublisher = CtrlJointPublisher
        self.CtrlJointSubscriber = CtrlJointSubscriber
        self.DiagnosticPcdRequestPublisher = DiagnosticPcdRequestPublisher
        self.LogDirPublisher = LogDirPublisher
        self.PhasePublisher = PhasePublisher
        self.PoseArrayPublisher = PoseArrayPublisher
        self.SceneScanDoneSubscriber = SceneScanDoneSubscriber
        self.SceneIdentifierSubscriber = SceneIdentifierSubscriber
        self.identify_scene_dir_from_initial_poses = (
            identify_scene_dir_from_initial_poses
        )
        self.correct_target_offset_frame_if_needed = (
            correct_target_offset_frame_if_needed
        )
        self.make_sceneid_runtime_args = make_sceneid_runtime_args
        self.sequence_runtime = sequence_runtime
        self.position_control = position_control
        self.sequential_grasp_control = sequential_grasp_control
        self.Phase = Phase
        self.get_unique_dir = get_unique_dir
        self.resolve_thread_count = resolve_thread_count
