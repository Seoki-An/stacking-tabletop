"""Scene scan PCD transfer, desktop SceneID, preview, and scene-state updates."""

from .execution_common import *


class ExecutionSceneScanMixin:
    def _finish_step(self, n_step):
        with self._suppress_live_layout_updates():
            self._reset_viewer_state()

            try:
                if PERCEPTION_ON:
                    self._run_scene_scan(n_step, allow_planned_scene_prior=True)
            finally:
                self._viz_overlay_entries = []
                self._clear_scene_scan_pcd_preview(redraw=False)

    @staticmethod
    def _safe_scene_pcd_tag(frame_id: str, fallback: str) -> str:
        tag = (frame_id or "").strip().strip("/").replace("/", "_")
        return tag or fallback

    def _start_scene_pcd_transfer(self, scene_dir: str) -> None:
        os.makedirs(scene_dir, exist_ok=True)
        self._scene_scan_display_entry = None
        self._scene_pcd_transfer = {
            "scene_dir": scene_dir,
            "merged": o3d.geometry.PointCloud(),
            "frames": 0,
            "last_frame_time": None,
        }
        self.scene_pcd_sub.reset_get_flag()

    def _collect_scene_pcd_frame(self) -> bool:
        transfer = self._scene_pcd_transfer
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
        if not str(frame_id).startswith("scene_base_"):
            self._log(
                "Ignoring non-scene PCD frame during scene scan transfer: "
                f"{frame_id or '<empty>'}"
            )
            return False

        frame = o3d.geometry.PointCloud()
        frame.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        frame_count = int(transfer["frames"])
        tag = self._safe_scene_pcd_tag(
            frame_id,
            fallback=f"scene_base_transferred_swing{frame_count}",
        )
        if not tag.startswith("scene_base_"):
            tag = f"scene_base_{tag}"
        if "_swing" not in tag:
            tag = f"{tag}_swing{frame_count}"

        scene_dir = transfer["scene_dir"]
        save_path = os.path.join(scene_dir, f"{tag}.ply")
        if os.path.exists(save_path):
            save_path = os.path.join(scene_dir, f"{tag}_{frame_count}.ply")
        o3d.io.write_point_cloud(save_path, frame)

        transfer["merged"] += frame
        transfer["frames"] = frame_count + 1
        transfer["last_frame_time"] = time.time()
        if transfer["frames"] == 1 or transfer["frames"] % 5 == 0:
            self._log(
                "Received scene PCD frame "
                f"{transfer['frames']}: {os.path.basename(save_path)} "
                f"({len(frame.points)} points)"
            )
        return True

    def _drain_scene_pcd_after_done(self) -> None:
        drain_seconds = float(
            self.planning_params.get(
                "scene_pcd_done_drain_seconds",
                SCENE_PCD_DONE_DRAIN_SECONDS,
            )
        )
        deadline = time.time() + max(0.0, drain_seconds)
        while time.time() < deadline and not self._abort.is_set():
            with self._ros_spin_lock:
                self.rclpy.spin_once(self.scene_pcd_sub, timeout_sec=0.05)
            self._collect_scene_pcd_frame()

    def _finish_scene_pcd_transfer(self) -> bool:
        transfer = self._scene_pcd_transfer
        if transfer is None:
            return False

        scene_dir = transfer["scene_dir"]
        merged = transfer["merged"]
        frame_count = int(transfer["frames"])
        status_path = os.path.join(scene_dir, "scene_scan_status.txt")

        if merged.is_empty():
            with open(status_path, "w") as f:
                f.write("failed: desktop_received_no_scene_pcd_frames\n")
                f.write(f"transferred_frames: {frame_count}\n")
            self._log("No scene PCD frames were received from the NUC.")
            self._scene_pcd_transfer = None
            return False

        voxel = float(
            self.planning_params.get("scene_pcd_merge_voxel", SCENE_PCD_MERGE_VOXEL)
        )
        if voxel > 0.0:
            merged = merged.voxel_down_sample(voxel)

        merged_path = os.path.join(scene_dir, "scene_scan_merged.ply")
        o3d.io.write_point_cloud(merged_path, merged)
        self._scene_scan_display_entry = self._make_scene_scan_pcd_display_entry(
            merged,
            frame_count,
        )
        with open(status_path, "w") as f:
            f.write("succeeded\n")
            f.write("source: desktop_scene_pcd_transfer\n")
            f.write(f"transferred_frames: {frame_count}\n")
            f.write(f"merge_voxel: {voxel}\n")
            f.write(f"merged_points: {len(merged.points)}\n")

        self._log(
            "Saved transferred scene scan: "
            f"{frame_count} frame(s), {len(merged.points)} merged points."
        )
        self._scene_pcd_transfer = None
        return True

    def _make_scene_scan_pcd_display_entry(self, merged, frame_count: int):
        pcd = o3d.geometry.PointCloud(merged)
        voxel = float(
            self.planning_params.get(
                "scene_pcd_display_voxel",
                SCENE_PCD_DISPLAY_VOXEL,
            )
        )
        if voxel > 0.0:
            pcd = pcd.voxel_down_sample(voxel)
        if pcd.is_empty():
            return None
        pcd.paint_uniform_color(C_SCENE_SCAN_PCD)
        self._log(
            "Scene scan GUI preview PCD: "
            f"{len(pcd.points)} point(s), display_voxel={voxel}, "
            f"source_frames={int(frame_count)}."
        )
        return (
            pcd,
            C_SCENE_SCAN_PCD,
            {
                "name": "scene_scan_transferred_pcd",
                "alpha": 0.95,
            },
        )

    def _show_scene_scan_pcd_preview(self, n_step: int) -> None:
        if self._scene_scan_display_entry is None:
            return
        self._viz_overlay_entries = [self._scene_scan_display_entry]
        self._on_display_geometries(
            self._base_scene(self.Q_SCAN)
            + [self._scene_scan_display_entry]
            + [self.origin_frame],
            f"Step {n_step + 1}: transferred scene scan PCD",
        )

    def _clear_scene_scan_pcd_preview(self, redraw: bool = False) -> None:
        had_entry = self._scene_scan_display_entry is not None
        self._scene_scan_display_entry = None
        removed = self._remove_overlay_entries_by_name("scene_scan_transferred_pcd")
        if not redraw or not (had_entry or removed):
            return
        self._on_display_geometries(
            self._base_scene(self.Q_SCAN)
            + self._viz_overlay_entries
            + [self.origin_frame],
            "Scene scan PCD cleared",
        )

    def _run_scene_scan(self, n_step, allow_planned_scene_prior: bool = False):
        self._set_status(phase="Scene scan")
        step_log_dir = f"{self.base_log_dir}/step_{n_step + 1}"
        scene_dir = os.path.join(step_log_dir, "scene_scan")
        self.scene_scan_done_sub.reset_get_flag()
        self._start_scene_pcd_transfer(scene_dir)
        target_pos = np.array(
            [
                self.target_structure_offset[0],
                self.target_structure_offset[1],
                0.0,
            ],
            dtype=np.float64,
        )
        target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        scene_prior_entries = self._scene_pose_prior_entries_from_state(n_step)
        scene_prior_source = "state.pkl"
        if allow_planned_scene_prior:
            expected_ids = []
            for action in self.action_sequence[: n_step + 1]:
                try:
                    expected_ids.append(int(action["stone_id"]))
                except (TypeError, ValueError, KeyError):
                    continue
            pose_by_id = {
                int(stone_id): np.asarray(pose, dtype=np.float64).copy()
                for stone_id, pose in scene_prior_entries
            }
            missing_ids = [
                stone_id for stone_id in expected_ids if stone_id not in pose_by_id
            ]
            if missing_ids:
                current_entries = self._scene_pose_prior_entries_from_current_scene(n_step)
                current_by_id = {
                    int(stone_id): np.asarray(pose, dtype=np.float64).copy()
                    for stone_id, pose in current_entries
                }
                backfilled_ids = []
                for stone_id in missing_ids:
                    pose = current_by_id.get(stone_id)
                    if pose is None:
                        continue
                    pose_by_id[stone_id] = pose
                    backfilled_ids.append(stone_id)
                if backfilled_ids:
                    scene_prior_entries = [
                        (stone_id, pose_by_id[stone_id])
                        for stone_id in expected_ids
                        if stone_id in pose_by_id
                    ]
                    scene_prior_source = (
                        "state.pkl + current reconstructed scene"
                        if len(backfilled_ids) < len(scene_prior_entries)
                        else "current reconstructed scene"
                    )
                    self._log(
                        "Scene pose init backfilled from current reconstructed scene "
                        f"for missing stone ids: {backfilled_ids}"
                    )
        if scene_prior_entries:
            self._log(
                f"Scene pose init from {scene_prior_source}: "
                f"{len(scene_prior_entries)} pose(s), stone ids "
                f"{[stone_id for stone_id, _ in scene_prior_entries]}"
            )
        else:
            self._log("Scene pose init from state.pkl is empty.")
        for _ in range(10):
            self.log_dir_pub.publish(step_log_dir)
            self.integrated_planner.field_pose_init_pub.publish(
                target_pos,
                target_quat,
                -1,
            )
            if scene_prior_entries:
                self._publish_scene_pose_prior(scene_prior_entries)
            self.phase_node_pub.publish(self.Phase.SCENESCAN)

        self._on_display_geometries(
            self._base_scene(self.Q_SCAN) + [self.origin_frame],
            f"Step {n_step + 1}: scene scan",
        )
        scan_done = self._wait_for_scene_scan_done(n_step, step_log_dir)
        pcd_ready = self._finish_scene_pcd_transfer()
        if self._abort.is_set():
            return
        if not pcd_ready:
            self._log(
                "Scene scan PCD transfer failed; continuing with existing "
                "planned scene poses."
            )
            return
        self._show_scene_scan_pcd_preview(n_step)
        self._set_status(phase="Scene scan review")
        if not scan_done:
            self._log(
                "Scene scan-done flag timed out; review the transferred frames "
                "received so far before running desktop SceneID."
            )
        attempt = 1
        while not self._abort.is_set():
            decision = self._wait_for_decision(
                "scene_icp_request",
                {"step": n_step + 1, "scan_done": scan_done, "attempt": attempt},
            )
            if decision == "skip":
                self._log("Scene pose ICP skipped by operator.")
                return

            manual_init = decision == "retry"
            result = self._run_desktop_scene_pose_identification(
                n_step,
                step_log_dir,
                scene_prior_entries,
                manual_init=manual_init,
                attempt=attempt,
            )
            if result == "accepted":
                return
            if result == "retry":
                attempt += 1
                self._log("Retrying scene pose ICP with a fresh initialization choice.")
                self._show_scene_scan_pcd_preview(n_step)
                self._set_status(phase="Scene scan review")
                continue
            if result == "keep_current":
                return

            self._log("Scene pose ICP attempt failed; keeping current scene poses.")
            return

    def _wait_for_scene_scan_done(self, n_step, step_log_dir):
        timeout = float(
            self.planning_params.get(
                "scene_scan_done_wait_timeout", SCENE_SCAN_DONE_WAIT_TIMEOUT
            )
        )
        self._set_status(phase="Scene scan")
        self._log(
            "Waiting for scene scan completion "
            f"(/scene_scan_done, timeout={timeout:.1f}s)..."
        )
        deadline = time.time() + timeout
        while time.time() < deadline and not self._abort.is_set():
            with self._ros_spin_lock:
                self.rclpy.spin_once(self.scene_pcd_sub, timeout_sec=0.05)
                self.rclpy.spin_once(self.scene_scan_done_sub, timeout_sec=0.0)
            self._collect_scene_pcd_frame()
            if not self.scene_scan_done_sub.get_flag():
                continue
            done_log_dir = self.scene_scan_done_sub.step_log_dir
            self.scene_scan_done_sub.reset_get_flag()
            if equivalent_log_paths(done_log_dir, step_log_dir):
                self._drain_scene_pcd_after_done()
                self._log(f"Scene scan PCD ready for step {n_step + 1}.")
                return True
            self._log(
                "Ignoring scene scan completion for another step: " f"{done_log_dir}"
            )
        return False

    def _scene_pose_reference_poses(
        self, initial_poses: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        references = {}
        for stone_id, pose in initial_poses.items():
            reference = self._scene_config_pose_array(int(stone_id))
            if reference is None:
                reference = np.asarray(pose, dtype=np.float64).copy()
            references[int(stone_id)] = reference
        return references

    def _correct_desktop_sceneid_target_offset(
        self,
        scene_dir: Path,
        output_prefix: str,
        poses_by_stone: dict[int, np.ndarray],
        initial_poses: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        reference_poses = self._scene_pose_reference_poses(initial_poses)
        corrected, frame_shift = self.correct_target_offset_frame_if_needed(
            poses_by_stone,
            reference_poses,
            self.target_structure_offset,
        )
        if frame_shift is None:
            return poses_by_stone

        self._log(
            "Corrected desktop SceneID output by target_structure_offset "
            f"{frame_shift.tolist()} before preview/apply."
        )
        self._rewrite_sceneid_result_after_frame_correction(
            scene_dir,
            output_prefix,
            corrected,
            frame_shift,
        )
        return corrected

    def _rewrite_sceneid_result_after_frame_correction(
        self,
        scene_dir: Path,
        output_prefix: str,
        poses_by_stone: dict[int, np.ndarray],
        frame_shift: np.ndarray,
    ) -> None:
        pkl_path = scene_dir / f"{output_prefix}.pkl"
        txt_path = scene_dir / f"{output_prefix}.txt"
        try:
            data = {}
            if pkl_path.exists():
                with pkl_path.open("rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    data = loaded
            data["optimal_poses"] = {
                int(stone_id): np.asarray(pose, dtype=np.float64).copy()
                for stone_id, pose in poses_by_stone.items()
            }
            data["stone_ids"] = sorted(data["optimal_poses"])
            data["target_offset_frame_correction"] = {
                "source": "desktop_execution_review_guard",
                "xy_shift": np.asarray(frame_shift, dtype=np.float64).copy(),
            }
            with pkl_path.open("wb") as f:
                pickle.dump(data, f)
            initial_poses = data.get("initial_poses", {}) or {}
            initial_sources = data.get("initial_pose_sources", {}) or {}
            with txt_path.open("w") as f:
                f.write(f"plan_dir: {data.get('plan_dir')}\n")
                f.write(f"action_sequence: {data.get('action_sequence')}\n")
                f.write(f"step_index: {data.get('step_index')}\n")
                f.write(f"scene_pcd_path: {data.get('scene_pcd_path')}\n")
                f.write(f"ground_height: {data.get('ground_height')}\n")
                ground_plane_model = data.get("ground_plane_model")
                if ground_plane_model is not None:
                    ground_plane_model = np.asarray(
                        ground_plane_model, dtype=np.float64
                    ).tolist()
                f.write(f"ground_plane_model: {ground_plane_model}\n")
                f.write(f"skipped_sceneid_solve: {data.get('skipped_sceneid_solve')}\n")
                stone_point_counts = data.get("sceneid_stone_point_counts") or {}
                if stone_point_counts:
                    f.write("sceneid_stone_point_counts:\n")
                    for stone_id in sorted(stone_point_counts, key=int):
                        item = stone_point_counts[stone_id]
                        f.write(
                            f"  stone {stone_id}: points={item.get('points')} "
                            f"min_points={item.get('min_points')} "
                            f"margin={item.get('margin')} "
                            f"source={item.get('source')} "
                            f"kept={item.get('kept')}\n"
                        )
                skipped_low_points = data.get("skipped_low_scene_point_stones") or {}
                if skipped_low_points:
                    f.write("skipped_low_scene_point_stones:\n")
                    for stone_id in sorted(skipped_low_points, key=int):
                        item = skipped_low_points[stone_id]
                        f.write(
                            f"  stone {stone_id}: points={item.get('points')} "
                            f"min_points={item.get('min_points')} "
                            f"margin={item.get('margin')} "
                            f"source={item.get('source')}\n"
                        )
                elapsed = data.get("elapsed")
                if elapsed is None:
                    f.write("elapsed: None\n")
                else:
                    f.write(f"elapsed: {float(elapsed):.6f}\n")
                f.write(
                    "target_offset_frame_correction: "
                    f"{np.asarray(frame_shift, dtype=np.float64).tolist()}\n"
                )
                for stone_id in sorted(poses_by_stone):
                    pose = np.asarray(poses_by_stone[stone_id], dtype=np.float64).copy()
                    init_pose = initial_poses.get(stone_id)
                    if init_pose is None:
                        init_pose = initial_poses.get(str(stone_id))
                    if init_pose is not None:
                        init_pose = np.asarray(init_pose, dtype=np.float64).tolist()
                    init_source = initial_sources.get(stone_id)
                    if init_source is None:
                        init_source = initial_sources.get(str(stone_id), "")
                    f.write(
                        f"stone {stone_id}: init_source={init_source} "
                        f"init={init_pose} optimal={pose.tolist()}\n"
                    )
        except Exception as exc:
            self._log(f"Failed to rewrite corrected SceneID logs: {exc}")

    def _run_desktop_scene_pose_identification(
        self,
        n_step: int,
        step_log_dir: str,
        scene_prior_entries: list[tuple[int, np.ndarray]],
        manual_init: bool = False,
        attempt: int = 1,
    ) -> str:
        initial_poses = {
            int(stone_id): np.asarray(pose, dtype=np.float64).copy()
            for stone_id, pose in scene_prior_entries
        }
        if not initial_poses:
            self._log("Desktop SceneID skipped: no initial scene poses available.")
            return "failed"

        scene_dir = Path(step_log_dir) / "scene_scan"
        args = self.make_sceneid_runtime_args()
        args.asset_dir = str(self.asset_dir)
        args.target_structure_offset = self.target_structure_offset.copy()
        args.manual_init_gui = bool(manual_init)
        args.manual_fix_gui = bool(manual_init)
        args.manual_init_fixed = False
        args.fix_initial_steps = []
        args.manual_init_gui_output = None
        if manual_init:
            args.manual_init_gui_output = (
                scene_dir / f"manual_init_live_attempt{attempt}.json"
            )
            self._log(
                "SceneID manual initialization/fixed-stone GUI requested."
            )
        for key in ("sceneid_min_stone_points", "sceneid_stone_point_margin"):
            if key in self.planning_params:
                setattr(args, key, self.planning_params[key])

        self._set_status(phase="Desktop scene pose identification")
        mode = "manual initialization" if manual_init else "automatic initialization"
        self._log(
            "Running scene pose identification on desktop "
            f"(attempt {attempt}, {mode}) from {scene_dir}."
        )
        try:
            ok, poses, ground_height, ground_plane_model = (
                self.identify_scene_dir_from_initial_poses(
                    scene_dir,
                    args,
                    initial_poses,
                    self._sceneid_prior_poses,
                    self._sceneid_ground_height,
                    self._sceneid_ground_plane,
                    asset_dir=self.asset_dir,
                    action_sequence=self.action_sequence[: n_step + 1],
                )
            )
        except Exception as exc:
            self._log(f"Desktop SceneID failed: {exc}")
            try:
                scene_dir.mkdir(parents=True, exist_ok=True)
                with (scene_dir / "sceneid_failure.txt").open("w") as f:
                    f.write(str(exc) + "\n")
            except Exception:
                pass
            return "failed"

        if not ok:
            self._log("Desktop SceneID returned no valid result.")
            return "failed"

        poses = self._correct_desktop_sceneid_target_offset(
            scene_dir,
            args.output_prefix,
            poses,
            initial_poses,
        )

        metadata = self._load_scene_pose_result_metadata(step_log_dir)
        self._show_identified_scene_pose_preview(poses, n_step, metadata)
        decision = self._wait_for_decision(
            "scene_pose_review",
            {
                "step": n_step + 1,
                "num_poses": len(poses),
                "ground_height": metadata.get("ground_height"),
            },
        )
        if decision == "continue":
            if self._sceneid_ground_height is None:
                self._sceneid_ground_height = ground_height
                self._sceneid_ground_plane = ground_plane_model
            self._sceneid_prior_poses.update(
                {stone_id: pose.copy() for stone_id, pose in poses.items()}
            )
            self._apply_identified_scene_poses(poses, n_step)
            return "accepted"
        if decision == "retry":
            self._log("Scene pose ICP result rejected; retry requested.")
            return "retry"
        self._log(
            "Rejected identified scene poses; continuing with existing "
            "planned scene poses."
        )
        return "keep_current"

    def _load_scene_pose_result_metadata(self, step_log_dir):
        result_path = os.path.join(
            step_log_dir, "scene_scan", "sceneid_scene_poses.pkl"
        )
        if not os.path.exists(result_path):
            self._log(f"Scene pose result metadata not found: {result_path}")
            return {}
        try:
            with open(result_path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            self._log(f"Failed to load scene pose result metadata: {exc}")
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _show_identified_scene_pose_preview(self, poses_by_stone, n_step, metadata):
        placed_ids = {
            int(action["stone_id"]) for action in self.action_sequence[: n_step + 1]
        }
        preview_entries = []
        deltas = []
        pose_count = 0
        for stone_id, pose in sorted(poses_by_stone.items()):
            if stone_id not in placed_ids:
                continue
            pose_T = self._pose_array_to_matrix(pose)
            if pose_T is None:
                continue
            pose_count += 1
            current_config = self.scene_configs.get(stone_id)
            if current_config is not None:
                current_T = np.asarray(
                    current_config.pose.as_matrix(), dtype=np.float64
                )
                current_geom = self._copy_base_stone_mesh(stone_id)
                current_geom.transform(current_T)
                preview_entries.append((current_geom, C_SCENEID_PRE, "wireframe"))

                delta_line = self._trajectory_lineset(
                    [current_T[:3, 3], pose_T[:3, 3]], C_SCENEID_DELTA
                )
                if delta_line is not None:
                    preview_entries.append(delta_line)

                translation, rotation, _ = self._pose_delta(current_T, pose_T)
                deltas.append((stone_id, translation, rotation))
            preview_entries.append((self._stone_geometry(stone_id, pose_T), C_SCENEID))

        plane_grid = self._scene_pose_ground_grid(poses_by_stone, metadata)
        if plane_grid is not None:
            preview_entries.append((plane_grid, C_SCENEID_GROUND))
        if self._scene_scan_display_entry is not None:
            preview_entries.append(self._scene_scan_display_entry)

        self._on_display_geometries(
            self._base_scene(self.Q_SCAN) + preview_entries + [self.origin_frame],
            f"Step {n_step + 1}: identified vs current scene poses",
        )

        ground_height = metadata.get("ground_height")
        if ground_height is None:
            self._log(f"Review identified scene poses: {pose_count} stone(s).")
        else:
            self._log(
                f"Review identified scene poses: {pose_count} stone(s). "
                "Estimated local placing ground "
                f"height: {float(ground_height):.4f} m"
            )
        for stone_id, translation, rotation in deltas:
            self._log(
                f"Scene pose delta stone {stone_id}: "
                f"{translation:.3f} m, {rotation:.1f} deg"
            )

    def _scene_pose_ground_grid(self, poses_by_stone, metadata):
        ground_height = metadata.get("ground_height")
        plane_model = metadata.get("ground_plane_model")
        if plane_model is None and ground_height is None:
            return None

        if plane_model is None:
            plane_model = np.array([0.0, 0.0, 1.0, -float(ground_height)])
        plane_model = np.asarray(plane_model, dtype=np.float64).reshape(-1)
        if plane_model.shape[0] != 4 or not np.all(np.isfinite(plane_model)):
            return None

        normal = plane_model[:3]
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return None
        normal = normal / norm
        d = float(plane_model[3]) / norm

        pose_points = [
            np.asarray(pose[:3], dtype=np.float64)
            for pose in poses_by_stone.values()
            if np.asarray(pose).shape[0] >= 3
        ]
        if pose_points:
            center_xy = np.mean(np.stack(pose_points, axis=0)[:, :2], axis=0)
        else:
            center_xy = np.asarray(self.target_structure_offset[:2], dtype=np.float64)
        center = np.array([center_xy[0], center_xy[1], 0.0], dtype=np.float64)
        center = center - normal * (float(np.dot(normal, center)) + d)

        axis_u = np.cross(normal, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(axis_u) < 1e-6:
            axis_u = np.array([1.0, 0.0, 0.0])
        axis_u = axis_u / np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        axis_v = axis_v / np.linalg.norm(axis_v)

        half_extent = 2.0
        n_lines = 8
        coords = np.linspace(-half_extent, half_extent, n_lines + 1)
        points = []
        lines = []
        for c in coords:
            base_idx = len(points)
            points.append(center - half_extent * axis_u + c * axis_v)
            points.append(center + half_extent * axis_u + c * axis_v)
            lines.append([base_idx, base_idx + 1])

            base_idx = len(points)
            points.append(center + c * axis_u - half_extent * axis_v)
            points.append(center + c * axis_u + half_extent * axis_v)
            lines.append([base_idx, base_idx + 1])

        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(np.asarray(points))
        grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        grid.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray(C_SCENEID_GROUND, dtype=np.float64), (len(lines), 1))
        )
        return grid

    def _apply_identified_scene_poses(
        self, poses_by_stone: dict[int, np.ndarray], n_step: int
    ):
        if not poses_by_stone:
            self._log("Received empty scene pose identification result.")
            return

        placed_ids = {
            int(action["stone_id"]) for action in self.action_sequence[: n_step + 1]
        }
        applied = []
        for stone_id, pose in sorted(poses_by_stone.items()):
            if stone_id not in placed_ids:
                continue
            config = copy.deepcopy(self.stone_configs[stone_id])
            config.pose.setPosition(pose[:3])
            config.pose.setOrientation(pose[3:7])

            old_body_id = self.place_body_ids_by_stone.get(stone_id)
            if old_body_id is not None:
                self.context.remove_body(old_body_id)
            self.place_body_ids_by_stone[stone_id] = self.context.add_place_body(config)
            self._set_scene_stone_pose(stone_id, config)
            self.resume_scene_poses[stone_id] = pose.copy()
            applied.append(stone_id)

        if applied:
            self._log(
                "Applied identified scene poses to planner context for stones: "
                f"{applied}"
            )
            self._sync_identified_scene_poses_to_planning_state(
                poses_by_stone,
                applied,
            )
        else:
            self._log(
                "Received scene pose identification result, but it contained no "
                "stones placed by the current step."
            )

    def _sync_identified_scene_poses_to_planning_state(
        self,
        poses_by_stone: dict[int, np.ndarray],
        applied: list[int],
    ) -> None:
        planner = getattr(self, "integrated_planner", None)
        if planner is None:
            return
        try:
            state = planner.env.get_state()
        except Exception as exc:
            self._log(f"Could not read planner state after SceneID: {exc}")
            return

        for stone_id in applied:
            pose = np.asarray(poses_by_stone[int(stone_id)], dtype=np.float64).copy()
            pose[:2] -= self.target_structure_offset[:2]
            state.stone_poses[int(stone_id)] = pose

        pose_identified_ids = set(
            int(stone_id)
            for stone_id in getattr(state, "pose_identified_stone_ids", set()) or set()
        )
        pose_identified_ids.update(int(stone_id) for stone_id in applied)
        state.pose_identified_stone_ids = pose_identified_ids
        try:
            planner.env.update_from_state(state)
        except Exception as exc:
            self._log(f"Could not update planner state after SceneID: {exc}")
            return

        if not self.online_mode:
            return

        if self._online_debug_data is not None:
            self._online_debug_data["resume_state"] = copy.deepcopy(state)
            self._online_debug_data["resume_step"] = len(self.action_sequence)
            self.sequence_runtime._save_debug_data(
                self._online_debug_data,
                os.path.join(self.plan_dir, "state.pkl"),
            )

        self.planning_params["resume_scene_poses"] = {
            int(stone_id): np.asarray(pose, dtype=np.float64).copy()
            for stone_id, pose in state.stone_poses.items()
            if int(stone_id) in self._online_placed_stone_ids()
        }
        with open(os.path.join(self.plan_dir, "planning_params.pkl"), "wb") as f:
            pickle.dump(self.planning_params, f)
        self._log(
            "Updated online MCTS planner state from accepted desktop SceneID poses."
        )
