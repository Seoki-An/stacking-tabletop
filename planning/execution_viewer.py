"""Viewer scene construction, live updates, geometry helpers, and pose utilities."""

from .execution_common import *


class ExecutionViewerMixin:
    def _publish_phase(self, phase, count):
        for _ in range(count):
            self.phase_node_pub.publish(phase)

    def _publish_log_and_phase(self, log_dir, phase, count):
        for _ in range(count):
            self.log_dir_pub.publish(log_dir)
            self.phase_node_pub.publish(phase)

    # --- scene / viewer helpers -------------------------------------------
    def _reset_viewer_state(self):
        self._viz_target_id = None
        self._viz_target_mode = None
        self._viz_inhand_T = None
        self._viz_inhand_T_init = None
        self._viz_pick_pose_T = None
        self._viz_place_pose_T = None
        self._viz_opening_angle = 0.0
        self._viz_overlay_entries = []
        self._last_live_q = None
        self._last_live_layout_key = None

    def _remove_overlay_entries_by_name(self, *names) -> bool:
        names = {name for name in names if name}
        if not names:
            return False
        kept = []
        removed = False
        for item in self._viz_overlay_entries:
            meta = item[2] if isinstance(item, tuple) and len(item) >= 3 else None
            if isinstance(meta, dict) and meta.get("name") in names:
                removed = True
                continue
            kept.append(item)
        if removed:
            self._viz_overlay_entries = kept
            self._last_live_layout_key = None
        return removed

    def _base_scene(self, q, opening_angle=None):
        return (
            [(self.ground_mesh, C_GROUND)]
            + self._excavator_at(q, opening_angle=opening_angle)
            + self._lidar_frame_entries(q, opening_angle=opening_angle)
            + self._other_stone_entries()
            + self._target_stone_entries()
            + self._ghost_entries()
            + self._target_structure_entries()
        )

    def _excavator_at(self, q, opening_angle=None):
        q8 = self._q_with_opening(q, opening_angle=opening_angle)
        transforms = self._excavator_link_transforms(q8)
        return [
            self._transform_entry(
                self.excavator_meshes[name],
                C_EXCAV,
                self._excavator_live_name(name),
                transforms[name],
            )
            for name in self.excavator_meshes.keys()
            if name in transforms
        ]

    def _q_with_opening(self, q, opening_angle=None):
        if opening_angle is None:
            opening_angle = self._viz_opening_angle
        q = np.asarray(q, dtype=np.float64)
        if q.shape[0] == 6:
            q = np.concatenate([q, [opening_angle, opening_angle]])
        return q

    def _excavator_link_transforms(self, q8):
        q8 = self._q_with_opening(q8)
        self.excavator_model.SetState(q8)
        transforms = {}
        for name in self.excavator_meshes.keys():
            try:
                transforms[name] = self._link_transform(name)
            except Exception:
                continue
        return transforms

    def _link_transform(self, name):
        geom = self.excavator_model.GetLink(name).geoms[0]
        T = np.eye(4)
        T[:3, :3] = geom.GetRotation()
        T[:3, 3] = geom.GetPosition()
        return T

    def _link_frame_transform(self, name):
        link = self.excavator_model.GetLink(name)
        T = np.eye(4)
        T[:3, :3] = link.GetRotation()
        T[:3, 3] = link.GetPosition()
        return T

    def _lidar_frame_name(self, name):
        return f"live_lidar_frame_{name}"

    def _make_lidar_frame_geometry(self, size=0.35):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [size, 0.0, 0.0],
                [0.0, size, 0.0],
                [0.0, 0.0, size],
            ],
            dtype=np.float64,
        )
        lines = np.asarray([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
        colors = np.asarray(
            [[1.0, 0.05, 0.05], [0.05, 0.9, 0.05], [0.1, 0.35, 1.0]],
            dtype=np.float64,
        )
        frame = o3d.geometry.LineSet()
        frame.points = o3d.utility.Vector3dVector(points)
        frame.lines = o3d.utility.Vector2iVector(lines)
        frame.colors = o3d.utility.Vector3dVector(colors)
        return frame

    def _copy_lidar_frame_geometry(self):
        return o3d.geometry.LineSet(self.lidar_frame_geometry)

    def _lidar_frame_transforms(self, q8):
        if not self._lidar_frame_mode:
            return {}
        q8 = self._q_with_opening(q8)
        self.excavator_model.SetState(q8)
        transforms = {}
        for name in LIDAR_FRAME_LINK_NAMES:
            try:
                transforms[name] = self._link_frame_transform(name)
            except Exception:
                continue
        return transforms

    def _lidar_frame_entries(self, q, opening_angle=None):
        if not self._lidar_frame_mode:
            return []
        q8 = self._q_with_opening(q, opening_angle=opening_angle)
        return [
            (
                self._copy_lidar_frame_geometry(),
                [1.0, 1.0, 1.0],
                {"name": self._lidar_frame_name(name), "transform": transform},
            )
            for name, transform in self._lidar_frame_transforms(q8).items()
        ]

    def _excavator_live_name(self, name):
        return f"live_excavator_{name}"

    def _live_target_name(self, suffix="current"):
        return f"live_target_stone_{suffix}"

    def _transform_entry(self, geom, color, name, transform):
        return (geom, color, {"name": name, "transform": np.asarray(transform)})

    def _target_stone_entries(self):
        entries = []
        tid = self._viz_target_id
        if tid is None or tid not in self.scene_meshes:
            return entries

        if self._viz_target_mode == "pick_pose" and self._viz_pick_pose_T is not None:
            entries.append((self._stone_geometry(tid, self._viz_pick_pose_T), C_TARGET))
        elif self._viz_target_mode == "place_pose":
            if self._viz_place_pose_T is not None:
                entries.append(
                    (self._stone_geometry(tid, self._viz_place_pose_T), C_TARGET)
                )
            else:
                entries.append((self._scene_stone_geometry(tid), C_TARGET))
        elif self._viz_target_mode is None:
            entries.append((self._scene_stone_geometry(tid), C_TARGET))
        return entries

    def _other_stone_entries(self):
        return [
            (self._scene_stone_geometry(sid), C_STONE)
            for sid in self.scene_meshes.keys()
            if sid != self._viz_target_id
        ]

    def _ghost_entries(self):
        tid = self._viz_target_id
        if (
            tid is None
            or self._viz_place_pose_T is None
            or self._viz_target_mode == "place_pose"
        ):
            return []
        ghost = self._stone_geometry(tid, self._viz_place_pose_T)
        return [(ghost, C_GHOST, "wireframe")]

    def _target_structure_entries(self):
        entries = self._place_height_box_entries()
        for sid, config in self.target_structure_configs:
            mesh = self._copy_base_stone_mesh(sid)
            mesh = mesh.simplify_vertex_clustering(TARGET_STRUCTURE_VOXEL)
            mesh.transform(config.pose.as_matrix())
            entries.append((mesh, C_STRUCTURE, {"alpha": TARGET_STRUCTURE_ALPHA}))
        return entries

    def _place_height_box_entries(self):
        if self.place_height_box_mesh is None:
            return []
        return [(self.place_height_box_mesh, C_GROUND)]

    def _make_place_height_box_mesh(self):
        wall = self.integrated_planner.env.inventory.target_wall
        points = []
        for geom in wall.geometries:
            if hasattr(geom, "points"):
                pts = np.asarray(geom.points, dtype=np.float64)
            else:
                pts = np.asarray(geom.get_mesh().vertices, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] >= 3 and pts.size > 0:
                points.append(pts[:, :3])
        if not points:
            return None

        pts = np.concatenate(points, axis=0)
        if not np.all(np.isfinite(pts)):
            return None

        xy_offset = np.asarray(self.target_structure_offset[:2], dtype=np.float64)
        xy_min = pts[:, :2].min(axis=0) + xy_offset
        xy_max = pts[:, :2].max(axis=0) + xy_offset
        xy_min -= TARGET_STRUCTURE_PLACE_HEIGHT_BOX_MARGIN
        xy_max += TARGET_STRUCTURE_PLACE_HEIGHT_BOX_MARGIN
        extent_xy = xy_max - xy_min
        if np.any(extent_xy <= 0.0):
            return None

        z_top = float(self.place_plane_height)
        thickness = max(
            abs(float(self.place_plane_height) - float(self.pick_plane_height)),
            TARGET_STRUCTURE_PLACE_HEIGHT_BOX_MIN_THICKNESS,
        )
        z_bottom = z_top - thickness

        mesh = o3d.geometry.TriangleMesh.create_box(
            float(extent_xy[0]), float(extent_xy[1]), float(thickness)
        )
        mesh.translate((float(xy_min[0]), float(xy_min[1]), z_bottom))
        mesh.compute_vertex_normals()
        return mesh

    def _live_state_cb(self, q_actual):
        q8 = self._q_with_opening(q_actual)
        layout_key = self._live_layout_key()
        layout_changed = layout_key != self._last_live_layout_key
        if not layout_changed and self._should_skip_live_viewer_emit(q8):
            return

        if layout_changed:
            if self._live_layout_suppressed:
                if self._should_skip_live_viewer_emit(q8):
                    return
                self._on_live_joint_state(self._live_joint_state_payload(q8))
                return
            self._emit_live_scene_layout(q8)
            self._last_live_layout_key = layout_key
            self._record_live_viewer_emit(q8)

        self._on_live_joint_state(self._live_joint_state_payload(q8))

    def _live_layout_key(self):
        return (
            self._viz_target_id,
            self._viz_target_mode,
            self._viz_inhand_T is not None,
            self._has_inhand_initial_delta(),
            self._stone_pcd_mode,
            self._stone_highpoly_mesh_mode,
            self._lidar_frame_mode,
            id(self._viz_overlay_entries),
            len(self._viz_overlay_entries),
        )

    def _has_inhand_initial_delta(self):
        return (
            self._viz_inhand_T is not None
            and self._viz_inhand_T_init is not None
            and not np.allclose(self._viz_inhand_T_init, self._viz_inhand_T)
        )

    def _emit_live_scene_layout(self, q8):
        viz = (
            [(self.ground_mesh, C_GROUND)]
            + self._excavator_at(q8)
            + self._lidar_frame_entries(q8)
            + self._other_stone_entries()
            + self._live_target_stone_entries(q8)
            + self._ghost_entries()
            + self._target_structure_entries()
            + self._viz_overlay_entries
            + [self.origin_frame]
        )
        self._on_display_geometries(viz, "live layout")

    def _live_joint_state_payload(self, q8):
        transforms = {
            self._excavator_live_name(name): T
            for name, T in self._excavator_link_transforms(q8).items()
        }
        transforms.update(
            {
                self._lidar_frame_name(name): T
                for name, T in self._lidar_frame_transforms(q8).items()
            }
        )
        transforms.update(self._live_target_transforms(q8))
        return {"transforms": transforms}

    def _live_target_transforms(self, q8):
        transforms = {}
        tid = self._viz_target_id
        if (
            tid is None
            or tid not in self.scene_meshes
            or self._viz_target_mode != "in_hand"
            or self._viz_inhand_T is None
        ):
            return transforms

        grip_world = self._grip_body_world(q8)
        transforms[self._live_target_name()] = grip_world @ self._viz_inhand_T

        if self._has_inhand_initial_delta():
            transforms[self._live_target_name("initial")] = (
                grip_world @ self._viz_inhand_T_init
            )
        return transforms

    def _should_skip_live_viewer_emit(self, q8):
        now = time.time()
        if now - self._last_viewer_emit < self._viewer_min_interval:
            return True

        if self._last_live_q is not None:
            q_delta = np.linalg.norm(q8 - self._last_live_q)
            if q_delta < 1e-4 and now - self._last_viewer_emit < 1.0:
                return True

        self._record_live_viewer_emit(q8, now=now)
        return False

    def _record_live_viewer_emit(self, q8, now=None):
        self._last_viewer_emit = time.time() if now is None else now
        self._last_live_q = q8.copy()

    def _live_target_stone_entries(self, q8):
        tid = self._viz_target_id
        if tid is None or tid not in self.scene_meshes:
            return []

        if self._viz_target_mode == "in_hand" and self._viz_inhand_T is not None:
            grip_world = self._grip_body_world(q8)
            stone_world = grip_world @ self._viz_inhand_T
            entries = [
                self._dynamic_stone_entry(
                    tid, self._live_target_name(), stone_world, C_TARGET
                )
            ]

            if self._has_inhand_initial_delta():
                stone_world_init = grip_world @ self._viz_inhand_T_init
                entries.append(
                    self._dynamic_stone_entry(
                        tid,
                        self._live_target_name("initial"),
                        stone_world_init,
                        C_INHAND_INIT,
                    )
                )
            return entries

        if self._viz_target_mode == "pick_pose" and self._viz_pick_pose_T is not None:
            return [(self._stone_geometry(tid, self._viz_pick_pose_T), C_TARGET)]

        if self._viz_target_mode == "place_pose" and self._viz_place_pose_T is not None:
            return [(self._stone_geometry(tid, self._viz_place_pose_T), C_TARGET)]

        return [(self._scene_stone_geometry(tid), C_TARGET)]

    def _dynamic_stone_entry(self, target_id, name, pose_T, color):
        if self._stone_pcd_mode and target_id in self.stone_pcds:
            geom = copy.deepcopy(self.stone_pcds[target_id])
        else:
            geom = self._copy_base_stone_mesh(target_id)
        return self._transform_entry(
            geom,
            color,
            name,
            pose_T,
        )

    def _grip_body_world(self, q):
        q8 = self._q_with_opening(q)
        self.excavator_model.SetState(q8)
        geom = self.excavator_model.GetLink("grip_body").geoms[0]
        pose = np.eye(4)
        pose[:3, :3] = geom.GetRotation()
        pose[:3, 3] = geom.GetPosition()
        return pose

    def _trajectory_lineset(self, points, color):
        points = [
            p
            for p in (np.asarray(p, dtype=np.float64) for p in points)
            if p.shape == (3,) and np.all(np.isfinite(p))
        ]
        deduped = []
        for point in points:
            if deduped and np.linalg.norm(point - deduped[-1]) <= 1e-6:
                continue
            deduped.append(point)
        points = deduped
        if len(points) < 2:
            return None
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.asarray(points))
        lines = np.asarray([[i, i + 1] for i in range(len(points) - 1)], dtype=np.int32)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        colors = np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
        lineset.colors = o3d.utility.Vector3dVector(colors)
        return lineset

    def _target_trajectory_markers(self, target_path, color, max_markers=80):
        if len(target_path) == 0:
            return []
        stride = max(1, int(np.ceil(len(target_path) / max_markers)))
        entries = []
        for i, T in enumerate(target_path):
            if i % stride != 0 and i != len(target_path) - 1:
                continue
            T = np.asarray(T, dtype=np.float64)
            if T.shape != (4, 4) or not np.all(np.isfinite(T)):
                continue
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.06, resolution=8)
            marker.translate(T[:3, 3])
            entries.append((marker, color))
        return entries

    def _place_target_path_matrices(self, result):
        n_path = len(result.target_path_sequence)
        if n_path == 2:
            start_idx = 0
        elif n_path == 4:
            start_idx = 2
        elif n_path == 6:
            start_idx = 4
        elif n_path == 8:
            start_idx = 6
        elif n_path == 10:
            start_idx = 8
        else:
            start_idx = 0

        target_path = []
        for path_sub in result.target_path_sequence[start_idx : start_idx + 2]:
            for pose_t in path_sub:
                T = np.asarray(pose_t.as_matrix(), dtype=np.float64)
                if T.shape == (4, 4) and np.all(np.isfinite(T)):
                    target_path.append(T)
        return target_path

    def _stone_geometry(self, target_id, pose_T):
        if self._stone_pcd_mode and target_id in self.stone_pcds:
            geom = copy.deepcopy(self.stone_pcds[target_id])
        else:
            geom = self._copy_base_stone_mesh(target_id)
        geom.transform(pose_T)
        return geom

    def _save_pose_data_snapshot(self, target_id, pos, quat, reason: str):
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        self.poses[int(target_id)] = (pos.copy(), quat.copy())

        pose_data = {}
        for stone_id, pose in self.poses.items():
            if isinstance(pose, (tuple, list)) and len(pose) >= 2:
                pose_data[int(stone_id)] = (
                    np.asarray(pose[0], dtype=np.float64).copy(),
                    np.asarray(pose[1], dtype=np.float64).copy(),
                )
            else:
                arr = np.asarray(pose, dtype=np.float64).reshape(-1)
                if arr.shape[0] >= 7:
                    pose_data[int(stone_id)] = (arr[:3].copy(), arr[3:7].copy())

        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = os.path.join(
            self.plan_dir,
            f"pose_data_recovered_{timestamp}_stone_{int(target_id)}.pkl",
        )
        with open(save_path, "wb") as f:
            pickle.dump(pose_data, f)

        self._log("Saved recovered field pose data " f"({reason}) to {save_path}")

    def _copy_base_stone_mesh(self, target_id):
        if self._stone_highpoly_mesh_mode and target_id in self.stone_meshes:
            return copy.deepcopy(self.stone_meshes[target_id])
        return copy.deepcopy(self.stone_dsf_meshes[target_id])

    def _scene_stone_geometry(self, target_id):
        if target_id in self.scene_configs:
            return self._stone_geometry(
                target_id, self.scene_configs[target_id].pose.as_matrix()
            )
        return self.scene_meshes[target_id]

    def _set_scene_stone_pose(self, target_id, config):
        mesh = self._copy_base_stone_mesh(target_id)
        mesh.transform(config.pose.as_matrix())
        self.scene_meshes[target_id] = mesh
        self.scene_configs[target_id] = copy.deepcopy(config)

    def _update_scene_mesh_from_config(self, target_id):
        self._set_scene_stone_pose(target_id, self.stone_configs[target_id])

    # --- math / config helpers --------------------------------------------
    def _needs_inhand_replan(self, inhand_T_opt, inhand_T):
        translation_error, rotation_error, _ = self._pose_delta(inhand_T_opt, inhand_T)
        return (
            translation_error > self.inhand_replan_translation_threshold
            or rotation_error > self.inhand_replan_rotation_threshold_deg
        )

    def _log_inhand_delta(self, inhand_T_opt, inhand_T):
        inhand_T_rel = np.linalg.inv(inhand_T_opt) @ inhand_T
        translation_error, rotation_error, _ = self._pose_delta(inhand_T_opt, inhand_T)
        self._log(f"Relative transform (optimized vs original):\n{inhand_T_rel}")
        self._log(
            "In-hand pose delta: "
            f"{translation_error:.3f} m, {rotation_error:.1f} deg"
        )

    def _pose_delta(self, a_T, b_T):
        delta_T = np.linalg.inv(a_T) @ b_T
        translation = float(np.linalg.norm(delta_T[:3, 3]))
        rotation = self._rotation_angle_deg(delta_T[:3, :3])
        return translation, rotation, delta_T

    def _rotation_angle_deg(self, rotation):
        cos_theta = (np.trace(rotation) - 1.0) * 0.5
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _make_intermediate_config(self, base_config):
        intermediate_config = copy.deepcopy(base_config)
        pos = np.array(intermediate_config.pose.position(), dtype=np.float64, copy=True)
        pos[:2] = np.asarray(self.regrasp_xy_pos, dtype=np.float64)
        pos[2] = 1.0
        intermediate_config.pose.setPosition(pos)
        self.posegen.config().obj.k_potential = 0.2
        self.posegen.config().obj.k_gap_c = 20
        self.posegen.config().obj.k_xy = 0.0
        self.posegen.config().obj.k_reg = 0.0
        posegen_result = self.posegen.solve(intermediate_config)
        intermediate_config.pose = posegen_result.optimal_pose

        self.posegen.config().obj.k_potential = 0.2
        self.posegen.config().obj.k_gap_c = 100
        self.posegen.config().obj.k_xy = 0.0
        self.posegen.config().obj.k_reg = 0.0
        posegen_result = self.posegen.solve(intermediate_config)
        intermediate_config.pose = posegen_result.optimal_pose
        pos[2] = intermediate_config.pose.position()[2] + Z_OFFSET_INTERMEDIATE
        intermediate_config.pose.setPosition(pos)
        return intermediate_config
