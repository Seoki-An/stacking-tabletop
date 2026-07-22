"""Manual place control and step completion dispatch."""

from .execution_common import *


class ExecutionManualMixin:
    def _planning_result_is_valid(self, result):
        return (
            result is not None
            and result.is_feasible
            and len(result.q_path_sequence) > 0
        )

    def _run_place_control(
        self,
        result,
        opening_angle_pick,
        n_step=None,
        target_id=None,
        place_body_id=None,
        place_config=None,
    ):
        return self._run_remaining_grasp_control(
            result,
            opening_angle_pick,
            "Place control",
            n_step=n_step,
            target_id=target_id,
            place_body_id=place_body_id,
            place_config=place_config,
        )

    def _run_manual_place_control(self, target_id, place_body_id, opening_angle):
        if not ROS_CONTROL_ON:
            return True

        inhand_T = self._manual_place_inhand_T
        if inhand_T is None:
            inhand_T = self._viz_inhand_T
        if inhand_T is None:
            self._log(
                "Manual place requested, but no identified in-hand pose is available."
            )
            return False

        self._set_status(phase="Manual place")
        self._viz_target_id = target_id
        self._viz_target_mode = "in_hand"
        self._viz_inhand_T = np.asarray(inhand_T, dtype=np.float64).copy()
        self._viz_inhand_T_init = None
        self._viz_opening_angle = opening_angle
        self._viz_overlay_entries = []

        with self._pause_live_joint_polling():
            q_place = self._manual_joint_prompt(
                "manual_place",
                self._manual_control_start_q(),
                "close",
            )
            if isinstance(q_place, str):
                return False
            self._publish_manual_joint(q_place, "close", MANUAL_PLACE_PUBLISH_COUNT)

            q_release = self._manual_joint_prompt(
                "manual_release",
                q_place,
                "close",
            )
            if isinstance(q_release, str):
                return False
            self._publish_manual_joint(
                q_release, "open", MANUAL_PLACE_OPEN_PUBLISH_COUNT
            )

            release_T = (
                self._grip_body_world(self._q_with_opening(q_release, opening_angle))
                @ self._viz_inhand_T
            )
            self._viz_place_pose_T = release_T
            self._viz_target_mode = "place_pose"
            self._viz_opening_angle = 0.0
            self._update_manual_place_context(target_id, place_body_id, release_T)

            q_retreat = self._manual_joint_prompt(
                "manual_retreat",
                q_release,
                "open",
            )
            if isinstance(q_retreat, str):
                return False
            self.position_control(
                q_retreat,
                np.zeros_like(q_retreat),
                self.joint_node_pub,
                self.joint_node_sub,
                error_tol=0.1,
                time_limit=10.0,
                grab="open",
                state_cb=self._live_joint_state_cb(),
                spin_lock=self._ros_spin_lock,
            )
            self.q_joint = np.asarray(q_retreat, dtype=np.float64).copy()

        self._log("Manual place control finished.")
        return True

    def _manual_joint_prompt(self, phase, q, grab):
        return self._confirm_adjustable_grasp(phase, q, grab)

    def _manual_control_start_q(self):
        candidates = [
            self._last_live_q,
            (
                getattr(self.joint_node_sub, "pos", None)
                if hasattr(self, "joint_node_sub")
                else None
            ),
            getattr(self, "q_joint", None),
            self.q_home,
        ]
        for q in candidates:
            if q is None:
                continue
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            if q.size >= 6 and np.all(np.isfinite(q[:6])):
                return q[:6].copy()
        return np.asarray(self.q_home, dtype=np.float64).copy()

    def _publish_manual_joint(self, q, grab, count):
        q = np.asarray(q, dtype=np.float64).reshape(-1)[:6].copy()
        v = np.zeros_like(q)
        for _ in range(int(count)):
            self.joint_node_pub.publish(q, v, grab)
        if LIVE_JOINT_VIEWER_ON:
            self._live_state_cb(q)

    def _update_manual_place_context(self, target_id, place_body_id, release_T):
        config = copy.deepcopy(self.stone_configs[target_id])
        config.pose = self._pose_from_matrix(release_T)

        old_body_id = self.place_body_ids_by_stone.get(int(target_id), place_body_id)
        if old_body_id is not None:
            self.context.remove_body(old_body_id)
        self.place_body_ids_by_stone[int(target_id)] = self.context.add_place_body(
            config
        )
        self._set_scene_stone_pose(target_id, config)
        self.resume_scene_poses[int(target_id)] = self._pose_array_from_matrix(
            release_T
        )
