import numpy as np
from scipy.spatial.transform import Rotation

from .action.orientation import inward_xy, top_exposed_surface_normal
from .action.support import placement_support_score
from .inventory import InventoryManager
from .state import State


class RewardComputer:
    def __init__(self, cfg):
        self.cfg = cfg

    def compute(self, inventory: InventoryManager, state: State):

        stones = inventory.stones
        target = inventory.target_wall
        seq = state.stone_seq
        c_feq = state.action_history[-1].c_feq
        place_robustness_displacement = state.action_history[
            -1
        ].place_robustness_displacement
        place_cfg = self.cfg.reward.get("place_stability", {})
        place_robustness_nonfinite = not np.isfinite(place_robustness_displacement)
        if place_robustness_nonfinite:
            place_robustness_displacement = float(
                place_cfg.get("nonfinite_displacement", 2.0)
            )
        max_displacement = place_cfg.get("max_displacement", None)
        place_robustness_clipped = False
        if max_displacement is not None and place_robustness_displacement > float(
            max_displacement
        ):
            place_robustness_displacement = float(max_displacement)
            place_robustness_clipped = True

        previous_target_IoU = 0.0
        for idx in seq[:-1]:
            st = stones[idx]
            pose = state.stone_poses.get(st.id, st.pose)
            previous_target_IoU += target.compute_IoU(st, pose, False, False)

        target_IoU = 0.0
        for idx in seq:
            st = stones[idx]
            pose = state.stone_poses.get(st.id, st.pose)
            target_IoU += target.compute_IoU(st, pose, False, False)
        target_IoU_delta = target_IoU - previous_target_IoU
        target_IoU_increment = max(target_IoU_delta, 0.0)
        placed_idx = seq[-1]
        placed_stone = stones[placed_idx]
        placed_pose = state.stone_poses.get(placed_stone.id, placed_stone.pose)
        stone_IoU = target.compute_IoU(
            placed_stone, placed_pose, True, True
        )

        stability = -min(c_feq / self.cfg.reward.posegen_thresh, 1.0)
        place_stability = -place_robustness_displacement
        large_stone_lower = self._large_stone_lower_score(
            stones, seq, state.stone_poses, inventory.target_wall.height
        )
        inward_orientation = self._inward_orientation_score(
            inventory,
            placed_idx,
            placed_pose,
        )
        support_score, support_count, support_has_ground = placement_support_score(
            inventory,
            state,
            placed_idx,
            placed_pose,
        )
        stone_IoU_reward = min(
            (2 * stone_IoU - self.cfg.reward.IoU_thresh) / self.cfg.reward.IoU_thresh,
            1.0,
        )

        reward = (
            self.cfg.reward.weights.stability * stability
            + self.cfg.reward.weights.stone_IoU * stone_IoU_reward
            + self.cfg.reward.weights.get("target_IoU_increment", 0.0)
            * target_IoU_increment
            + self.cfg.reward.weights.get("place_stability", 0.0) * place_stability
            + self.cfg.reward.weights.get("large_stone_lower", 0.0) * large_stone_lower
            + self.cfg.reward.weights.get("inward_orientation", 0.0)
            * inward_orientation
            + self.cfg.reward.weights.get("support", 0.0) * support_score
        )

        return reward, {
            "stability": stability,
            "target_IoU": target_IoU,
            "previous_target_IoU": previous_target_IoU,
            "target_IoU_delta": target_IoU_delta,
            "target_IoU_increment": target_IoU_increment,
            "stone_IoU": stone_IoU,
            "stone_IoU_reward": stone_IoU_reward,
            "place_stability": place_stability,
            "place_robustness_displacement": place_robustness_displacement,
            "place_robustness_nonfinite": place_robustness_nonfinite,
            "place_robustness_clipped": place_robustness_clipped,
            "large_stone_lower": large_stone_lower,
            "inward_orientation": inward_orientation,
            "support": support_score,
            "support_count": support_count,
            "support_has_ground": support_has_ground,
        }

    def _large_stone_lower_score(
        self,
        stones,
        seq,
        stone_poses: dict,
        target_height: float,
    ) -> float:
        volumes = np.array([stone.volume for stone in stones], dtype=float)
        volume_range = volumes.max() - volumes.min()
        if volume_range <= 0.0:
            volume_score = 0.5
        else:
            volume_score = (stones[seq[-1]].volume - volumes.min()) / volume_range

        placed_stone = stones[seq[-1]]
        placed_pose = stone_poses.get(placed_stone.id, placed_stone.pose)
        z = float(placed_pose[2])
        lowness_score = 1.0 - float(np.clip(z / max(target_height, 1e-6), 0.0, 1.0))

        return float(volume_score * lowness_score)

    def _inward_orientation_score(
        self,
        inventory: InventoryManager,
        stone_idx: int,
        pose: np.ndarray,
    ) -> float:
        stone = inventory.stones[stone_idx]
        normals, areas = stone.representative_face_normals()
        if len(normals) == 0:
            return 0.0

        orient_cfg = (
            self.cfg.action.planar.get("floor_fill", {})
            .get("orientation", {})
        )
        world_normals = normals @ Rotation.from_quat(pose[3:]).as_matrix().T
        min_z = float(orient_cfg.get("upper_face_min_z", 0.0))
        normal = top_exposed_surface_normal(world_normals, areas, min_z=min_z)
        if normal is None:
            return 0.0

        min_upward = float(orient_cfg.get("min_upward_dot", 0.7))
        if min_upward >= 1.0:
            upward_score = 1.0 if float(normal[2]) > min_upward else 0.0
        else:
            upward_score = np.clip(
                (float(normal[2]) - min_upward) / (1.0 - min_upward),
                0.0,
                1.0,
            )

        heading = np.asarray(normal[:2], dtype=float)
        heading_norm = np.linalg.norm(heading)
        min_horizontal = float(orient_cfg.get("upper_face_min_horizontal", 0.05))
        if heading_norm < min_horizontal:
            inward_score = 0.5
        else:
            inward = inward_xy(inventory, pose[:2])
            if np.linalg.norm(inward) < 1e-9:
                inward_score = 0.5
            else:
                max_angle = float(orient_cfg.get("max_inward_angle_deg", 90.0))
                min_dot = float(np.cos(np.deg2rad(max_angle)))
                if np.isclose(max_angle, 90.0):
                    min_dot = 0.0
                dot = float(np.dot(heading / heading_norm, inward))
                denom = max(1.0 - min_dot, 1e-9)
                inward_score = float(np.clip((dot - min_dot) / denom, 0.0, 1.0))
        return float(upward_score * inward_score)
