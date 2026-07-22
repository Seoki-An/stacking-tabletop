import os
import time
import copy
import ray
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from omegaconf import OmegaConf
import warnings
from typing import Dict, Optional, Tuple
import rclpy
from contextlib import nullcontext

from ros2.joint_node import OpeningAnglePublisher, OpeningAngleSubscriber
from ros2.pose_node import PosePublisher, PoseSubscriber
from ros2.grasp_status_node import GraspStatusSubscriber
from ros2.field_recovery_status_node import FieldRecoveryStatusSubscriber

from .env import StoneStackingEnv
from .mcts import MCTS_Node, MonteCarloTreeSearch
from .rl_models import StackingQfunction, load_model

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RAY_OBJECT_STORE_MIN = 256 * 10**6
RAY_OBJECT_STORE_MAX = 4 * 10**9
RAY_OBJECT_STORE_FRACTION = 0.20
POSE_ID_RESPONSE_TIMEOUT = float(
    os.environ.get("STACKING_POSE_ID_RESPONSE_TIMEOUT", "300.0")
)


def _num_ray_workers(cfg: OmegaConf) -> int:
    return int(float(getattr(cfg.resource, "num_workers", 0)))


def _available_memory_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _ray_object_store_memory() -> int:
    env_value = os.environ.get("STACKING_RAY_OBJECT_STORE_MEMORY")
    if env_value:
        return int(env_value)

    available = _available_memory_bytes()
    if available is None:
        return int(min(RAY_OBJECT_STORE_MAX, 2 * 10**9))
    target = int(available * RAY_OBJECT_STORE_FRACTION)
    return int(max(RAY_OBJECT_STORE_MIN, min(RAY_OBJECT_STORE_MAX, target)))


def get_env_and_policy(
    cfg: OmegaConf,
) -> Tuple[StoneStackingEnv, torch.nn.Module, Dict[str, OmegaConf], OmegaConf]:

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device[-1]
    print("cuda visible devices: ", os.environ["CUDA_VISIBLE_DEVICES"])

    if (
        cfg.algorithm.name in ["dqn_mcts", "bc_mcts", "implicit"]
        and _num_ray_workers(cfg) > 0
    ):
        object_store_memory = _ray_object_store_memory()
        print(f"ray object store memory: {object_store_memory}")
        ray.init(
            object_store_memory=object_store_memory,
            ignore_reinit_error=True,
        )

    n_threads = (
        cfg.resource.num_cpus
        if cfg.resource.num_workers == 0
        else max(cfg.resource.num_cpus // cfg.resource.num_workers - 1, 1)
    )
    env_args = {"cfg": cfg.environment, "n_threads": n_threads}

    env = StoneStackingEnv(env_args)

    if cfg.algorithm.name == "mcts":
        policy = None
    elif cfg.algorithm.name in ["dqn_mcts", "bc_mcts", "implicit"]:
        if cfg.qfunction.load:
            policy, _ = load_model(
                StackingQfunction, cfg.qfunction.load_dir, cfg.device
            )
        else:
            cfg.qfunction.update(OmegaConf.load(cfg.qfunction.config))
            policy = StackingQfunction(cfg.qfunction)
            policy.set_device(cfg.device)
        policy.eval()
    else:
        raise NotImplementedError("undefined algorithm name")
    env.reset()

    return env, policy, env_args, cfg


class IntegratedPlanner:

    def __init__(self, cfg, parallel=True, use_ros=True):
        self.cfg = cfg
        self.parallel = parallel and (_num_ray_workers(cfg) > 0)
        self.env, self.policy, self.env_args, _ = get_env_and_policy(cfg)
        self.num_workers = cfg.resource.num_workers
        self.mcts = MonteCarloTreeSearch(
            cfg.algorithm.mcts, StoneStackingEnv, self.env_args
        )
        if self.parallel:
            MonteCarloTreeSearchRay = ray.remote(MonteCarloTreeSearch)
            self.l_mcts = [
                MonteCarloTreeSearchRay.options(
                    num_cpus=cfg.resource.num_cpus / cfg.resource.num_workers,
                    num_gpus=cfg.resource.num_gpus / cfg.resource.num_workers,
                ).remote(cfg.algorithm.mcts, StoneStackingEnv, self.env_args)
                for _ in range(self.num_workers)
            ]
        if self.cfg.algorithm.name in ["dqn_mcts", "bc_mcts", "implicit"]:
            self.mcts.set_qfunction(
                StackingQfunction, self.policy.cfg, self.policy.state_dict(), cfg.device
            )
            if self.parallel:
                for mcts_ in self.l_mcts:
                    mcts_.set_qfunction.remote(
                        StackingQfunction,
                        self.policy.cfg,
                        self.policy.state_dict(),
                        cfg.device,
                    )
        else:
            raise NotImplementedError("undefined algorithm name")

        if use_ros:
            self.inhand_pose_sub = PoseSubscriber(
                name="inhand_poseid_opt_subscriber", topic="/inhand_poseid_opt"
            )
            self.inhand_pose_pub = PosePublisher(
                name="inhand_poseid_init_publisher", topic="/inhand_poseid_init"
            )
            self.field_pose_init_pub = PosePublisher(
                name="field_poseid_init_publisher", topic="/field_poseid_init"
            )
            self.opening_angle_pub = OpeningAnglePublisher(
                name="opening_angle_publisher", topic="/opening_angle"
            )
            self.inhand_opening_angle_sub = OpeningAngleSubscriber(
                name="inhand_opening_angle_opt_subscriber",
                topic="/inhand_opening_angle_opt",
            )
            self.grasp_status_sub = GraspStatusSubscriber(
                name="grasp_status_subscriber", topic="/grasp_status"
            )
            self.field_recovery_status_sub = FieldRecoveryStatusSubscriber(
                name="field_recovery_status_subscriber",
                topic="/field_recovery_status",
            )
            self.field_recovery_sub = PoseSubscriber(
                name="field_poseid_recovery_subscriber",
                topic="/field_poseid_recovery",
            )

    def plan_one_step(
        self,
        state,
        use_policy=False,
        execution_rejected_actions=None,
        execution_rejected_stone_ids=None,
        seed_actions=None,
    ):
        self.env.update_from_state(state)
        obs = self.env.get_observation()

        node = MCTS_Node(self.cfg.algorithm.mcts)
        node.update_state(state, obs, 0, False, False)

        score = -np.inf
        epsilon = 0.0
        preserve_tree = False

        while score == -np.inf and epsilon <= 1.0:
            if not self.parallel:
                nodes, scores = self.mcts.search(
                    node,
                    use_qfunction=use_policy,
                    preserve_tree=preserve_tree,
                    epsilon=epsilon,
                    eval=False,
                    multiple_nodes=True,
                    log_info=True,
                    execution_rejected_actions=execution_rejected_actions,
                    execution_rejected_stone_ids=execution_rejected_stone_ids,
                    seed_actions=seed_actions,
                )
                debug_nodes = list(
                    getattr(self.mcts, "_last_final_validation_debug_nodes", [])
                )
                if not scores:
                    return None, None, debug_nodes
                score = max(scores)
            else:
                result = ray.get(
                    [
                        mcts_.search.remote(
                            root=copy.deepcopy(node),
                            use_qfunction=use_policy,
                            preserve_tree=preserve_tree,
                            epsilon=epsilon,
                            eval=False,
                            multiple_nodes=True,
                            log_info=True,
                            execution_rejected_actions=execution_rejected_actions,
                            execution_rejected_stone_ids=execution_rejected_stone_ids,
                            seed_actions=seed_actions,
                        )
                        for mcts_ in self.l_mcts
                    ]
                )
                nodes = []
                scores = []
                debug_nodes = []
                for result_ in result:
                    nodes.extend(result_[0])
                    scores.extend(result_[1])

                if not scores:
                    return None, None, None
                debug_nodes = list(nodes)
                node = nodes[scores.index(max(scores))]
                score = max(scores)
            epsilon += 0.1

        if score == -np.inf:
            print("All actions are evaluated as -inf, return None")
            return None, None, None
        else:
            actions = []
            for node_ in nodes:
                action = {
                    "pose": node_.action.pose,
                    "stone_id": self.env.inventory.stone_set[node_.action.stone_idx],
                }
                actions.append(action)

            ranked = sorted(
                zip(nodes, actions, scores),
                key=lambda item: item[2],
                reverse=True,
            )
            nodes = [n for n, _, _ in ranked]
            actions = [a for _, a, _ in ranked]
            scores = [s for _, _, s in ranked]

            if not debug_nodes:
                debug_nodes = list(nodes)
            debug_nodes = sorted(
                debug_nodes,
                key=lambda n: (
                    int((getattr(n, "info", None) or {}).get("final_validation_rank", 10**9)),
                    -float(
                        (getattr(n, "info", None) or {}).get(
                            "final_validation_prior_score",
                            getattr(n, "q_value_init", -np.inf),
                        )
                    ),
                ),
            )

            return actions, nodes, debug_nodes

    def get_inhand_pose(
        self,
        inhand_T: np.ndarray,
        field_T: np.ndarray,
        target_id: int,
        opening_angle: float,
        live_joint_node=None,
        live_state_cb=None,
        spin_lock=None,
        recovery_status_cb=None,
        request_publish_cb=None,
        pcd_spin_cb=None,
        return_opening_angle: bool = False,
    ):
        """Send the planned inhand pose (gripper frame) and the global field
        pose (base frame) of the stone to the nuc, then wait for the result.

        The global field pose is required so the nuc can use it as the ICP
        initial guess if it has to fall back to a field re-scan.

        Returns:
            grasp_success: True if the nuc confirmed the grasp via the inhand
                scan, False if the grasp failed and the nuc rescanned the field.
            inhand_T_opt: optimized inhand transform (only valid on success).
            opening_angle_opt: optional measured in-hand gripper opening angle
                when ``return_opening_angle`` is True.
            field_pose: (pos, quat) of the stone re-identified on the field
                (only valid on failure).
        """
        def make_return(
            grasp_success,
            inhand_T_opt,
            opening_angle_opt,
            field_pose,
        ):
            if return_opening_angle:
                return grasp_success, inhand_T_opt, opening_angle_opt, field_pose
            return grasp_success, inhand_T_opt, field_pose

        def spin_inhand_opening_angle(timeout_sec: float = 0.0) -> None:
            if not hasattr(self, "inhand_opening_angle_sub"):
                return
            with spin_lock if spin_lock is not None else nullcontext():
                rclpy.spin_once(
                    self.inhand_opening_angle_sub,
                    timeout_sec=timeout_sec,
                )

        def read_inhand_opening_angle(default_opening_angle: float) -> float:
            sub = getattr(self, "inhand_opening_angle_sub", None)
            if sub is None:
                return float(default_opening_angle)
            if sub.get_flag():
                value = float(sub.data)
                sub.reset_get_flag()
                if np.isfinite(value):
                    return value

            deadline = time.time() + 2.0
            while time.time() < deadline:
                spin_inhand_opening_angle(timeout_sec=0.05)
                spin_live_joint()
                if not sub.get_flag():
                    continue
                value = float(sub.data)
                sub.reset_get_flag()
                if np.isfinite(value):
                    return value
            print(
                "Timed out waiting for optimized in-hand opening angle; "
                "using planned grasp opening angle."
            )
            return float(default_opening_angle)

        # Drain stale messages from the previous step. The nuc publishes
        # /grasp_status and /field_poseid_recovery in tight loops (10x and
        # 1000x respectively) so several stale messages remain in the
        # RELIABLE depth=10 queue after the previous iteration consumed one.
        # Without this drain the next get_inhand_pose() reads those stale
        # messages instead of the current step's result. A short drain is
        # safe because the nuc takes seconds to do a fresh scan.
        def spin_live_joint(timeout_sec: float = 0.0) -> None:
            if live_joint_node is None:
                return
            with spin_lock if spin_lock is not None else nullcontext():
                rclpy.spin_once(live_joint_node, timeout_sec=timeout_sec)
            if live_state_cb is not None and live_joint_node.get_flag():
                live_state_cb(live_joint_node.pos.copy())
                live_joint_node.reset_get_flag()

        def spin_scan_pcd(timeout_sec: float = 0.0) -> None:
            if pcd_spin_cb is not None:
                pcd_spin_cb(timeout_sec)

        drain_until = time.time() + 0.3
        while time.time() < drain_until:
            with spin_lock if spin_lock is not None else nullcontext():
                rclpy.spin_once(self.grasp_status_sub, timeout_sec=0.0)
                rclpy.spin_once(self.field_recovery_sub, timeout_sec=0.0)
                rclpy.spin_once(self.field_recovery_status_sub, timeout_sec=0.0)
                rclpy.spin_once(self.inhand_pose_sub, timeout_sec=0.0)
                if hasattr(self, "inhand_opening_angle_sub"):
                    rclpy.spin_once(self.inhand_opening_angle_sub, timeout_sec=0.0)
            spin_live_joint()
            spin_scan_pcd()
        self.grasp_status_sub.reset_get_flag()
        self.field_recovery_sub.reset_get_flag()
        self.field_recovery_status_sub.reset_get_flag()
        self.inhand_pose_sub.reset_get_flag()
        if hasattr(self, "inhand_opening_angle_sub"):
            self.inhand_opening_angle_sub.reset_get_flag()

        inhand_pos = inhand_T[:3, -1]
        inhand_quat = Rotation.from_matrix(inhand_T[:3, :3]).as_quat()
        field_pos = field_T[:3, -1]
        field_quat = Rotation.from_matrix(field_T[:3, :3]).as_quat()

        def publish_initial_poses(count: int = 3) -> None:
            if request_publish_cb is not None:
                request_publish_cb()
            for _ in range(count):
                self.inhand_pose_pub.publish(inhand_pos, inhand_quat, target_id)
                self.field_pose_init_pub.publish(field_pos, field_quat, target_id)
                self.opening_angle_pub.publish(opening_angle)

        def spin_field_recovery_status(timeout_sec: float = 0.0) -> None:
            with spin_lock if spin_lock is not None else nullcontext():
                rclpy.spin_once(
                    self.field_recovery_status_sub,
                    timeout_sec=timeout_sec,
                )
            if not self.field_recovery_status_sub.get_flag():
                return
            status = self.field_recovery_status_sub.status
            stone_id = self.field_recovery_status_sub.stone_id
            detail = self.field_recovery_status_sub.detail
            self.field_recovery_status_sub.reset_get_flag()
            if int(stone_id) != int(target_id):
                print(
                    "Ignoring field recovery status with mismatched id: "
                    f"expected {target_id}, got {stone_id}"
                )
                return
            if recovery_status_cb is not None:
                recovery_status_cb(status, stone_id, detail)

        publish_initial_poses(count=10)

        response_start = time.time()
        last_init_publish = time.time()
        while not self.grasp_status_sub.get_flag():
            if time.time() - response_start > POSE_ID_RESPONSE_TIMEOUT:
                print(
                    "Timed out waiting for grasp status from pose identification "
                    f"after {POSE_ID_RESPONSE_TIMEOUT:.1f}s."
                )
                return make_return(False, None, None, None)
            if time.time() - last_init_publish > 0.5:
                publish_initial_poses()
                last_init_publish = time.time()
            with spin_lock if spin_lock is not None else nullcontext():
                rclpy.spin_once(self.grasp_status_sub, timeout_sec=0.1)
            spin_live_joint()
            spin_scan_pcd()
        self.grasp_status_sub.reset_get_flag()
        grasp_success = self.grasp_status_sub.success

        if grasp_success:
            self.inhand_pose_sub.reset_get_flag()
            response_start = time.time()
            last_init_publish = time.time()
            while True:
                while not self.inhand_pose_sub.get_flag():
                    if time.time() - response_start > POSE_ID_RESPONSE_TIMEOUT:
                        print(
                            "Timed out waiting for optimized in-hand pose "
                            f"after {POSE_ID_RESPONSE_TIMEOUT:.1f}s."
                        )
                        return make_return(False, None, None, None)
                    if time.time() - last_init_publish > 0.5:
                        publish_initial_poses()
                        last_init_publish = time.time()
                    with spin_lock if spin_lock is not None else nullcontext():
                        rclpy.spin_once(self.inhand_pose_sub, timeout_sec=0.1)
                    spin_inhand_opening_angle(timeout_sec=0.0)
                    spin_live_joint()
                    spin_scan_pcd()
                    spin_field_recovery_status()

                if int(self.inhand_pose_sub.id) == int(target_id):
                    break
                print(
                    "Ignoring optimized in-hand pose with mismatched id: "
                    f"expected {target_id}, got {self.inhand_pose_sub.id}"
                )
                self.inhand_pose_sub.reset_get_flag()

            inhand_pos_opt = self.inhand_pose_sub.pos.copy()
            inhand_quat_opt = self.inhand_pose_sub.quat.copy()
            self.inhand_pose_sub.reset_get_flag()
            print(
                f"Identified inhand pose: {inhand_pos_opt}, "
                f"{inhand_quat_opt}"
            )
            inhand_T_opt = np.eye(4)
            inhand_T_opt[:3, -1] = inhand_pos_opt
            inhand_T_opt[:3, :3] = Rotation.from_quat(inhand_quat_opt).as_matrix()
            opening_angle_opt = read_inhand_opening_angle(opening_angle)
            print(
                "Identified inhand opening angle: "
                f"{opening_angle_opt:.6f} rad"
            )
            return make_return(True, inhand_T_opt, opening_angle_opt, None)

        self.field_recovery_sub.reset_get_flag()
        self.field_recovery_status_sub.reset_get_flag()
        response_start = time.time()
        last_init_publish = time.time()
        while not self.field_recovery_sub.get_flag():
            if time.time() - response_start > POSE_ID_RESPONSE_TIMEOUT:
                print(
                    "Timed out waiting for recovered field pose after grasp "
                    f"failure after {POSE_ID_RESPONSE_TIMEOUT:.1f}s."
                )
                return make_return(False, None, None, None)
            if time.time() - last_init_publish > 0.5:
                self.field_pose_init_pub.publish(field_pos, field_quat, target_id)
                last_init_publish = time.time()
            with spin_lock if spin_lock is not None else nullcontext():
                rclpy.spin_once(self.field_recovery_sub, timeout_sec=0.1)
            spin_live_joint()
            spin_scan_pcd()
            spin_field_recovery_status()
        self.field_recovery_sub.reset_get_flag()
        recovered_pose = (
            self.field_recovery_sub.pos.copy(),
            self.field_recovery_sub.quat.copy(),
        )
        if int(self.field_recovery_sub.id) != int(target_id):
            print(
                "Ignoring recovered field pose with mismatched id: "
                f"expected {target_id}, got {self.field_recovery_sub.id}"
            )
            return make_return(False, None, None, None)
        print(
            f"Grasp failed; recovered field pose: {recovered_pose[0]}, "
            f"{recovered_pose[1]}"
        )
        return make_return(False, None, None, recovered_pose)

    def request_field_pose_recovery(
        self,
        field_T: np.ndarray,
        target_id: int,
        spin_lock=None,
        pcd_spin_cb=None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Ask the nuc to re-identify a stone from a field scan."""
        drain_until = time.time() + 0.3
        while time.time() < drain_until:
            with spin_lock if spin_lock is not None else nullcontext():
                rclpy.spin_once(self.field_recovery_sub, timeout_sec=0.0)
                rclpy.spin_once(self.field_recovery_status_sub, timeout_sec=0.0)
            if pcd_spin_cb is not None:
                pcd_spin_cb(0.0)
        self.field_recovery_sub.reset_get_flag()
        self.field_recovery_status_sub.reset_get_flag()

        field_pos = field_T[:3, -1]
        field_quat = Rotation.from_matrix(field_T[:3, :3]).as_quat()

        def publish_field_init(count: int = 3) -> None:
            for _ in range(count):
                self.field_pose_init_pub.publish(field_pos, field_quat, target_id)

        publish_field_init(count=10)

        self.field_recovery_sub.reset_get_flag()
        response_start = time.time()
        last_init_publish = time.time()
        while not self.field_recovery_sub.get_flag():
            if time.time() - response_start > POSE_ID_RESPONSE_TIMEOUT:
                print(
                    "Timed out waiting for intermediate field pose recovery "
                    f"after {POSE_ID_RESPONSE_TIMEOUT:.1f}s."
                )
                return None
            if time.time() - last_init_publish > 0.5:
                publish_field_init()
                last_init_publish = time.time()
            with spin_lock if spin_lock is not None else nullcontext():
                rclpy.spin_once(self.field_recovery_sub, timeout_sec=0.1)
            if pcd_spin_cb is not None:
                pcd_spin_cb(0.0)
        self.field_recovery_sub.reset_get_flag()
        recovered_pose = (
            self.field_recovery_sub.pos.copy(),
            self.field_recovery_sub.quat.copy(),
        )
        if int(self.field_recovery_sub.id) != int(target_id):
            print(
                "Ignoring recovered intermediate field pose with mismatched id: "
                f"expected {target_id}, got {self.field_recovery_sub.id}"
            )
            return None
        print(
            f"Recovered intermediate field pose: {recovered_pose[0]}, "
            f"{recovered_pose[1]}"
        )
        return recovered_pose
