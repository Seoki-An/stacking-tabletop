import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Dict, Optional, Tuple

from .components.contexts import environment_ground_height, get_diffsim
from .components.inventory import InventoryManager
from .components.state import State, StoneTrajectory
from .components.action import Action

from utils._phase_timer import timed, increment
from utils.etc import resolve_thread_count


@dataclass(frozen=True)
class SimulationLimits:
    dt: float
    extra_n_step: int
    energy_thresh: float
    vel_thresh: float
    min_t: float
    max_t: float
    freeze_radius: Optional[float]


class Simulator:
    def __init__(self, cfg, fast: bool = False, n_threads: Optional[int] = None):
        self.cfg = cfg
        self.fast = fast
        self.n_threads = resolve_thread_count(n_threads, cfg)
        self.inventory = InventoryManager(cfg)

        # Cache hot-path config fields to avoid OmegaConf resolver overhead in tight loops.
        self._sim_dt         = float(cfg.sim.dt)
        self._n_stone        = int(cfg.n_stone)
        self._settle_n_step  = int(cfg.sim.get("settle_n_step", 0))
        self._ground_height  = environment_ground_height(cfg)
        # Simulate scene-scan-identified stones instead of freezing them. Their
        # body error_reduction_ratio is set to 0.0 so contacts among the
        # (interpenetrating) reconstructed stones get erp=max(0,0)=0 while
        # contacts against ordinary stones keep the default erp.
        # simulate_pose_identified_ids optionally restricts the simulated set to
        # specific stone ids (None = all identified when the flag is on).
        self._simulate_pose_identified = bool(
            cfg.sim.get("simulate_pose_identified", False)
        )
        ids = cfg.sim.get("simulate_pose_identified_ids", None)
        self._simulate_pose_identified_ids = (
            None if ids is None else {int(i) for i in ids}
        )

        self.sim = None
        self.plane = None

        # Opt-in settle diagnostic: per-simulation record of the peak motion,
        # which stone drove it (new vs pre-existing), and the residual motion
        # left by the phase-1 settle. Off by default (no overhead).
        self._collect_settle_diag = False
        self._settle_diag_samples: List[dict] = []
        self._last_settle_initial_energy = 0.0

        self.reset()

    def set_collect_settle_diag(self, enabled: bool) -> None:
        """Enable/disable per-simulation settle diagnostics and clear the buffer."""
        self._collect_settle_diag = bool(enabled)
        self._settle_diag_samples = []

    def take_settle_diag(self) -> List[dict]:
        """Return and clear the settle diagnostics collected since the last call."""
        samples = list(self._settle_diag_samples)
        self._settle_diag_samples = []
        return samples

    def _record_settle_diag(
        self, action, vel_integral, place_robustness_failed: bool
    ) -> None:
        n = len(self.stone_seq)
        vals = [float(v) for v in vel_integral[:n]]
        if not vals:
            return
        finite = bool(np.all(np.isfinite(vals)))
        argmax_i = int(np.argmax(vals))
        argmax_stone_idx = self.stone_seq[argmax_i]
        self._settle_diag_samples.append(
            {
                "vel_max": float(max(vals)) if finite else float("inf"),
                # phase-1 residual: instantaneous energy at the first phase-2 step
                "initial_energy": float(self._last_settle_initial_energy),
                "argmax_stone_id": int(self.inventory.stones[argmax_stone_idx].id),
                "is_new_stone": bool(argmax_stone_idx == action.stone_idx),
                "failed": bool(self.failed),
                "place_robustness_failed": bool(place_robustness_failed),
                "n_stones": n,
            }
        )

    def reset(self):
        self.sim, self.plane = get_diffsim(
            self.fast,
            ground_height=self._ground_height,
        )
        self.plane_id = self.sim.get_body_ids()[0]

        self.n_step = 0
        self.stone_seq = []

        self.inventory.reset()

        self.body_id_sim = []

        self.trajectories: List[Dict[int, StoneTrajectory]] = []
        self.action_history: List[Action] = []
        self.contact_points: List[dict] = []
        self.pose_identified_stone_ids: set[int] = set()
        self.terminated = False
        self.failed = False
        self.simulation_settled = True

        return self._build_state()

    def simulation_limits(
        self,
        mode: str,
        overrides=None,
    ) -> SimulationLimits:
        sim_cfg = self.cfg.sim
        profile = sim_cfg.get(mode, {})
        overrides = overrides or {}

        def value(name, default=None):
            if overrides.get(name, None) is not None:
                return overrides[name]
            if profile.get(name, None) is not None:
                return profile[name]
            return sim_cfg.get(name, default)

        return SimulationLimits(
            dt=float(value("dt")),
            extra_n_step=int(value("extra_n_step")),
            energy_thresh=float(value("energy_thresh")),
            vel_thresh=float(
                value("vel_integral_thresh", self.cfg.reward.vel_integral_thresh)
            ),
            min_t=float(value("min_t")),
            max_t=float(value("max_t")),
            freeze_radius=(
                None
                if value("freeze_radius", None) is None
                else float(value("freeze_radius"))
            ),
        )

    def recreate_physics_context(self) -> None:
        """Replace diffsim state without reloading or randomizing inventory."""
        self.sim, self.plane = get_diffsim(
            self.fast,
            ground_height=self._ground_height,
        )
        self.plane_id = self.sim.get_body_ids()[0]
        self.body_id_sim = []

    @timed("physics")
    def step(
        self,
        action: Action,
        simulate: bool = True,
        simulation_mode: str = "long",
        simulation_overrides=None,
    ):
        # Simulation owns this action copy. Frozen settling refines its solved
        # pose, and callers may be validating the same candidate concurrently.
        action = self._resolve_action_stone_index(action).copy()
        self.n_step += 1

        prev_stone_seq = self.stone_seq.copy()
        prev_stone_poses = {
            idx: self.inventory.stones[idx].pose.copy() for idx in prev_stone_seq
        }

        stone = self.inventory.stones[action.stone_idx]
        stone.pose = action.pose.copy()
        self.stone_seq.append(action.stone_idx)

        # add to sim
        self.body_id_sim.append(self.sim.add_body(stone.config))

        # Settling simulation: refine the new stone's pose with the existing
        # scene frozen before any structural-stability simulation.
        frozen_settle = self._freeze_all_but_last()
        settle_trajectory = self._simulate_settle()
        for body_id in frozen_settle:
            self.sim.unfreeze_body(body_id)
        self._update_last_stone_pose_from_sim()

        # Settling simulation is the final pose-solve refinement. Persist it in the
        # action so tree descendants and execution use the configuration that
        # was actually evaluated by the following short/long simulation.
        action.pose = self.inventory.stones[action.stone_idx].pose.copy()

        limits = self.simulation_limits(simulation_mode, simulation_overrides)

        place_robustness_failed = False
        if simulate and self._place_robustness_enabled():
            action.place_robustness_displacement = self._max_noisy_place_displacement(
                action, prev_stone_seq, prev_stone_poses, limits
            )
            place_robustness_failed = self._place_robustness_failed(
                action.place_robustness_displacement
            )
        self.action_history.append(action.copy())

        # Short/long simulation: evaluate structural stability from the refined
        # pose. The MCTS caller selects the active limits.
        pose_identified_body_ids = set(
            self._freeze_pose_identified_bodies(exclude_stone_idx=action.stone_idx)
        )
        frozen_ids = []
        try:
            frozen_ids = self._freeze_far_bodies(
                self.sim,
                self.body_id_sim,
                self.stone_seq,
                action.stone_idx,
                limits.freeze_radius,
            )

            # simulate overall structure stability
            vel_integral, trajectory, simulation_settled = (
                self._simulate(limits, settle_trajectory)
                if simulate
                else ([0.0] * self._n_stone, None, True)
            )
        finally:
            for body_id in frozen_ids:
                if body_id not in pose_identified_body_ids:
                    self.sim.unfreeze_body(body_id)
            for body_id in pose_identified_body_ids:
                self.sim.unfreeze_body(body_id)

        if trajectory is None:
            trajectory = {}
            for idx in self.stone_seq:
                stone = self.inventory.stones[idx]
                if idx == action.stone_idx:
                    trajectory[stone.id] = settle_trajectory
                else:
                    trajectory[stone.id] = StoneTrajectory(stone.id)
                    trajectory[stone.id].add_pose(stone.pose)
                trajectory[stone.id].vel_integral = 0.0

        self.trajectories.append(trajectory if simulate else trajectory)

        # update poses
        if simulate:
            self._update_state_from_sim()
            self.contact_points = self._extract_contact_points()
        else:
            self.contact_points = []

        self.terminated = self.n_step >= self._n_stone
        self.failed = max(
            vel_integral
        ) > limits.vel_thresh or not np.all(np.isfinite(vel_integral))
        self.failed = bool(self.failed or place_robustness_failed)
        self.simulation_settled = bool(simulation_settled)

        if self._collect_settle_diag and simulate:
            self._record_settle_diag(action, vel_integral, place_robustness_failed)

        state = self._build_state()

        return state

    def _resolve_action_stone_index(self, action: Action) -> Action:
        if action.stone_idx < 0 and action.stone_id < 0:
            raise ValueError("cannot step with a dummy action")

        stone_set = np.asarray(self.inventory.stone_set, dtype=int)
        resolved = action
        if action.stone_id >= 0:
            matches = np.flatnonzero(stone_set == int(action.stone_id))
            if len(matches) == 0:
                raise ValueError(
                    f"action stone_id {action.stone_id} is absent from inventory "
                    f"stone_set {stone_set.tolist()}"
                )
            stone_idx = int(matches[0])
            if stone_idx != action.stone_idx:
                resolved = action.copy()
                resolved.stone_idx = stone_idx
        elif action.stone_idx >= len(stone_set):
            raise IndexError(
                f"action stone_idx {action.stone_idx} is out of range for "
                f"inventory with {len(stone_set)} stones"
            )
        else:
            resolved = action.copy()
            resolved.stone_id = int(stone_set[action.stone_idx])

        if resolved.stone_idx in self.stone_seq:
            raise ValueError(
                f"stone_idx {resolved.stone_idx} / stone_id {resolved.stone_id} "
                "has already been placed"
            )
        return resolved

    def _place_robustness_enabled(self) -> bool:
        robust_cfg = self.cfg.reward.get("place_stability", {})
        return (
            bool(robust_cfg.get("enabled", False))
            and int(robust_cfg.get("n_noise", 0)) > 0
        )

    def _place_robustness_failed(self, displacement: float) -> bool:
        robust_cfg = self.cfg.reward.get("place_stability", {})
        fail_displacement = robust_cfg.get("fail_displacement", None)
        if fail_displacement is None:
            return False
        return (
            not np.isfinite(displacement)
            or float(displacement) > float(fail_displacement)
        )

    def _max_noisy_place_displacement(
        self,
        action: Action,
        prev_stone_seq: List[int],
        prev_stone_poses: Dict[int, np.ndarray],
        limits: SimulationLimits,
    ) -> float:
        robust_cfg = self.cfg.reward.place_stability
        rng = np.random.default_rng(self._place_robustness_seed(action, robust_cfg))
        n_noise = int(robust_cfg.n_noise)
        rotation_weight = float(robust_cfg.get("rotation_weight", 0.0))

        noisy_poses = [
            self._add_place_pose_noise(action.pose, robust_cfg, rng)
            for _ in range(n_noise)
        ]

        def _eval(noisy_pose: np.ndarray) -> float:
            final_pose = self._simulate_noisy_place(
                action.stone_idx,
                noisy_pose,
                prev_stone_seq,
                prev_stone_poses,
                limits,
            )
            return self._pose_displacement(noisy_pose, final_pose, rotation_weight)

        if self.n_threads > 1 and n_noise > 1:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                displacements = list(executor.map(_eval, noisy_poses))
        else:
            displacements = [_eval(p) for p in noisy_poses]

        displacement = float(max(displacements, default=0.0))
        max_displacement = robust_cfg.get("max_displacement", None)
        if max_displacement is None:
            return displacement
        if not np.isfinite(displacement):
            return float(max_displacement)
        return min(displacement, float(max_displacement))

    def _place_robustness_seed(self, action: Action, robust_cfg) -> int:
        stone_id = int(self.inventory.stone_set[action.stone_idx])
        seed = int(robust_cfg.get("seed", 0))
        return seed + 1009 * self.n_step + 6361 * action.stone_idx + 9176 * stone_id

    def _add_place_pose_noise(self, pose: np.ndarray, robust_cfg, rng) -> np.ndarray:
        noisy_pose = pose.copy()
        position_std = np.asarray(robust_cfg.get("position_std", 0.0), dtype=float)
        if position_std.ndim == 0:
            position_std = np.repeat(float(position_std), 3)
        noisy_pose[:3] += rng.normal(0.0, position_std, size=3)

        rotation_std_deg = float(robust_cfg.get("rotation_std_deg", 0.0))
        if rotation_std_deg > 0.0:
            rot_noise = rng.normal(0.0, np.deg2rad(rotation_std_deg), size=3)
            noisy_pose[3:] = (
                Rotation.from_rotvec(rot_noise) * Rotation.from_quat(noisy_pose[3:])
            ).as_quat()
        return noisy_pose

    def _simulate_noisy_place(
        self,
        stone_idx: int,
        place_pose: np.ndarray,
        prev_stone_seq: List[int],
        prev_stone_poses: Dict[int, np.ndarray],
        limits: SimulationLimits,
    ) -> np.ndarray:
        sim, plane = get_diffsim(self.fast, ground_height=self._ground_height)
        sim.add_body(plane)

        noisy_seq = prev_stone_seq + [stone_idx]
        body_ids = []
        for idx in prev_stone_seq:
            stone = self.inventory.stones[idx].copy()
            stone.pose = prev_stone_poses[idx]
            body_ids.append(sim.add_body(stone.config))

        stone = self.inventory.stones[stone_idx].copy()
        stone.pose = place_pose
        body_ids.append(sim.add_body(stone.config))

        # Freeze all previously placed stones; only the new stone is dynamic.
        frozen_ids = body_ids[:-1]
        for body_id in frozen_ids:
            self._freeze_body_at_rest(sim, body_id)
        self._run_local_sim(sim, body_ids, noisy_seq, limits)
        for body_id in frozen_ids:
            sim.unfreeze_body(body_id)
        final_pose = sim.state().pose(body_ids[-1]).vectorized()
        if not np.all(np.isfinite(final_pose)):
            return np.full_like(place_pose, np.inf)
        return final_pose.copy()

    def _freeze_body_at_rest(self, sim, body_id: int) -> None:
        """Freeze a body with its motion zeroed.

        diffsim keeps a frozen body's last velocity and feeds it into the
        contact error as a kinematic term (and restores it on unfreeze), so a
        stone frozen mid-motion would push its neighbours like a conveyor.
        Frozen stones represent static scenery; they must be at rest.
        """
        motion = sim.state().motion(body_id)
        motion.setLinear(np.zeros(3))
        motion.setAngular(np.zeros(3))
        sim.freeze_body(body_id)

    def _support_chain_body_ids(self, sim, body_ids: List[int], new_body_id: int):
        """Bodies transitively supporting the new stone (descending contact walk).

        Walks the current contact graph downward from the newly placed stone:
        a contact neighbour whose center is not clearly above the current body
        is treated as (part of) its support. These bodies carry the new
        stone's load, so freezing them would hide exactly the collapse modes
        the stability sim exists to detect.
        """
        chain = {int(new_body_id)}
        try:
            contacts = sim.get_contact_points()
        except Exception:
            return chain

        adjacency = {}
        for cp in contacts:
            a, b = int(cp.id_1), int(cp.id_2)
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        stone_body_ids = {int(b) for b in body_ids}

        def center_z(body_id: int) -> float:
            return float(sim.state().pose(body_id).position()[2])

        stack = [int(new_body_id)]
        while stack:
            cur = stack.pop()
            z_cur = center_z(cur)
            for nb in adjacency.get(cur, ()):
                if nb not in stone_body_ids or nb in chain:
                    continue  # skips the ground plane and visited bodies
                if center_z(nb) < z_cur + 0.05:
                    chain.add(nb)
                    stack.append(nb)
        return chain

    def _freeze_far_bodies(
        self,
        sim,
        body_ids: List[int],
        stone_seq: List[int],
        new_stone_idx: int,
        radius: Optional[float],
    ) -> List[int]:
        """Freeze bodies farther (in XY) than the active profile radius from the
        newly placed stone, except bodies in its support chain. Returns the
        list of frozen body IDs.

        The distance is horizontal only and the support chain is always kept
        dynamic: with a 3D radius a tall stack froze the stones directly
        underneath the new one, so validation simulated the tower on a rigid
        foundation and could not see it collapse.
        """
        if not radius:
            return []

        new_pos = self.inventory.stones[new_stone_idx].pose[:3]

        new_body_id = None
        for i, idx in enumerate(stone_seq):
            if idx == new_stone_idx:
                new_body_id = body_ids[i]
        support_chain = (
            self._support_chain_body_ids(sim, body_ids, new_body_id)
            if new_body_id is not None
            else set()
        )

        frozen = []
        simulate_ids = self._pose_identified_simulate_set()
        for i, idx in enumerate(stone_seq):
            if idx == new_stone_idx:
                continue
            if int(body_ids[i]) in support_chain:
                continue
            stone_id = int(self.inventory.stones[idx].id)
            # Identified stones are frozen separately unless they are being
            # simulated, in which case the normal freeze-radius policy applies.
            if (
                stone_id in self.pose_identified_stone_ids
                and stone_id not in simulate_ids
            ):
                continue
            stone_pos = self.inventory.stones[idx].pose[:3]
            if np.linalg.norm(stone_pos[:2] - new_pos[:2]) > radius:
                self._freeze_body_at_rest(sim, body_ids[i])
                frozen.append(body_ids[i])
        return frozen

    def set_simulate_pose_identified(self, enabled: bool, stone_ids=None) -> None:
        """Runtime override (e.g. from the preview GUI stone selection)."""
        self._simulate_pose_identified = bool(enabled)
        self._simulate_pose_identified_ids = (
            None if stone_ids is None else {int(i) for i in stone_ids}
        )

    def _pose_identified_simulate_set(self) -> set:
        """Identified stone ids that should be simulated instead of frozen."""
        if not self._simulate_pose_identified:
            return set()
        identified = {int(i) for i in self.pose_identified_stone_ids}
        if self._simulate_pose_identified_ids is None:
            return identified
        return identified & self._simulate_pose_identified_ids

    def _freeze_pose_identified_bodies(
        self, exclude_stone_idx: Optional[int] = None
    ) -> List[int]:
        """Freeze placed stones whose poses came from a scene scan."""
        if not self.pose_identified_stone_ids:
            return []
        simulate_ids = self._pose_identified_simulate_set()

        frozen = []
        for body_id, stone_idx in zip(self.body_id_sim, self.stone_seq):
            if exclude_stone_idx is not None and stone_idx == exclude_stone_idx:
                continue
            stone_id = int(self.inventory.stones[stone_idx].id)
            if stone_id not in self.pose_identified_stone_ids:
                continue
            if stone_id in simulate_ids:
                continue
            self._freeze_body_at_rest(self.sim, body_id)
            frozen.append(body_id)
        return frozen

    def _freeze_all_but_last(self) -> List[int]:
        """Freeze every placed stone (at rest) except the most-recently-added one."""
        frozen = []
        for body_id in self.body_id_sim[:-1]:
            self._freeze_body_at_rest(self.sim, body_id)
            frozen.append(body_id)
        return frozen

    def _simulate_settle(self) -> StoneTrajectory:
        """Refine the solved pose while all existing stones remain fixed.

        Keep bounded diagnostics for the refinement, then expose only its final
        pose as the starting point of the structural stability trajectory.
        """
        stone_idx = self.stone_seq[-1]
        stone = self.inventory.stones[stone_idx]
        body_id = self.body_id_sim[-1]
        trajectory = StoneTrajectory(stone.id)

        start_pose = self.sim.state().pose(body_id).vectorized().copy()
        if np.all(np.isfinite(start_pose)):
            trajectory.add_pose(start_pose)

        # The viewer draws at most 24 ghosts. Keeping at most about 64 settle
        # samples preserves the path without bloating every MCTS state.
        sample_stride = max(1, self._settle_n_step // 64)
        for step_idx in range(self._settle_n_step):
            self.sim.step(self._sim_dt)

            is_sample = (
                (step_idx + 1) % sample_stride == 0
                or step_idx + 1 == self._settle_n_step
            )
            if not is_sample:
                continue
            pose = self.sim.state().pose(body_id).vectorized().copy()
            if np.all(np.isfinite(pose)):
                trajectory.add_pose(pose)

        poses = np.asarray(trajectory.poses, dtype=float)
        trajectory.settle_pose_count = len(trajectory.poses)
        if len(poses) >= 2:
            trajectory.settle_position_delta = float(
                np.linalg.norm(poses[-1, :3] - poses[0, :3])
            )
            trajectory.settle_path_length = float(
                np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()
            )
        elif not np.all(np.isfinite(self.sim.state().pose(body_id).vectorized())):
            trajectory.settle_position_delta = float("inf")
            trajectory.settle_path_length = float("inf")
        if len(poses) >= 1:
            trajectory.poses = [poses[-1].copy()]
        return trajectory

    def _update_last_stone_pose_from_sim(self) -> None:
        """Read back the most-recently-placed stone's pose from the simulator."""
        if not self.body_id_sim:
            return
        pose = self.sim.state().pose(self.body_id_sim[-1])
        if np.all(np.isfinite(pose.vectorized())):
            self.inventory.stones[self.stone_seq[-1]].pose = pose

    def _run_local_sim(
        self,
        sim,
        body_ids: List[int],
        stone_seq: List[int],
        limits: SimulationLimits,
    ) -> List[float]:
        energy = [0.0 for _ in stone_seq]
        vel_integral = [0.0 for _ in stone_seq]
        t = 0.0
        extra = 0
        extra_n_step = limits.extra_n_step
        extra_n_step *= 2 if self.n_step == self._n_stone else 1

        dt = 2 * limits.dt
        energy_thresh = limits.energy_thresh
        vel_thresh = limits.vel_thresh
        min_t = limits.min_t
        max_t = limits.max_t
        while extra < extra_n_step:
            sim.step(dt)
            t += dt
            for i, idx in enumerate(stone_seq):
                pose = sim.state().pose(body_ids[i])
                motion = sim.state().motion(body_ids[i])
                pose_vec = pose.vectorized()
                pose_finite = bool(np.all(np.isfinite(pose_vec)))

                mass = self.inventory.stones[idx].mass
                inertia = self.inventory.stones[idx].inertia

                if pose_finite:
                    rot = Rotation.from_quat(pose.orientation()).as_matrix()
                    ang_vel = rot.T @ motion.angular()
                else:
                    ang_vel = np.zeros(3)

                energy[i] = (
                    mass * np.sum(motion.linear() ** 2) + ang_vel.T @ inertia @ ang_vel
                ) / mass
                vel_integral[i] += np.sqrt(energy[i]) * dt if pose_finite else np.inf

            if (
                max(energy) < energy_thresh
                or max(vel_integral) > 0.9 * vel_thresh
            ) and t > min_t:
                extra += 1
            else:
                extra = 0

            if (
                t > max_t
                or max(vel_integral) > vel_thresh
            ):
                break

        increment("sim_t_local", t)
        return vel_integral

    @staticmethod
    def _pose_displacement(
        pose_a: np.ndarray, pose_b: np.ndarray, rotation_weight: float = 0.0
    ) -> float:
        if not np.all(np.isfinite(pose_b)):
            return float("inf")
        translation = np.linalg.norm(pose_b[:3] - pose_a[:3], ord=2)
        if rotation_weight <= 0.0:
            return float(translation)
        rot_delta = (
            Rotation.from_quat(pose_b[3:]) * Rotation.from_quat(pose_a[3:]).inv()
        )
        return float(
            translation + rotation_weight * np.linalg.norm(rot_delta.as_rotvec())
        )

    def _simulate(
        self,
        limits: SimulationLimits,
        settle_trajectory: Optional[StoneTrajectory] = None,
    ) -> Tuple[List[float], Dict[int, StoneTrajectory], bool]:
        energy = [0.0 for _ in range(self._n_stone)]
        vel_integral = [0.0 for _ in range(self._n_stone)]
        t = 0.0
        extra = 0
        extra_n_step = limits.extra_n_step
        extra_n_step *= 2 if self.n_step == self._n_stone else 1

        dt = limits.dt
        energy_thresh = limits.energy_thresh
        vel_thresh = limits.vel_thresh
        min_t = limits.min_t
        max_t = limits.max_t

        trajectory = {}
        for body_pos, idx in enumerate(self.stone_seq):
            id = self.inventory.stones[idx].id
            trajectory[id] = StoneTrajectory(id)
            body_id = self.body_id_sim[body_pos]
            pose = self.sim.state().pose(body_id).vectorized().copy()
            if np.all(np.isfinite(pose)):
                trajectory[id].add_pose(pose)
        if settle_trajectory is not None:
            target_id = self.inventory.stones[self.stone_seq[-1]].id
            trajectory[target_id] = settle_trajectory

        captured_initial = False
        while extra < extra_n_step:

            self.sim.step(dt)
            t += dt
            for i, idx in enumerate(self.stone_seq):

                pose = self.sim.state().pose(self.body_id_sim[i])
                motion = self.sim.state().motion(self.body_id_sim[i])
                pose_vec = pose.vectorized()
                pose_finite = bool(np.all(np.isfinite(pose_vec)))

                mass = self.inventory.stones[idx].mass
                inertia = self.inventory.stones[idx].inertia

                if pose_finite:
                    rot = Rotation.from_quat(pose.orientation()).as_matrix()
                    ang_vel = rot.T @ motion.angular()
                else:
                    rot = np.eye(3)
                    ang_vel = np.zeros(3)

                energy[i] = (
                    mass * np.sum(motion.linear() ** 2) + ang_vel.T @ inertia @ ang_vel
                ) / mass

                vel_integral[i] += (
                    np.sqrt(energy[i]) * dt if pose_finite else np.inf
                )
                id = self.inventory.stones[idx].id
                if pose_finite:
                    # Skip NaN/Inf poses so visualizers and downstream consumers
                    # don't see corrupted entries.
                    trajectory[id].add_pose(pose_vec.copy())
                trajectory[id].vel_integral = vel_integral[i]

            # Peak instantaneous energy at the first step reflects motion left
            # over from the phase-1 settle (0 => the new stone came to rest).
            if not captured_initial:
                self._last_settle_initial_energy = float(max(energy))
                captured_initial = True

            if max(energy) < energy_thresh and t > min_t:
                extra += 1
            else:
                extra = 0

            if (
                t > max_t
                or max(vel_integral) > vel_thresh
            ):
                break

        increment("sim_t_main", t)
        return vel_integral, trajectory, extra >= extra_n_step

    def _update_state_from_sim(self):
        for i, stone_idx in enumerate(self.stone_seq):
            pose = self.sim.state().pose(self.body_id_sim[i])
            if np.all(np.isfinite(pose.vectorized())):
                self.inventory.stones[stone_idx].pose = pose
            # else: keep the pre-sim pose (action.pose) so render / reward
            # don't see NaN-transformed meshes.

    def _extract_contact_points(self) -> List[dict]:
        if not hasattr(self.sim, "get_contact_points"):
            return []

        body_to_stone_idx = {
            int(body_id): int(stone_idx)
            for body_id, stone_idx in zip(self.body_id_sim, self.stone_seq)
        }
        records: List[dict] = []
        try:
            contact_points = self.sim.get_contact_points()
        except Exception:
            return []

        for cp in contact_points:
            try:
                id_1 = int(cp.id_1)
                id_2 = int(cp.id_2)
                records.append(
                    {
                        "id_1": id_1,
                        "id_2": id_2,
                        "stone_idx_1": body_to_stone_idx.get(id_1),
                        "stone_idx_2": body_to_stone_idx.get(id_2),
                        "is_ground_1": id_1 == int(self.plane_id),
                        "is_ground_2": id_2 == int(self.plane_id),
                        "s_1": np.asarray(cp.s_1, dtype=float).copy(),
                        "s_2": np.asarray(cp.s_2, dtype=float).copy(),
                        "normal": np.asarray(cp.normal, dtype=float).copy(),
                        "gap": float(cp.gap),
                    }
                )
            except Exception:
                continue
        return records

    def _build_state(self):
        stone_poses = {}
        for idx in self.stone_seq:
            st = self.inventory.stones[idx]
            stone_poses[st.id] = st.pose
        return State(
            stone_set=self.inventory.stone_set.copy(),
            stone_seq=self.stone_seq.copy(),
            stone_poses=stone_poses,
            trajectories=copy.deepcopy(self.trajectories),
            action_history=copy.deepcopy(self.action_history),
            terminated=self.terminated,
            failed=self.failed,
            simulation_settled=self.simulation_settled,
            contact_points=copy.deepcopy(self.contact_points),
            pose_identified_stone_ids={
                int(stone_id)
                for stone_id in self.pose_identified_stone_ids
                if int(stone_id) in stone_poses
            },
        )

    def is_done(self, state):
        return state.terminated or state.failed

    def copy(self):
        new_simulator = Simulator(self.cfg, fast=self.fast, n_threads=self.n_threads)
        new_simulator.sim, new_simulator.plane = get_diffsim(
            self.fast,
            ground_height=self._ground_height,
        )

        new_simulator.n_step = self.n_step
        new_simulator.stone_seq = self.stone_seq.copy()

        new_simulator.inventory = self.inventory.copy()

        new_simulator.sim.clear()
        new_simulator.plane_id = new_simulator.sim.add_body(new_simulator.plane)

        new_simulator.body_id_sim = []
        for idx in new_simulator.stone_seq:
            st = new_simulator.inventory.stones[idx]
            new_simulator.body_id_sim.append(new_simulator.sim.add_body(st.config))

        new_simulator.trajectories = copy.deepcopy(self.trajectories)
        new_simulator.action_history = copy.deepcopy(self.action_history)
        new_simulator.contact_points = copy.deepcopy(self.contact_points)
        new_simulator.pose_identified_stone_ids = {
            int(stone_id) for stone_id in self.pose_identified_stone_ids
        }
        new_simulator.terminated = self.terminated
        new_simulator.failed = self.failed
        new_simulator.simulation_settled = self.simulation_settled

        return new_simulator

    def update_from_state(self, state: State):
        self.n_step = len(state.stone_seq)
        self.stone_seq = state.stone_seq.copy()
        self.inventory.update_from_state(state)

        if self._simulate_pose_identified:
            identified = {
                int(stone_id)
                for stone_id in getattr(state, "pose_identified_stone_ids", set())
            }
            # ERR 0 is applied to ALL identified stones (not only the simulated
            # subset): erp of a contact pair is max over the two bodies, so
            # identified-vs-identified contacts get 0 (no penetration correction
            # inside the reconstructed scene, even against frozen neighbors)
            # while identified-vs-ordinary contacts keep the default erp.
            for stone in self.inventory.stones:
                if int(stone.id) in identified:
                    stone.config.error_reduction_ratio = 0.0

        self.sim.clear()
        self.plane_id = self.sim.add_body(self.plane)

        self.body_id_sim = []
        for idx in self.stone_seq:
            st = self.inventory.stones[idx]
            self.body_id_sim.append(self.sim.add_body(st.config))

        self.trajectories = copy.deepcopy(state.trajectories)
        self.action_history = copy.deepcopy(state.action_history)
        self.contact_points = copy.deepcopy(getattr(state, "contact_points", []))
        self.pose_identified_stone_ids = {
            int(stone_id)
            for stone_id in getattr(state, "pose_identified_stone_ids", set())
        }
        self.terminated = state.terminated
        self.failed = state.failed
        self.simulation_settled = bool(
            getattr(state, "simulation_settled", True)
        )
