from dataclasses import dataclass
from math import radians
from typing import Optional


def _cfg_get(cfg, key: str, default=None):
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _cfg_section(cfg, key: str):
    value = _cfg_get(cfg, key, {})
    return {} if value is None else value


def _env_config_from_source(source):
    cfg = getattr(source, "cfg", source)
    env_cfg = _cfg_get(cfg, "environment", None)
    has_env_shape = (
        _cfg_get(cfg, "action", None) is not None
        or _cfg_get(cfg, "reward", None) is not None
    )
    return env_cfg if env_cfg is not None and not has_env_shape else cfg


def _optional_float(value) -> Optional[float]:
    return None if value is None else float(value)


@dataclass(frozen=True)
class SupportConfig:
    env: object
    action: object
    constraint: object
    reward: object

    @classmethod
    def from_source(cls, source) -> "SupportConfig":
        env = _env_config_from_source(source)
        action = _cfg_section(env, "action")
        reward_root = _cfg_section(env, "reward")
        return cls(
            env=env,
            action=action,
            constraint=_cfg_section(action, "support_constraint"),
            reward=_cfg_section(reward_root, "support"),
        )

    @property
    def min_sources(self) -> int:
        return int(self.constraint.get("min_supports", 2))

    @property
    def desired_sources(self) -> int:
        return int(self.reward.get("desired_sources", self.min_sources))

    @property
    def ground_z(self) -> float:
        return float(self.reward.get("ground_z", 0.0))

    @property
    def contact_gap_tolerance(self) -> float:
        return float(self.reward.get("contact_gap_tolerance", 0.05))

    @property
    def xy_factor(self) -> float:
        return float(self.constraint.get("xy_factor", 0.75))

    @property
    def score_xy_factor(self) -> float:
        return float(
            self.reward.get("xy_factor", self.constraint.get("xy_factor", 0.75))
        )

    @property
    def score_z_tolerance(self) -> float:
        return float(
            self.reward.get("z_tolerance", self.constraint.get("z_tolerance", 0.20))
        )

    def pre_pose_xy_factor(self, fallback_to_xy_factor: bool = True) -> float:
        fallback = (
            self.constraint.get("xy_factor", 1.0) if fallback_to_xy_factor else 1.0
        )
        return float(self.constraint.get("pre_pose_xy_factor", fallback))

    @property
    def z_tolerance(self) -> float:
        return float(self.constraint.get("z_tolerance", 0.20))

    @property
    def pair_z_tolerance(self) -> float:
        return float(self.constraint.get("pair_z_tolerance", 0.35))

    @property
    def pair_distance_scale(self) -> float:
        return float(self.constraint.get("pair_distance_scale", 1.25))

    @property
    def enabled(self) -> bool:
        return bool(self.constraint.get("enabled", False))

    @property
    def use_posegen_contacts(self) -> bool:
        return bool(self.constraint.get("use_posegen_contacts", True))

    @property
    def contact_fallback_to_geometry(self) -> bool:
        return bool(self.constraint.get("contact_fallback_to_geometry", True))

    @property
    def contact_z_tolerance(self) -> float:
        return float(self.constraint.get("contact_z_tolerance", self.z_tolerance))

    @property
    def contact_xy_margin(self) -> float:
        return float(self.constraint.get("contact_xy_margin", 0.04))

    @property
    def contact_match_tolerance(self) -> float:
        return float(self.constraint.get("contact_match_tolerance", 0.03))

    @property
    def allow_single_support(self) -> bool:
        return bool(self.constraint.get("allow_single_support", False))

    @property
    def large_below_volume_ratio(self) -> float:
        return float(self.constraint.get("large_below_volume_ratio", 1.5))

    @property
    def spread_check(self):
        return _cfg_section(self.constraint, "spread_check")

    @property
    def spread_check_enabled(self) -> bool:
        return bool(self.spread_check.get("enabled", True))

    @property
    def spread_min_separation_scale(self) -> float:
        return float(self.spread_check.get("min_separation_scale", 0.65))

    @property
    def spread_max_line_distance_scale(self) -> float:
        return float(self.spread_check.get("max_line_distance_scale", 0.45))

    @property
    def spread_min_area_scale(self) -> float:
        return float(self.spread_check.get("min_area_scale", 0.15))

    @property
    def spread_com_margin(self) -> float:
        return float(self.spread_check.get("com_margin", -0.05))

    @property
    def stable_single_support_fallback(self):
        return _cfg_section(self.constraint, "stable_single_support_fallback")

    @property
    def stable_single_support_fallback_enabled(self) -> bool:
        return bool(self.stable_single_support_fallback.get("enabled", False))

    @property
    def stable_single_support_max_displacement(self) -> float:
        return float(
            self.stable_single_support_fallback.get("max_place_displacement", 0.12)
        )

    @property
    def pre_pose_filter(self) -> bool:
        return bool(self.constraint.get("pre_pose_filter", True))

    @property
    def pre_pose_activate_after(self) -> int:
        return int(self.constraint.get("pre_pose_activate_after", self.min_sources))

    @property
    def pre_pose_fallback_to_unfiltered(self) -> bool:
        return bool(self.constraint.get("pre_pose_fallback_to_unfiltered", True))

    @property
    def pre_pose_allow_empty_space(self) -> bool:
        return bool(self.constraint.get("pre_pose_allow_empty_space", False))

    @property
    def empty_space_distance_scale(self) -> float:
        return float(self.constraint.get("empty_space_distance_scale", 1.0))

    @property
    def frontier_fill(self):
        return _cfg_section(self.constraint, "frontier_fill")

    @property
    def frontier_fill_enabled(self) -> bool:
        return bool(self.frontier_fill.get("enabled", True))

    @property
    def frontier_fill_min_support_count(self) -> int:
        return int(self.frontier_fill.get("min_support_count", 1))

    @property
    def frontier_fill_max_support_count(self) -> int:
        return int(self.frontier_fill.get("max_support_count", 1))

    @property
    def frontier_fill_boundary_only(self) -> bool:
        return bool(self.frontier_fill.get("boundary_only", True))

    @property
    def frontier_fill_boundary_inset_radius_scale(self) -> Optional[float]:
        return _optional_float(
            self.frontier_fill.get("boundary_inset_radius_scale", None)
        )

    @property
    def connected_ground(self):
        return _cfg_section(self.constraint, "connected_ground")

    @property
    def connected_ground_enabled(self) -> bool:
        return bool(self.connected_ground.get("enabled", True))

    @property
    def connected_ground_activate_after(self) -> int:
        return int(self.connected_ground.get("activate_after", 5))

    @property
    def connected_ground_max_xy_gap(self) -> float:
        return float(self.connected_ground.get("max_xy_gap", 0.10))

    @property
    def connected_ground_xy_factor(self) -> float:
        return float(self.connected_ground.get("xy_factor", 1.0))

    @property
    def connected_ground_allow_lower_fill(self) -> bool:
        return bool(self.connected_ground.get("allow_lower_fill", True))


@dataclass(frozen=True)
class MeshFitConfig:
    cfg: object
    support: SupportConfig

    @classmethod
    def from_source(cls, source) -> "MeshFitConfig":
        env = _env_config_from_source(source)
        action = _cfg_section(env, "action")
        return cls(
            cfg=_cfg_section(action, "mesh_face_fit"),
            support=support_config(source),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    @property
    def min_support_bodies(self) -> int:
        return int(self.cfg.get("min_support_bodies", self.support.min_sources))

    @property
    def activate_after(self) -> int:
        return int(self.cfg.get("activate_after", self.min_support_bodies))

    @property
    def max_support_faces(self) -> int:
        return int(self.cfg.get("max_support_faces", 12))

    @property
    def max_faces_per_support(self) -> int:
        return int(self.cfg.get("max_faces_per_support", 4))

    @property
    def preselect_faces_per_support(self) -> int:
        return int(self.cfg.get("preselect_faces_per_support", 64))

    @property
    def support_search_radius(self) -> float:
        return float(self.cfg.get("support_search_radius", 0.75))

    @property
    def support_min_normal_z(self) -> float:
        return float(self.cfg.get("support_min_normal_z", 0.55))

    @property
    def support_max_z_gap(self) -> float:
        return float(self.cfg.get("support_max_z_gap", 0.40))

    @property
    def target_min_down_z(self) -> float:
        return float(self.cfg.get("target_min_down_z", 0.45))

    @property
    def max_target_faces(self) -> int:
        return int(self.cfg.get("max_target_faces", 8))

    @property
    def max_rotation_deg(self) -> float:
        return float(self.cfg.get("max_rotation_deg", 25.0))

    @property
    def max_rotation_rad(self) -> float:
        return float(radians(self.max_rotation_deg))

    @property
    def max_lower_protrusion(self) -> float:
        return float(self.cfg.get("max_lower_protrusion", 0.08))

    @property
    def clearance(self) -> float:
        return float(self.cfg.get("clearance", 0.015))

    @property
    def xy_regularization(self) -> float:
        return float(self.cfg.get("xy_regularization", 0.25))

    @property
    def area_weight(self) -> float:
        return float(self.cfg.get("area_weight", 0.05))

    @property
    def max_pre_icp_xy_translation(self) -> Optional[float]:
        return _optional_float(self.cfg.get("max_pre_icp_xy_translation", None))

    @property
    def pre_icp_patch_centroid_snap(self) -> bool:
        return bool(self.cfg.get("pre_icp_patch_centroid_snap", True))

    @property
    def max_pre_icp_patch_translation(self) -> Optional[float]:
        return _optional_float(self.cfg.get("max_pre_icp_patch_translation", None))

    @property
    def pre_icp_contact_snap(self) -> bool:
        return bool(self.cfg.get("pre_icp_contact_snap", True))

    @property
    def pre_icp_contact_distance(self) -> float:
        default = 0.5 * self.icp_max_correspondence_distance
        value = float(self.cfg.get("pre_icp_contact_distance", default))
        return float(
            max(min(value, max(self.icp_max_correspondence_distance * 0.95, 0.0)), 0.0)
        )

    @property
    def max_pre_icp_contact_translation(self) -> Optional[float]:
        value = self.cfg.get("max_pre_icp_contact_translation", None)
        if value is None:
            value = self.max_pre_icp_patch_translation
        return _optional_float(value)

    @property
    def icp_enabled(self) -> bool:
        return bool(self.cfg.get("icp_enabled", True))

    @property
    def icp_estimation(self) -> str:
        return str(self.cfg.get("icp_estimation", "point_to_point"))

    @property
    def icp_samples_per_face(self) -> int:
        return int(self.cfg.get("icp_samples_per_face", 6))

    @property
    def icp_max_correspondence_distance(self) -> float:
        return float(self.cfg.get("icp_max_correspondence_distance", 0.12))

    @property
    def icp_max_iteration(self) -> int:
        return int(self.cfg.get("icp_max_iteration", 12))

    @property
    def icp_min_fitness(self) -> float:
        return float(self.cfg.get("icp_min_fitness", 1e-6))

    @property
    def icp_max_translation(self) -> float:
        return float(self.cfg.get("icp_max_translation", 0.12))

    @property
    def icp_max_xy_translation(self) -> Optional[float]:
        return _optional_float(self.cfg.get("icp_max_xy_translation", None))

    @property
    def icp_max_rotation_deg(self) -> float:
        return float(self.cfg.get("icp_max_rotation_deg", 10.0))

    @property
    def icp_max_rotation_rad(self) -> float:
        return float(radians(self.icp_max_rotation_deg))

    @property
    def icp_fitness_weight(self) -> float:
        return float(self.cfg.get("icp_fitness_weight", 0.5))

    @property
    def icp_rmse_weight(self) -> float:
        return float(self.cfg.get("icp_rmse_weight", 1.0))


DEFAULT_DEEPER_SAMPLING_PRIORITY = "lower_floor_first"
DEFAULT_PRIORITY_CHILD_DEPTH = 2
DEFAULT_POSEGEN_PRUNE_ENABLED = True


def _mcts_config_from_source(source):
    cfg = getattr(source, "cfg", source)
    algorithm = _cfg_get(cfg, "algorithm", None)
    if algorithm is not None and _cfg_get(algorithm, "mcts", None) is not None:
        return _cfg_get(algorithm, "mcts")
    return cfg


@dataclass(frozen=True)
class MCTSSamplingConfig:
    mcts: object
    action_generation: object
    posegen_prune: object

    @classmethod
    def from_source(cls, source) -> "MCTSSamplingConfig":
        cfg = _mcts_config_from_source(source)
        action_generation = _cfg_section(cfg, "action_generation")
        return cls(
            mcts=cfg,
            action_generation=action_generation,
            posegen_prune=_cfg_section(action_generation, "posegen_prune"),
        )

    @property
    def deeper_sampling_priority(self) -> Optional[str]:
        value = self.action_generation.get(
            "deeper_sampling_priority",
            DEFAULT_DEEPER_SAMPLING_PRIORITY,
        )
        return None if value is None else str(value)

    @property
    def priority_child_depth(self) -> int:
        return int(
            self.action_generation.get(
                "priority_child_depth",
                DEFAULT_PRIORITY_CHILD_DEPTH,
            )
        )

    @property
    def posegen_prune_enabled(self) -> bool:
        return bool(self.posegen_prune.get("enabled", DEFAULT_POSEGEN_PRUNE_ENABLED))

    def posegen_prune_threshold(self, env=None) -> Optional[float]:
        value = self.posegen_prune.get("threshold", None)
        if value is None and env is not None:
            value = _cfg_get(_cfg_get(env.cfg, "reward", {}), "posegen_thresh", None)
        if value is None:
            return None
        return float(value)

@dataclass(frozen=True)
class MCTSActionGenerationConfig:
    mcts: object
    action_generation: object
    execution_rejection: object

    @classmethod
    def from_source(cls, source) -> "MCTSActionGenerationConfig":
        cfg = _mcts_config_from_source(source)
        action_generation = _cfg_section(cfg, "action_generation")
        return cls(
            mcts=cfg,
            action_generation=action_generation,
            execution_rejection=_cfg_section(
                action_generation,
                "execution_rejection",
            ),
        )

    @property
    def batch_size(self) -> int:
        return max(int(self.action_generation.get("batch_size", 1)), 1)

    @property
    def expansion_retry_batches(self) -> int:
        return max(int(self.action_generation.get("expansion_retry_batches", 3)), 1)

    @property
    def min_feasible_children(self) -> int:
        return max(int(self.action_generation.get("min_feasible_children", 3)), 1)

    @property
    def attempt_multiplier(self) -> float:
        return max(float(self.action_generation.get("attempt_multiplier", 3.0)), 1.0)

    @property
    def root_sampling_priority(self) -> Optional[str]:
        value = self.action_generation.get("root_sampling_priority", None)
        return None if value is None or value == "" else str(value)

    def root_sampling_priority_for_state(self, state) -> Optional[str]:
        priority = self.root_sampling_priority
        if priority is None:
            return None
        if state is None or len(state.stone_seq) == 0:
            return None
        return priority

@dataclass(frozen=True)
class MCTSRootProposalConfig:
    mcts: object
    root: object

    @classmethod
    def from_source(cls, source) -> "MCTSRootProposalConfig":
        cfg = _mcts_config_from_source(source)
        root = _cfg_section(cfg, "root_proposal")
        return cls(
            mcts=cfg,
            root=root,
        )

    def root_keep(self, fallback: int) -> int:
        return int(self.root.get("keep", fallback))

    def population(self, root_keep: int) -> int:
        return int(self.root.get("population", max(root_keep, 16)))

    def elite(self, population: int) -> int:
        return int(self.root.get("elite", max(1, population // 4)))

    @property
    def iterations(self) -> int:
        return int(self.root.get("iterations", 3))

    @property
    def mutation_xy_std(self) -> float:
        return float(self.root.get("mutation_xy_std", 0.12))

    @property
    def mutation_xy_mode(self) -> str:
        return str(self.root.get("mutation_xy_mode", "gaussian"))

    @property
    def mutation_yaw_std_rad(self) -> float:
        return float(radians(float(self.root.get("mutation_yaw_std_deg", 20.0))))

    @property
    def fresh_fraction(self) -> float:
        return float(self.root.get("fresh_fraction", 0.0))

    @property
    def max_per_stone(self) -> Optional[int]:
        value = self.root.get("max_per_stone", None)
        return None if value is None else int(value)

    @property
    def height_map_score_weight(self) -> float:
        return float(self.root.get("height_map_score_weight", 1.0))


@dataclass(frozen=True)
class MCTSValidationConfig:
    mcts: object
    short: object
    long: object
    validation: object

    @classmethod
    def from_source(cls, source) -> "MCTSValidationConfig":
        cfg = _mcts_config_from_source(source)
        simulation = _cfg_section(cfg, "simulation")
        short = _cfg_section(simulation, "short")
        long = _cfg_section(simulation, "long")
        validation = _cfg_section(cfg, "validation")

        # Old saved configs nested both stages under root_proposal.final_sim.
        root = _cfg_section(cfg, "root_proposal")
        legacy = _cfg_section(root, "final_sim")
        if not legacy:
            legacy = _cfg_section(root, "final_simulation")
        if not short:
            short = _cfg_section(legacy, "short_simulation")
        if not long:
            long = _cfg_section(legacy, "validation")
        if not long:
            long = _cfg_section(cfg, "final_validation")
        if legacy:
            inherited = dict(legacy)
            inherited.update(dict(long))
            long = inherited
        if not validation:
            validation = long
        return cls(
            mcts=cfg,
            short=short,
            long=long,
            validation=validation,
        )

    @property
    def legacy_short_profile(self) -> dict:
        return self._profile(self.short, full_dynamic=False)

    @property
    def legacy_long_profile(self) -> dict:
        return self._profile(self.long, full_dynamic=self.long_full_dynamic)

    @property
    def max_candidates(self) -> Optional[int]:
        value = self.validation.get("max_candidates", None)
        return None if value is None else max(int(value), 0)

    @property
    def debug_extra_candidates(self) -> int:
        return max(int(self.validation.get("debug_extra_candidates", 0)), 0)

    @property
    def max_place_displacement(self) -> Optional[float]:
        return _optional_float(
            self.validation.get("max_place_displacement", None)
        )

    @property
    def long_full_dynamic(self) -> bool:
        return bool(self.long.get("full_dynamic", True))

    def worker_threads(self, env_args: dict) -> int:
        n_threads = self.validation.get("n_threads", env_args.get("n_threads", 1))
        n = max(int(n_threads or 1), 1)
        try:
            cpu_cap = int(env_args["cfg"].resource.num_cpus) - 1
            if cpu_cap > 0:
                n = min(n, cpu_cap)
        except Exception:
            pass
        return n

    @staticmethod
    def _profile(cfg, full_dynamic: bool) -> dict:
        profile = {
            key: cfg[key]
            for key in (
                "max_t",
                "min_t",
                "extra_n_step",
                "energy_thresh",
                "vel_integral_thresh",
                "freeze_radius",
            )
            if cfg.get(key, None) is not None
        }
        if full_dynamic:
            profile["freeze_radius"] = 0
        return profile


def support_config(source) -> SupportConfig:
    return SupportConfig.from_source(source)


def mcts_sampling_config(source) -> MCTSSamplingConfig:
    return MCTSSamplingConfig.from_source(source)


def mcts_action_generation_config(source) -> MCTSActionGenerationConfig:
    return MCTSActionGenerationConfig.from_source(source)


def mesh_fit_config(source) -> MeshFitConfig:
    return MeshFitConfig.from_source(source)


def mcts_root_proposal_config(source) -> MCTSRootProposalConfig:
    return MCTSRootProposalConfig.from_source(source)


def mcts_validation_config(source) -> MCTSValidationConfig:
    return MCTSValidationConfig.from_source(source)
