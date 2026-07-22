import gymnasium as gym

from typing import Dict, Tuple
from omegaconf import OmegaConf

from .simulator import Simulator
from .components.reward import RewardComputer
from .components.state import State, Observation, ObservationBuilder
from .components.inventory import InventoryManager
from .components.action import ActionBuilder
from .visualization import (
    VisualInfo,
    save_configuration_figure as save_configuration_figure_image,
    visualization,
)


class StoneStackingEnv(gym.Env):
    def __init__(self, args, **kwargs):
        self.cfg: OmegaConf = args["cfg"]
        self.n_threads = args.get("n_threads")
        self.build_action_builder = bool(args.get("build_action_builder", True))

        self.simulator = Simulator(
            self.cfg,
            fast=bool(args.get("fast_sim", False)),
            n_threads=self.n_threads,
        )
        self.inventory: InventoryManager = self.simulator.inventory
        self.reward_fn = RewardComputer(self.cfg)
        self.obs_builder = ObservationBuilder(self.cfg)
        self.action_builder = (
            ActionBuilder(self.cfg, n_threads=self.n_threads)
            if self.build_action_builder
            else None
        )

    def step(
        self,
        action,
        simulate: bool = True,
        simulation_mode: str = "long",
        simulation_overrides=None,
    ) -> Tuple[State, Observation, bool, float, Dict]:
        state = self.simulator.step(
            action,
            simulate=simulate,
            simulation_mode=simulation_mode,
            simulation_overrides=simulation_overrides,
        )

        reward, info_r = self.reward_fn.compute(self.inventory, state)
        obs = self.obs_builder.build(self.inventory, state)

        done = self.simulator.is_done(state)
        return state, obs, done, reward, info_r

    def reset(self):
        state = self.simulator.reset()
        obs = self.obs_builder.build(self.inventory, state)
        return state, obs

    def copy(self):
        env_copy = StoneStackingEnv(
            {
                "cfg": self.cfg,
                "n_threads": self.n_threads,
                "build_action_builder": self.build_action_builder,
            }
        )
        env_copy.simulator = self.simulator.copy()
        env_copy.inventory = env_copy.simulator.inventory
        return env_copy

    def get_state(self):
        return self.simulator._build_state()

    def get_action(self, state, init_pose, action_idx):
        self._require_action_builder()
        return self.action_builder.get_action(
            self.inventory, state, init_pose, action_idx
        )

    def get_action_samples(
        self,
        state,
        scene_height_map=None,
        n_action_samples=None,
        sampling_priority=None,
    ):
        self._require_action_builder()
        return self.action_builder.get_action_samples(
            self.inventory,
            state,
            scene_height_map=scene_height_map,
            n_action_samples=n_action_samples,
            sampling_priority=sampling_priority,
        )

    def get_action_samples_avoiding(
        self,
        state,
        scene_height_map=None,
        n_action_samples=None,
        rejected_actions=None,
        reject_xy_radius=0.0,
        sampling_priority=None,
    ):
        self._require_action_builder()
        return self.action_builder.get_action_samples(
            self.inventory,
            state,
            scene_height_map=scene_height_map,
            n_action_samples=n_action_samples,
            rejected_actions=rejected_actions,
            reject_xy_radius=reject_xy_radius,
            sampling_priority=sampling_priority,
        )

    def is_action_supported(self, state, action):
        self._require_action_builder()
        return self.action_builder.is_action_supported(self.inventory, state, action)

    def _require_action_builder(self):
        if self.action_builder is None:
            raise RuntimeError("this environment was created without an action builder")

    def close(self):
        if self.action_builder is not None:
            self.action_builder.close()

    def update_from_state(self, state):
        self.simulator.update_from_state(state)
        self.inventory = self.simulator.inventory

    def get_observation(self):
        obs = self.obs_builder.build(self.inventory, self.simulator._build_state())
        return obs

    def visualization(self, video_filename, fps, time_scale):
        vis_info = VisualInfo(
            cfg=self.cfg,
            state=self.simulator._build_state(),
            inventory=self.inventory,
        )
        visualization(video_filename, fps, time_scale, vis_info)

    def save_configuration_figure(self, image_filename):
        vis_info = VisualInfo(
            cfg=self.cfg,
            state=self.simulator._build_state(),
            inventory=self.inventory,
        )
        save_configuration_figure_image(image_filename, vis_info)
