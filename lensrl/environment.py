"""LensRL - Environment module.

This module defines the reinforcement learning environment for the lens system.
The environment is based on the OpenAI Gym framework and provides a way to
interact with the lens system using reinforcement learning.

Kramer Harrison, 2025
"""
import numpy as np
from gymnasium import Env
from .spaces import LensObservationSpace, LensActionSpace
from .configurable_optic import ConfigurableOptic
from .actions import OPTIMIZATION_ACTIONS, UPDATE_ACTIONS


class LensDesignEnv(Env):
    """Reinforcement learning environment for lens design."""

    def __init__(self, reward):
        self.reward = reward

        # Constants for the environment - TODO: Move to config file
        max_lenses = 1
        lens_feature_size = 6
        system_feature_size = 4
        max_optim_actions = 3
        max_update_actions = 4
        continuous_param_dim = 4
        f_number = 10
        max_steps = 25

        self.max_steps = max_steps
        self.current_step = 0

        # Define spaces
        self.observation_space = LensObservationSpace(
            max_lenses=max_lenses,
            lens_feature_size=lens_feature_size,
            system_feature_size=system_feature_size
        )
        self.action_space = LensActionSpace(
            max_optim_actions=max_optim_actions,
            max_update_actions=max_update_actions,
            max_lenses=max_lenses,
            continuous_param_dim=continuous_param_dim
        )

        # Initialize the configurable optic
        self.lens = ConfigurableOptic(f_number=f_number, max_lenses=max_lenses)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.lens.reset()
        self.current_step = 0
        return self._get_normalized_observation(), {}

    def step(self, action):
        # denormalize action
        action = self.action_space.decode(action)

        # Get optimization action
        opt_idx = action[0]
        opt_action = OPTIMIZATION_ACTIONS[opt_idx]()

        # Get update action
        update_idx = action[1]
        update_action = UPDATE_ACTIONS[update_idx]()

        # Check action validity
        is_valid = self._validate_action(action, opt_action, update_action)

        if not is_valid:
            reward = -10
            obs = self._get_normalized_observation()
            return obs, reward, False, False, {}

        # Execute actions
        self._execute_actions(action, opt_action, update_action)

        reward = self._calculate_reward()
        obs = self._get_normalized_observation()
        done = self._check_done()
        self.current_step += 1

        truncated = False
        info = {}

        return obs, reward, done, truncated, info

    def _validate_action(self, action, opt_action, update_action):
        params = action[2:]

        opt_valid = opt_action.validate(self.lens, params)
        update_valid = update_action.validate(self.lens, params)
        return opt_valid and update_valid

    def _execute_actions(self, action, opt_action, update_action):
        params = action[2:]

        # Execute optimization action
        update_action.execute(self.lens, params)

        # Execute update action
        opt_action.execute(self.lens, params)

    def _get_normalized_observation(self):
        # Generate observation
        raw_observation = self.lens.get_raw_observation()

        # Normalize observation
        obs = self.observation_space.normalize(raw_observation)

        # clip for numerical stability
        return np.clip(obs, -1, 1)

    def _calculate_reward(self):
        return self.reward(self.lens)

    def _check_done(self):
        return self.current_step >= self.max_steps
