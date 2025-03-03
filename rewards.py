"""LensRL - Rewards module.

This module defines the rewards that can be used in the reinforcement learning
environment. The rewards are used to guide the optimization process and
encourage the system to converge to a desired solution.

Kramer Harrison, 2025
"""
import warnings
import numpy as np
from optiland import analysis


class BaseReward:
    def __call__(self, lens):
        raise NotImplementedError("Subclasses must implement __call__")


class RMSReward(BaseReward):
    def __init__(self, weight=1.0, include_delta=True, delta_weight=1.0):
        self.weight = weight
        self.include_delta = include_delta
        self.delta_weight = delta_weight

        self.prev_rms = None
        self.nan_penalty = -1e3

    def __call__(self, lens):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spot = analysis.SpotDiagram(lens)
            rms = np.mean(np.abs(spot.rms_spot_radius()))
            lens.rms = rms  # save for RMS state
            reward = -self.weight * rms  # Negative for minimization

        if np.isnan(rms):
            rms = 1e6
            lens.rms = rms  # save for RMS state
            return self.nan_penalty

        if self.include_delta:
            if self.prev_rms is not None:
                delta = self.prev_rms - rms
                reward += delta * self.delta_weight
            self.prev_rms = rms

        return reward


class ComplexityReward(BaseReward):
    def __init__(self, weight=0.1):
        self.weight = weight

    def __call__(self, lens):
        return -self.weight * lens.complexity()


class ApertureFOVReward(BaseReward):
    def __init__(self, aperture_weight=1.0, fov_weight=1.0):
        self.aperture_weight = aperture_weight
        self.fov_weight = fov_weight

    def __call__(self, lens):
        ap_reward = -self.aperture_weight * lens.aperture.value
        fov_reward = self.fov_weight * lens.field_of_view
        return ap_reward + fov_reward


class CompletionReward(BaseReward):
    def __init__(self, target_fov, target_rms=0.0, weight=1.0):
        self.target_fov = target_fov
        self.target_rms = target_rms
        self.weight = weight

    def __call__(self, lens):
        fov_reward = np.abs(self.target_fov - lens.field_of_view)
        rms_reward = np.abs(self.target_rms - lens.rms)
        return -self.weight * (fov_reward + rms_reward)


class CompositeReward(BaseReward):
    def __init__(self, rewards):
        self.rewards = rewards

    def __call__(self, lens):
        return sum(reward(lens) for reward in self.rewards)
