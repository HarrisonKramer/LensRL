"""LensRL - Spaces module.

This module defines the observation and action spaces for the lens environment.
The observation space is a Box space that contains the parameters of the lens
system. The action space is a Box space that contains the parameters of the
action that can be performed on the lens system.

Kramer Harrison, 2025
"""
import numpy as np
from gymnasium import spaces
from .normalization import normalizer
from .glass import VALID_GLASSES


class LensObservationSpace(spaces.Box):
    """
    - Max lenses = 5 (default)
    - Lens feature size = 6 (index, Abbe number, radius0, thickness0, radius1, thickness1)
    - System feature size = 4 (RMS spot size, f-number, field of view, number surfaces)
    - bounds = tuple of [min, max] for RMS spot size, f-number, field of view, number of surfaces, refractive index,
        V-number, radius, lens thickness, air gap thickness
    """
    def __init__(self,
                 normalizer=normalizer,
                 max_lenses=5,
                 lens_feature_size=6,
                 system_feature_size=4):
        self.normalizer = normalizer
        self.max_lenses = max_lenses
        self.lens_feature_size = lens_feature_size
        self.system_feature_size = system_feature_size

        self.parameters = ['rms_spot_size', 'f_number', 'field_of_view', 'number_of_surfaces'] + \
            ['index', 'abbe', 'radius', 'lens_thickness', 'radius', 'air_thickness'] * max_lenses

        # Define shape of the flattened observation
        shape = (self.system_feature_size + self.max_lenses * self.lens_feature_size,)
        super().__init__(low=-1.0, high=1.0, shape=shape, dtype=np.float32)

    def normalize(self, raw_observation):
        normalized = []
        for param_name, value in zip(self.parameters, raw_observation):
            normalized.append(self.normalizer.normalize(param_name, value))
        normalized = np.array(normalized)

        # mask
        num_lenses = (raw_observation[3] - 2) // 2
        idx = int(self.system_feature_size + num_lenses * self.lens_feature_size)
        normalized[idx:] = 0

        return normalized

    def denormalize(self, observation):
        raw = []
        for param_name, value in zip(self.parameters, normalized_observation):
            raw.append(self.normalizer.denormalize(param_name, value))
        denormalized = np.array(raw)

        # mask
        num_lenses = (denormalized[3] - 2) // 2
        idx = int(self.system_feature_size + num_lenses * self.lens_feature_size)
        denormalized[idx:] = 0

        return denormalized


class LensActionSpace(spaces.Box):
    """
    Custom action space for the lens environment.

    Definition:
        - 0: optimization action (0, 1, or 2)
        - 1: update action (0, 1, 2, or 3)
        - 2: lens index (discrete)
        - 3: surface index (discrete)
        - 4: material index (discrete)
        - 5: radius 0 (continuous)
        - 6: radius 1 (continuous)
        - 7: thickness 0 (continuous)
        - 8: thickness 1 (continuous)
    """
    def __init__(self,
                 normalizer=normalizer,
                 max_optim_actions=3,
                 max_update_actions=4,
                 max_lenses=5,
                 max_materials=len(VALID_GLASSES),
                 continuous_param_dim=4,
                 params=['radius', 'radius', 'lens_thickness', 'air_thickness']):
        """
        Shape = 5 + continuous_param_dim
        """
        shape = (5 + continuous_param_dim,)
        super().__init__(low=-1, high=1, shape=shape, dtype=np.float32)

        max_surfaces = 2 * max_lenses + 2

        self.normalizer = normalizer
        self.max_optim_actions = max_optim_actions
        self.max_update_actions = max_update_actions
        self.max_lenses = max_lenses
        self.max_surfaces = max_surfaces
        self.max_materials = max_materials
        self.continuous_param_dim = continuous_param_dim
        self.params = params

    def decode(self, action: np.ndarray):
        """
        Decode and denormalize the action into format consisting of integers and floats.
        """
        discrete_actions = [int((action[0] + 1) / 2 * (self.max_optim_actions - 1)),
                            int((action[1] + 1) / 2 * (self.max_update_actions - 1)),
                            int((action[2] + 1) / 2 * (self.max_lenses - 1)),
                            int((action[3] + 1) / 2 * (self.max_surfaces - 1)),
                            int((action[4] + 1) / 2 * (self.max_materials - 1))]

        idx = self.continuous_param_dim
        continuous_actions = [self.normalizer.denormalize(param, val) for param, val in zip(self.params, action[-idx:])]
        action = tuple(discrete_actions) + tuple(continuous_actions)
        return action
