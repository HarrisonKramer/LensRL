"""LensRL - Normalization module.

This module defines the normalization utility for the reinforcement learning
environment. The utility is used to normalize and denormalize the parameters
of the lens system. This scales the parameters appropriately and brings them
into the range -1 to 1. This is important for the neural network, which
makes the reinforcement learning process more stable.

Kramer Harrison, 2025
"""
import numpy as np


class NormalizationUtility:
    """Normalization utility for the reinforcement learning environment."""

    def __init__(self):
        self.scaling_methods = {}

    def register_parameter(self, name, method, **kwargs):
        """Register a parameter with a specific normalization method."""
        if name in self.scaling_methods:
            raise ValueError(f"Parameter {name} already registered.")
        self.scaling_methods[name] = (method, kwargs)

    def normalize(self, name, value):
        """Normalize a value for a given parameter."""
        if name not in self.scaling_methods:
            raise ValueError(f"Parameter {name} not registered.")
        method, kwargs = self.scaling_methods[name]
        # Remove 'inverse' from kwargs before passing to the method
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('inverse', None)
        return method(value, **kwargs_copy).astype(np.float32)

    def denormalize(self, name, normalized_value):
        """Denormalize a value for a given parameter."""
        if name not in self.scaling_methods:
            raise ValueError(f"Parameter {name} not registered.")
        method, kwargs = self.scaling_methods[name]
        if "inverse" not in kwargs:
            raise ValueError(f"Inverse function not defined for "
                             f"parameter {name}.")
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('inverse', None)
        value = kwargs["inverse"](normalized_value, **kwargs_copy)
        return value.astype(np.float32)


def min_max_scaling(value, min_val, max_val):
    return 2 * (value - min_val) / (max_val - min_val) - 1


def inverse_min_max_scaling(normalized_value, min_val, max_val):
    return (normalized_value + 1) * (max_val - min_val) / 2 + min_val


def log_scaling(value, log_min, log_max):
    return 2 * (np.log(value) - log_min) / (log_max - log_min) - 1


def inverse_log_scaling(normalized_value, log_min, log_max):
    return np.exp((normalized_value + 1) * (log_max - log_min) / 2 + log_min)


def reciprocal_scaling(value, scale_factor):
    epsilon = 1e-10
    denominator = 1 + abs(value) / (scale_factor + epsilon)
    scaled_value = value / denominator

    return 2 * scaled_value - 1


def inverse_reciprocal_scaling(normalized_value, scale_factor):
    epsilon = 1e-10
    value = normalized_value / 2 + 0.5
    alpha = scale_factor + epsilon
    if normalized_value >= 0:
        return value / (1 - value / alpha)
    else:
        return value / (1 + value / alpha)


normalizer = NormalizationUtility()

normalizer.register_parameter(
    "rms_spot_size",
    log_scaling,
    log_min=np.log(1e-6), log_max=np.log(1e6),
    inverse=inverse_log_scaling
)

normalizer.register_parameter(
    "f_number",
    min_max_scaling,
    min_val=0, max_val=20,
    inverse=inverse_min_max_scaling
)

normalizer.register_parameter(
    "field_of_view",
    min_max_scaling,
    min_val=0, max_val=40,
    inverse=inverse_min_max_scaling
)

normalizer.register_parameter(
    "number_of_surfaces",
    min_max_scaling,
    min_val=0, max_val=12,
    inverse=inverse_min_max_scaling
)

normalizer.register_parameter(
    "index",
    min_max_scaling,
    min_val=1.0, max_val=2.0,
    inverse=inverse_min_max_scaling
)

normalizer.register_parameter(
    "abbe",
    min_max_scaling,
    min_val=0.0, max_val=100.0,
    inverse=inverse_min_max_scaling
)

normalizer.register_parameter(
    "radius",
    reciprocal_scaling,
    scale_factor=100,
    inverse=inverse_reciprocal_scaling
)

normalizer.register_parameter(
    "lens_thickness",
    min_max_scaling,
    min_val=1.0, max_val=10.0,
    inverse=inverse_min_max_scaling
)

normalizer.register_parameter(
    "air_thickness",
    min_max_scaling,
    min_val=0.1, max_val=100.0,
    inverse=inverse_min_max_scaling
)
