"""Input contracts shared by reference and research renderers."""

import math
from numbers import Integral, Real

import torch


def validate_count(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_bounds(near, far, *, lindisp=False):
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in (near, far)
        )
        or not 0 <= near < far
    ):
        raise ValueError("Rendering bounds must satisfy finite 0 <= near < far")
    if lindisp and near == 0:
        raise ValueError("Inverse-depth sampling requires near > 0")


def validate_rays(origins, directions, view_directions=None):
    if (
        not isinstance(origins, torch.Tensor)
        or not isinstance(directions, torch.Tensor)
        or origins.ndim != 2
        or origins.shape != directions.shape
        or origins.shape[1] != 3
        or len(origins) == 0
    ):
        raise ValueError("Expected nonempty matching ray tensors of shape N x 3")
    if (
        origins.dtype not in (torch.float32, torch.float64)
        or directions.dtype != origins.dtype
        or origins.device != directions.device
    ):
        raise ValueError(
            "Geometry rays require matching float32 or float64 dtype and device"
        )
    if not torch.isfinite(origins).all() or not torch.isfinite(directions).all():
        raise ValueError("Ray origins and directions must be finite")
    lengths = torch.linalg.vector_norm(directions, dim=-1)
    if not torch.isfinite(lengths).all() or (lengths == 0).any():
        raise ValueError("Ray directions must have finite nonzero lengths")
    if view_directions is not None:
        if (
            not isinstance(view_directions, torch.Tensor)
            or view_directions.shape != directions.shape
            or view_directions.dtype != directions.dtype
            or not torch.isfinite(view_directions).all()
        ):
            raise ValueError(
                "Viewing directions must be finite and match geometry shape and dtype"
            )
        lengths = torch.linalg.vector_norm(view_directions, dim=-1)
        if not torch.allclose(lengths, torch.ones_like(lengths), atol=1e-5, rtol=1e-5):
            raise ValueError("World viewing directions must have unit length")


def validate_sampling_options(perturb, raw_noise_std):
    for name, value in (("perturb", perturb), ("raw_noise_std", raw_noise_std)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
