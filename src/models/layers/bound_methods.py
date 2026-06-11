"""Pluggable Lipschitz bound and projection methods for Lips layers."""

from __future__ import annotations

from typing import Callable, Dict, Protocol

import torch
from torch import Tensor

from .norm_ops import orthogonalize, power_iterate, spectral_normalize


class BoundMethod(Protocol):
    """Compute an operator-norm bound for a weight tensor."""

    def compute_bound(self, weight: Tensor, lips_weight_scale: Tensor) -> float: ...

    def project_slice(
        self, slice_weight: Tensor, scale: Tensor, w_max: float
    ) -> Tensor: ...


def linear_linf_operator_norm(weight: Tensor) -> Tensor:
    """Induced L-infinity operator norm: max row sum of absolute values."""
    return weight.abs().sum(dim=-1).max()


def modular_linf_conv_bound(weight: Tensor, lips_weight_scale: Tensor) -> float:
    """
    Modular L-infinity bound for Conv2d weights.

    Takes the maximum induced L-infinity norm over per-kernel-position slices,
    matching the modular decomposition used for RMS spectral bounds.
    """
    _, _, kh, kw = weight.shape
    max_val = float("-inf")
    for i in range(kh):
        for j in range(kw):
            slice_ij = weight[:, :, i, j] / lips_weight_scale
            norm_ij = linear_linf_operator_norm(slice_ij)
            max_val = max(max_val, float(norm_ij))
    return max_val


def modular_linf_linear_bound(weight: Tensor, lips_weight_scale: Tensor) -> float:
    """Modular L-infinity bound for a linear layer."""
    normalized = weight / lips_weight_scale
    return float(linear_linf_operator_norm(normalized))


def rms_spectral_linear_bound(weight: Tensor, lips_weight_scale: Tensor) -> float:
    normalized = weight / lips_weight_scale
    _, sigma_max, _ = power_iterate(normalized)
    return float(sigma_max)


def rms_spectral_conv_bound(weight: Tensor, lips_weight_scale: Tensor) -> float:
    _, _, kh, kw = weight.shape
    max_val = float("-inf")
    for i in range(kh):
        for j in range(kw):
            slice_ij = weight[:, :, i, j] / lips_weight_scale
            _, sigma_max, _ = _power_iterate(slice_ij)
            max_val = max(max_val, float(sigma_max))
    return max_val


def _project_linf_cap(slice_weight: Tensor, scale: Tensor, w_max: float) -> Tensor:
    """Scale a weight slice so its L-infinity operator norm is at most ``w_max``."""
    unscaled = slice_weight / scale
    norm = linear_linf_operator_norm(unscaled)
    if float(norm) <= w_max:
        return slice_weight
    return unscaled * (w_max / (norm + 1e-12)) * scale


def _project_spectral_normalize(
    slice_weight: Tensor, scale: Tensor, w_max: float
) -> Tensor:
    del w_max
    unscaled = slice_weight / scale
    return spectral_normalize(unscaled) * scale


def _project_orthogonalize(slice_weight: Tensor, scale: Tensor, w_max: float) -> Tensor:
    del w_max
    unscaled = slice_weight / scale
    return orthogonalize(unscaled) * scale


class _RmsSpectralBound:
    linear_bound = staticmethod(rms_spectral_linear_bound)
    conv_bound = staticmethod(rms_spectral_conv_bound)
    project_slice = staticmethod(_project_spectral_normalize)


class _ModularLinfBound:
    linear_bound = staticmethod(modular_linf_linear_bound)
    conv_bound = staticmethod(modular_linf_conv_bound)
    project_slice = staticmethod(_project_linf_cap)


BOUND_METHODS: Dict[str, BoundMethod] = {
    "rms_spectral": _RmsSpectralBound(),
    "modular_linf": _ModularLinfBound(),
}

PROJECTION_ALIASES: Dict[str, str] = {
    "spectral_normalize": "rms_spectral",
    "orthogonalize": "rms_spectral",
    "modular_linf_cap": "modular_linf",
    "linf_cap": "modular_linf",
}


def resolve_bound_method(
    bound_method: str | None, projection: str | None
) -> str:
    """Pick the bound method from explicit config or projection alias."""
    if bound_method is not None:
        if bound_method not in BOUND_METHODS:
            raise ValueError(
                f"Unknown bound_method '{bound_method}'. "
                f"Available: {sorted(BOUND_METHODS)}"
            )
        return bound_method
    if projection in PROJECTION_ALIASES:
        return PROJECTION_ALIASES[projection]
    return "rms_spectral"


def get_projection_fn(projection: str | None) -> Callable[[Tensor, Tensor, float], Tensor]:
    if projection in (None, "null"):
        raise ValueError("projection is disabled")
    if projection == "spectral_normalize":
        return _project_spectral_normalize
    if projection == "orthogonalize":
        return _project_orthogonalize
    if projection in ("modular_linf_cap", "linf_cap"):
        return _project_linf_cap
    raise ValueError(f"Unknown projection '{projection}'")
