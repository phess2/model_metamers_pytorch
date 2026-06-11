from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .LipsLayers import _power_iterate


def product_bound(bounds: Iterable[float]) -> float:
    """Multiply a sequence of per-operator bounds."""
    product = 1.0
    for bound in bounds:
        product *= float(bound)
    return product


def linear_rms_lips_bound(module: nn.Linear) -> float:
    """RMS->RMS bound for a plain linear layer."""
    weight = module.weight.data
    scale = torch.sqrt(
        torch.tensor(
            module.out_features / module.in_features,
            device=weight.device,
            dtype=weight.dtype,
        )
    )
    normalized = weight / scale
    _, sigma_max, _ = _power_iterate(normalized)
    return float(sigma_max)


def conv2d_rms_lips_bound(module: nn.Conv2d) -> float:
    """RMS->RMS bound for a plain Conv2d via max per-kernel-slice norm."""
    weight = module.weight.data
    out_channels, _in_channels_per_group, kh, kw = weight.shape
    scale = torch.sqrt(
        torch.tensor(
            out_channels / module.in_channels,
            device=weight.device,
            dtype=weight.dtype,
        )
    )
    scale /= kh * kw

    max_val = float("-inf")
    for i in range(kh):
        for j in range(kw):
            slice_ij = weight[:, :, i, j] / scale
            _, sigma_max, _ = _power_iterate(slice_ij)
            max_val = max(max_val, float(sigma_max))
    return max_val


def batchnorm2d_lips_bound(module: nn.BatchNorm2d) -> float:
    """
    RMS->RMS bound for BatchNorm2d in eval mode.

    The affine map is channel-wise diagonal with gain
    gamma / sqrt(running_var + eps), so the induced 2-norm is the maximum
    absolute channel gain.
    """
    if module.running_var is None:
        denom = torch.sqrt(torch.tensor(module.eps, device=module.weight.device))
    else:
        denom = torch.sqrt(module.running_var + module.eps)

    if module.affine and module.weight is not None:
        gain = module.weight.abs() / denom
    else:
        gain = torch.ones_like(denom) / denom
    return float(torch.max(gain.detach()))


def input_normalize_lips_bound(new_std: torch.Tensor) -> float:
    """RMS->RMS bound for channel-wise division by std; clamp is 1-Lipschitz."""
    std = torch.clamp(new_std.detach().abs(), min=1e-12)
    gain = 1.0 / std
    return float(torch.max(gain))
