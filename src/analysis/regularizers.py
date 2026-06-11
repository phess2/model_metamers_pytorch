"""
Regularizers for metamer synthesis.

Provides a total-variation (TV) smoothness loss and an Lp range
regularizer.  Both are modality-agnostic: they auto-detect whether the
input is a 2-D signal (images, ``B x C x H x W``) or a 1-D signal
(audio, ``B x C x T`` / ``B x T``) and compute the appropriate finite
differences.
"""

from __future__ import annotations

import torch
from torch import Tensor


def total_variation_loss(x: Tensor) -> Tensor:
    """L2 total-variation: sum of squared differences between neighbours.

    For 4-D tensors (images, ``B x C x H x W``) this computes horizontal
    **and** vertical squared differences.  For 3-D or 2-D tensors (audio)
    it computes temporal squared differences along the last axis.

    Parameters
    ----------
    x : Tensor
        The signal on which to compute TV.  Must be 2-D, 3-D, or 4-D.

    Returns
    -------
    Tensor
        Scalar TV value.
    """
    if x.ndim == 4:
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        return (diff_h**2).sum() + (diff_w**2).sum()

    if x.ndim in (2, 3):
        diff_t = x[..., 1:] - x[..., :-1]
        return (diff_t**2).sum()

    raise ValueError(
        f"total_variation_loss expects 2-D, 3-D, or 4-D input, got {x.ndim}-D"
    )


def range_regularizer(x: Tensor, p: int = 6) -> Tensor:
    """Lp norm of the mean-centred signal.

    .. math::

        R_\\alpha(x) = \\|x - \\bar x\\|_p

    where :math:`\\bar x = \\text{mean}(x)`.

    Parameters
    ----------
    x : Tensor
        The signal (any dimensionality).
    p : int
        Norm order.  Default ``6`` (from Mahendran & Vedaldi, 2015).

    Returns
    -------
    Tensor
        Scalar regularizer value.
    """
    return torch.norm(x - x.mean(), p=p)
