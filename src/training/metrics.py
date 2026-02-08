"""
Shared metric computation utilities.

These are pure functions (no state) so they can be used in any training
module or evaluation script without coupling to Lightning.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Compute top-k accuracy for the given *k* values.

    Args:
        output: Model logits of shape ``(batch_size, num_classes)``.
        target: Ground-truth labels of shape ``(batch_size,)``.
        topk: Tuple of *k* values to evaluate.

    Returns:
        Dictionary mapping ``"top{k}_acc"`` to the accuracy (0--1).
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results: Dict[str, float] = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results[f"top{k}_acc"] = (correct_k / batch_size).item()
    return results
