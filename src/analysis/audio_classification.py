"""Helpers for AudioSet-style multi-label audio classification evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch


def decode_audioset_labels(example: Mapping[str, Any]) -> list[int]:
    """Decode AudioSet label indices from a TFRecord example dict."""
    labels_value = example.get("labels")
    if labels_value is None:
        return []
    if hasattr(labels_value, "tolist"):
        return [int(x) for x in labels_value.tolist()]
    if isinstance(labels_value, (list, tuple)):
        return [int(x) for x in labels_value]
    return [int(labels_value)]


def multihot_topk_hit(topk_indices: Sequence[int], true_labels: Sequence[int]) -> bool:
    """Return True when any true label appears in the predicted top-k set."""
    true_label_set = set(int(x) for x in true_labels)
    if not true_label_set:
        return False
    return any(int(pred_idx) in true_label_set for pred_idx in topk_indices)


def summarize_multilabel_logits(
    logits: torch.Tensor,
    true_labels: Sequence[int],
) -> Dict[str, Any]:
    """
    Summarize top-k predictions for one sample with multi-label ground truth.

    Returns values suitable for CSV/JSON persistence.
    """
    if logits.dim() == 2:
        if logits.shape[0] != 1:
            raise ValueError(f"Expected a single-sample batch, got logits shape {tuple(logits.shape)}")
        logits = logits[0]
    if logits.dim() != 1:
        raise ValueError(f"Expected 1D logits tensor, got shape {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=0)
    num_classes = int(logits.shape[0])
    top1_k = 1
    top5_k = min(5, num_classes)

    top1_indices = torch.topk(probs, k=top1_k).indices.tolist()
    top5_indices = torch.topk(probs, k=top5_k).indices.tolist()
    top1_pred = int(top1_indices[0])

    true_labels_int = [int(x) for x in true_labels]
    true_label_max_softmax = (
        float(probs[true_labels_int].max().item()) if true_labels_int else float("nan")
    )

    return {
        "predicted_label_idx": top1_pred,
        "predicted_top5_label_indices": [int(x) for x in top5_indices],
        "top1_hit": multihot_topk_hit(top1_indices, true_labels_int),
        "top5_hit": multihot_topk_hit(top5_indices, true_labels_int),
        "true_label_max_softmax": true_label_max_softmax,
        "num_true_labels": len(true_labels_int),
    }


def aggregate_multilabel_topk(sample_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate top-1/top-5 hit rates over sample summaries."""
    if not sample_summaries:
        return {
            "top1_acc": 0.0,
            "top5_acc": 0.0,
            "num_samples": 0,
            "num_scored_samples": 0,
        }

    scored_samples = [
        summary
        for summary in sample_summaries
        if "top1_hit" in summary and "top5_hit" in summary
    ]
    if not scored_samples:
        return {
            "top1_acc": 0.0,
            "top5_acc": 0.0,
            "num_samples": len(sample_summaries),
            "num_scored_samples": 0,
        }

    top1_hits = sum(1 for summary in scored_samples if bool(summary["top1_hit"]))
    top5_hits = sum(1 for summary in scored_samples if bool(summary["top5_hit"]))
    denom = float(len(scored_samples))
    return {
        "top1_acc": top1_hits / denom,
        "top5_acc": top5_hits / denom,
        "num_samples": len(sample_summaries),
        "num_scored_samples": len(scored_samples),
    }
