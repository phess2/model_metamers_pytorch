"""
Saving utilities for metamer generation results.

Handles saving metamer tensors, original/metamer images (for visual
inspection), and JSONL metadata (one file per layer).  Modality-aware: for
vision, saves ``.png`` files; audio support can be added later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import torch
from torch import Tensor

# ImageNet de-normalisation constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _denormalize_imagenet(tensor: Tensor) -> Tensor:
    """Undo ImageNet normalisation: ``x * std + mean``.

    Expects *tensor* of shape ``(C, H, W)`` or ``(1, C, H, W)``.
    Returns a clipped ``[0, 1]`` tensor.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def _save_image(tensor: Tensor, path: Path) -> None:
    """Save a ``(C, H, W)`` float tensor in ``[0, 1]`` as a PNG image."""
    from torchvision.utils import save_image

    save_image(tensor, str(path))


def append_metamer_metadata(metadata: Dict, output_dir: Union[str, Path]) -> None:
    """Append a single metadata record to the layer's JSONL file.

    Opens *output_dir* / ``metadata.jsonl`` in append mode and writes one
    JSON line (no indent). Append-only, so multiple jobs can add samples
    without read-modify-write.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.jsonl"
    with open(meta_path, "a") as f:
        f.write(json.dumps(metadata, default=str) + "\n")


def load_layer_metadata(path: Union[str, Path]) -> Dict[int, Dict]:
    """Load metadata for a layer from its JSONL file.

    Reads *path* if it is a file, otherwise *path* / ``metadata.jsonl``.
    Each line is parsed as JSON; records are keyed by ``sample_idx``
    (last occurrence wins for duplicate indices).

    Returns
    -------
    dict[int, dict]
        Mapping from sample index to metadata dict for that sample.
    """
    path = Path(path)
    if path.suffix == ".jsonl" and path.is_file():
        meta_path = path
    else:
        meta_path = path / "metadata.jsonl"
    if not meta_path.exists():
        return {}
    result: Dict[int, Dict] = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            result[record["sample_idx"]] = record
    return result


def save_metamer_results(
    metamer: Tensor,
    original: Tensor,
    metadata: Dict,
    output_dir: Union[str, Path],
    sample_idx: int,
    class_label: str,
    modality: str = "vision",
) -> None:
    """
    Save metamer generation results to disk.

    Creates the following inside *output_dir*::

        idx{sample_idx:04d}_{class_label}_metamer.pt
        idx{sample_idx:04d}_{class_label}_metamer.png
        idx{sample_idx:04d}_{class_label}_original.png
        metadata.jsonl   (one JSON line per sample; shared across all samples in this dir)

    Use :func:`load_layer_metadata` to read metadata for a layer.

    Parameters
    ----------
    metamer : Tensor
        The optimised metamer tensor (normalised, on any device).
    original : Tensor
        The original stimulus tensor (normalised, on any device).
    metadata : dict
        Optimisation metadata (loss curves, parameters, …).
    output_dir : path-like
        Directory to write into (created if needed).
    sample_idx : int
        Dataset sample index (zero-padded to 4 digits in filenames).
    class_label : str
        Human-readable class label.
    modality : str
        ``"vision"`` saves PNG images; other modalities can be added.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"idx{sample_idx:04d}_{class_label}"

    # 1. Raw metamer tensor
    pt_path = output_dir / f"{prefix}_metamer.pt"
    torch.save(metamer.cpu(), pt_path)

    # 2. Append to layer's metadata JSONL (one line per sample)
    full_meta = {
        "sample_idx": sample_idx,
        "class_label": class_label,
        **metadata,
    }
    append_metamer_metadata(full_meta, output_dir)

    # 3. Modality-specific visual outputs
    # Both metamer and original are expected in pixel space [0, 1].
    if modality == "vision":
        metamer_img = metamer.cpu()
        original_img = original.cpu()
        # Squeeze batch dimension if present
        if metamer_img.dim() == 4:
            metamer_img = metamer_img.squeeze(0)
        if original_img.dim() == 4:
            original_img = original_img.squeeze(0)

        _save_image(metamer_img.clamp(0, 1), output_dir / f"{prefix}_metamer.png")
        _save_image(original_img.clamp(0, 1), output_dir / f"{prefix}_original.png")
    # Future: elif modality == "audio": save .wav
