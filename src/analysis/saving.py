"""
Saving utilities for metamer generation results.

Handles saving metamer tensors, original/metamer images (for visual
inspection), and JSONL metadata (one file per layer).  Modality-aware: for
vision, saves ``.png`` files; for audio, saves ``.wav`` files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile
from torch import Tensor


def _save_image(tensor: Tensor, path: Path) -> None:
    """Save a ``(C, H, W)`` float tensor in ``[0, 1]`` as a PNG image."""
    from torchvision.utils import save_image

    save_image(tensor, str(path))


def _save_audio_waveform(tensor: Tensor, path: Path, sample_rate: int) -> None:
    """Save a waveform tensor as int16 WAV."""
    waveform = tensor.detach().cpu()
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    if waveform.dim() == 2:
        waveform = waveform[0]
    waveform = waveform.clamp(-1.0, 1.0).numpy()
    wav_int16 = (waveform * 32767.0).astype("int16")
    wavfile.write(str(path), int(sample_rate), wav_int16)


def _save_audio_spectrogram_image(
    tensor: Tensor,
    path: Path,
    sample_rate: int,
    title: str,
    n_fft: int = 512,
    hop_length: int = 160,
    eps: float = 1e-8,
) -> None:
    waveform = tensor.detach().cpu()
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    if waveform.dim() == 2:
        waveform = waveform[0]
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform for spectrogram save, got {waveform.shape}")

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        center=True,
    )
    spec = torch.log10(stft.abs().pow(2) + eps).numpy()
    times = np.linspace(0, waveform.shape[0] / float(sample_rate), spec.shape[1])
    freqs = np.linspace(0, sample_rate / 2.0, spec.shape[0])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.pcolormesh(times, freqs, spec, shading="auto")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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

    # 3. Modality-specific outputs
    # For vision, both metamer and original are expected in pixel space [0, 1].
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
    elif modality == "audio":
        sample_rate = metadata.get("sample_rate", metadata.get("file_sr", 16_000))
        _save_audio_waveform(
            original, output_dir / f"{prefix}_original.wav", sample_rate=sample_rate
        )
        _save_audio_waveform(
            metamer, output_dir / f"{prefix}_metamer.wav", sample_rate=sample_rate
        )


def save_adversarial_results_csv(
    rows: Sequence[Dict],
    output_path: Union[str, Path],
) -> Path:
    """
    Save per-sample adversarial evaluation records to CSV.

    The input rows should include:
      - true_class_label
      - predicted_class_label
      - is_correct
      - true_class_softmax

    Additional keys are allowed and will be included as extra columns.
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fieldnames = [
            "true_class_label",
            "predicted_class_label",
            "is_correct",
            "true_class_softmax",
        ]
    else:
        base_fieldnames = [
            "sample_idx",
            "dataset_idx",
            "true_label_idx",
            "predicted_label_idx",
            "true_class_label",
            "predicted_class_label",
            "is_correct",
            "true_class_softmax",
        ]
        first_row_keys = list(rows[0].keys())
        extras = [key for key in first_row_keys if key not in base_fieldnames]
        fieldnames = [name for name in base_fieldnames if name in first_row_keys] + extras

    with open(output_path_obj, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path_obj


def save_audio_adversarial_results(
    *,
    original: Tensor,
    adversarial: Tensor,
    metadata: Dict,
    output_dir: Union[str, Path],
    sample_idx: int,
    class_label: str,
    include_spectrograms: bool = True,
) -> Dict[str, str]:
    """Save waveform adversarial outputs and append metadata JSONL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"idx{sample_idx:04d}_{class_label}"
    sample_rate = int(metadata.get("sample_rate", metadata.get("file_sr", 16_000)))

    delta = (adversarial - original).detach()

    saved_paths = {
        "original_pt": str(output_dir / f"{prefix}_original.pt"),
        "adversarial_pt": str(output_dir / f"{prefix}_adversarial.pt"),
        "delta_pt": str(output_dir / f"{prefix}_delta.pt"),
        "original_wav": str(output_dir / f"{prefix}_original.wav"),
        "adversarial_wav": str(output_dir / f"{prefix}_adversarial.wav"),
        "delta_wav": str(output_dir / f"{prefix}_perturbation.wav"),
    }

    torch.save(original.detach().cpu(), saved_paths["original_pt"])
    torch.save(adversarial.detach().cpu(), saved_paths["adversarial_pt"])
    torch.save(delta.detach().cpu(), saved_paths["delta_pt"])

    _save_audio_waveform(original, Path(saved_paths["original_wav"]), sample_rate=sample_rate)
    _save_audio_waveform(
        adversarial, Path(saved_paths["adversarial_wav"]), sample_rate=sample_rate
    )
    _save_audio_waveform(delta, Path(saved_paths["delta_wav"]), sample_rate=sample_rate)

    if include_spectrograms:
        saved_paths.update(
            {
                "original_spectrogram_png": str(output_dir / f"{prefix}_original_spec.png"),
                "adversarial_spectrogram_png": str(
                    output_dir / f"{prefix}_adversarial_spec.png"
                ),
                "delta_spectrogram_png": str(output_dir / f"{prefix}_delta_spec.png"),
                "spec_difference_png": str(output_dir / f"{prefix}_spec_diff.png"),
            }
        )
        _save_audio_spectrogram_image(
            original,
            Path(saved_paths["original_spectrogram_png"]),
            sample_rate=sample_rate,
            title="Original Spectrogram",
        )
        _save_audio_spectrogram_image(
            adversarial,
            Path(saved_paths["adversarial_spectrogram_png"]),
            sample_rate=sample_rate,
            title="Adversarial Spectrogram",
        )
        _save_audio_spectrogram_image(
            delta,
            Path(saved_paths["delta_spectrogram_png"]),
            sample_rate=sample_rate,
            title="Perturbation Spectrogram",
        )
        _save_audio_spectrogram_image(
            adversarial - original,
            Path(saved_paths["spec_difference_png"]),
            sample_rate=sample_rate,
            title="Spectrogram Difference Proxy",
        )

    metadata_with_paths = {
        "sample_idx": sample_idx,
        "class_label": class_label,
        **metadata,
        "saved_paths": saved_paths,
    }
    append_metamer_metadata(metadata_with_paths, output_dir=output_dir)
    return saved_paths
