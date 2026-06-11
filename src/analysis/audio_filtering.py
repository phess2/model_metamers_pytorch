from __future__ import annotations

from typing import Any

import torch


def resolve_audio_lowpass_cutoff_hz(model_name: str) -> int | None:
    """Return the requested lowpass cutoff for supported audio model families."""
    normalized = model_name.strip().lower()
    if normalized.startswith("audiomae") or normalized.startswith("beats"):
        return 8_000
    if normalized.startswith("panns"):
        return 16_000
    if normalized.startswith("clap"):
        return 24_000
    return None


def lowpass_filter_waveform(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    cutoff_hz: float,
) -> torch.Tensor:
    """Apply an FFT lowpass filter to the waveform's last (time) dimension."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}.")
    if waveform.dim() not in (1, 2, 3):
        raise ValueError(
            "Expected waveform with shape [T], [B, T], or [B, C, T]; "
            f"got {tuple(waveform.shape)}."
        )

    num_samples = waveform.shape[-1]
    if num_samples <= 1:
        return waveform

    nyquist_hz = float(sample_rate) / 2.0
    if cutoff_hz >= nyquist_hz:
        return waveform

    original_shape = waveform.shape
    if waveform.dim() == 1:
        reshaped = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        reshaped = waveform
    else:
        reshaped = waveform.reshape(-1, num_samples)

    freq_bins = torch.fft.rfftfreq(num_samples, d=1.0 / float(sample_rate), device=waveform.device)
    keep_mask = (freq_bins <= cutoff_hz).to(dtype=reshaped.dtype).unsqueeze(0)
    spectrum = torch.fft.rfft(reshaped, dim=-1)
    filtered = torch.fft.irfft(spectrum * keep_mask, n=num_samples, dim=-1)
    return filtered.reshape(original_shape)


def lowpass_filter_for_model(
    waveform: torch.Tensor,
    *,
    model_name: str,
    sample_rate: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Filter waveform with model-aware cutoff and return metadata."""
    cutoff_hz = resolve_audio_lowpass_cutoff_hz(model_name)
    if cutoff_hz is None:
        return waveform, {
            "lowpass_filter_applied": False,
            "lowpass_filter_cutoff_hz": None,
            "lowpass_filter_effective_cutoff_hz": None,
            "lowpass_filter_reason": "unknown_model_family",
        }

    nyquist_hz = float(sample_rate) / 2.0
    if cutoff_hz >= nyquist_hz:
        return waveform, {
            "lowpass_filter_applied": False,
            "lowpass_filter_cutoff_hz": int(cutoff_hz),
            "lowpass_filter_effective_cutoff_hz": nyquist_hz,
            "lowpass_filter_reason": "cutoff_at_or_above_saved_nyquist",
        }

    filtered = lowpass_filter_waveform(
        waveform,
        sample_rate=sample_rate,
        cutoff_hz=float(cutoff_hz),
    )
    return filtered, {
        "lowpass_filter_applied": True,
        "lowpass_filter_cutoff_hz": int(cutoff_hz),
        "lowpass_filter_effective_cutoff_hz": float(cutoff_hz),
        "lowpass_filter_reason": "applied",
    }
