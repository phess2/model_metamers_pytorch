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

    freq_bins = torch.fft.rfftfreq(
        num_samples, d=1.0 / float(sample_rate), device=waveform.device
    )
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


def perturbation_high_frequency_energy_ratio(
    delta: torch.Tensor,
    *,
    sample_rate: int,
    cutoff_hz: float,
) -> float:
    """Return the fraction of perturbation FFT energy above ``cutoff_hz``."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}.")
    if delta.shape[-1] <= 1:
        return 0.0

    flattened = delta.reshape(-1, delta.shape[-1])
    spectrum = torch.fft.rfft(flattened, dim=-1)
    energy = spectrum.abs().square()
    frequencies = torch.fft.rfftfreq(
        delta.shape[-1],
        d=1.0 / float(sample_rate),
        device=delta.device,
    )
    high_frequency_energy = energy[:, frequencies > cutoff_hz].sum()
    total_energy = energy.sum()
    if float(total_energy.item()) == 0.0:
        return 0.0
    return float((high_frequency_energy / total_energy).item())


def finalize_audio_adversarial_for_model(
    *,
    source: torch.Tensor,
    adversarial: torch.Tensor,
    model_name: str,
    sample_rate: int,
    norm: str,
    epsilon: float,
    clamp_range: tuple[float, float] = (-1.0, 1.0),
    num_projection_iterations: int = 8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Band-limit the perturbation and place it on the requested norm boundary.

    Filtering the complete adversarial waveform changes the clean signal and can
    create a perturbation much larger than epsilon. This helper filters only the
    perturbation, then alternates norm-boundary scaling and waveform clamping.
    """
    if source.shape != adversarial.shape:
        raise ValueError(
            f"source/adversarial shape mismatch: {source.shape} vs {adversarial.shape}"
        )
    if norm not in {"l2", "linf"}:
        raise ValueError(f"Unsupported norm={norm!r}.")
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
    if num_projection_iterations <= 0:
        raise ValueError("num_projection_iterations must be positive.")

    clamp_min, clamp_max = clamp_range
    if clamp_min >= clamp_max:
        raise ValueError(f"Invalid clamp_range={clamp_range}.")

    source = source.detach()
    delta = adversarial.detach() - source
    cutoff_hz = resolve_audio_lowpass_cutoff_hz(model_name)
    filter_applied = cutoff_hz is not None and cutoff_hz < (sample_rate / 2.0)

    if epsilon == 0.0:
        delta = torch.zeros_like(delta)
    else:
        for _ in range(num_projection_iterations):
            if filter_applied:
                delta = lowpass_filter_waveform(
                    delta,
                    sample_rate=sample_rate,
                    cutoff_hz=float(cutoff_hz),
                )

            flattened = delta.reshape(delta.shape[0], -1)
            if norm == "l2":
                active_norms = torch.linalg.vector_norm(
                    flattened, ord=2, dim=1, keepdim=True
                )
            else:
                active_norms = flattened.abs().amax(dim=1, keepdim=True)
            if bool((active_norms <= 1e-12).any().item()):
                raise ValueError(
                    "Cannot use a nonzero perturbation budget with a zero perturbation."
                )
            scale = float(epsilon) / active_norms
            delta = (flattened * scale).reshape_as(delta)
            delta = (source + delta).clamp(clamp_min, clamp_max) - source

    finalized = source + delta
    flattened = delta.reshape(delta.shape[0], -1)
    if norm == "l2":
        active_norms = torch.linalg.vector_norm(flattened, ord=2, dim=1)
    else:
        active_norms = flattened.abs().amax(dim=1)
    max_active_norm = float(active_norms.max().item())
    utilization = max_active_norm / float(epsilon) if epsilon > 0.0 else None
    high_frequency_energy_ratio = (
        perturbation_high_frequency_energy_ratio(
            delta,
            sample_rate=sample_rate,
            cutoff_hz=float(cutoff_hz),
        )
        if cutoff_hz is not None
        else None
    )
    metadata = {
        "lowpass_filter_applied": filter_applied,
        "lowpass_filter_cutoff_hz": cutoff_hz,
        "lowpass_filter_effective_cutoff_hz": (
            float(cutoff_hz)
            if filter_applied and cutoff_hz is not None
            else float(sample_rate) / 2.0
        ),
        "lowpass_filter_reason": (
            "applied_to_perturbation"
            if filter_applied
            else "cutoff_at_or_above_saved_nyquist"
        ),
        "finalization_norm": norm,
        "finalization_max_active_norm": max_active_norm,
        "finalization_budget_utilization": utilization,
        "perturbation_high_frequency_energy_ratio": high_frequency_energy_ratio,
        "projection_iterations": int(num_projection_iterations),
    }
    return finalized, metadata
