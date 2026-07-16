import importlib.util
import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_AUDIO_FILTERING = _load_module_from_file(
    "analysis_audio_filtering_for_tests",
    _ROOT / "src" / "analysis" / "audio_filtering.py",
)

lowpass_filter_for_model = _AUDIO_FILTERING.lowpass_filter_for_model
lowpass_filter_waveform = _AUDIO_FILTERING.lowpass_filter_waveform
resolve_audio_lowpass_cutoff_hz = _AUDIO_FILTERING.resolve_audio_lowpass_cutoff_hz
finalize_audio_adversarial_for_model = (
    _AUDIO_FILTERING.finalize_audio_adversarial_for_model
)
perturbation_high_frequency_energy_ratio = (
    _AUDIO_FILTERING.perturbation_high_frequency_energy_ratio
)


class AudioFilteringTests(unittest.TestCase):
    def test_resolve_audio_lowpass_cutoff_hz_by_model_name(self):
        self.assertEqual(
            resolve_audio_lowpass_cutoff_hz("audiomae_as2m_ft_as20k"), 8_000
        )
        self.assertEqual(
            resolve_audio_lowpass_cutoff_hz("beats_iter3_plus_as2m"), 8_000
        )
        self.assertEqual(resolve_audio_lowpass_cutoff_hz("panns_cnn14"), 16_000)
        self.assertEqual(resolve_audio_lowpass_cutoff_hz("clap"), 24_000)
        self.assertIsNone(resolve_audio_lowpass_cutoff_hz("unknown_model"))

    def test_lowpass_filter_waveform_strongly_reduces_above_cutoff_energy(self):
        sample_rate = 16_000
        time = torch.arange(0, sample_rate, dtype=torch.float32) / float(sample_rate)
        low_tone = torch.sin(2.0 * torch.pi * 500.0 * time)
        high_tone = torch.sin(2.0 * torch.pi * 5_000.0 * time)
        waveform = (low_tone + high_tone).unsqueeze(0)

        filtered = lowpass_filter_waveform(
            waveform,
            sample_rate=sample_rate,
            cutoff_hz=2_000.0,
        )

        freq_bins = torch.fft.rfftfreq(waveform.shape[-1], d=1.0 / float(sample_rate))
        orig_spec = torch.fft.rfft(waveform, dim=-1).abs().squeeze(0)
        filt_spec = torch.fft.rfft(filtered, dim=-1).abs().squeeze(0)
        high_band = freq_bins >= 3_000.0
        ratio = float(
            (filt_spec[high_band].sum() / (orig_spec[high_band].sum() + 1e-8)).item()
        )
        self.assertLess(ratio, 0.05)

    def test_lowpass_filter_waveform_preserves_supported_shapes(self):
        sample_rate = 16_000
        waveform_1d = torch.randn(4_096)
        waveform_2d = torch.randn(2, 4_096)
        waveform_3d = torch.randn(2, 1, 4_096)

        filtered_1d = lowpass_filter_waveform(
            waveform_1d, sample_rate=sample_rate, cutoff_hz=2_000.0
        )
        filtered_2d = lowpass_filter_waveform(
            waveform_2d, sample_rate=sample_rate, cutoff_hz=2_000.0
        )
        filtered_3d = lowpass_filter_waveform(
            waveform_3d, sample_rate=sample_rate, cutoff_hz=2_000.0
        )

        self.assertEqual(filtered_1d.shape, waveform_1d.shape)
        self.assertEqual(filtered_2d.shape, waveform_2d.shape)
        self.assertEqual(filtered_3d.shape, waveform_3d.shape)
        self.assertEqual(filtered_1d.dtype, waveform_1d.dtype)

    def test_lowpass_filter_for_model_metadata_cases(self):
        waveform = torch.randn(1, 2_048)

        _, unknown_meta = lowpass_filter_for_model(
            waveform,
            model_name="not_registered",
            sample_rate=16_000,
        )
        self.assertFalse(unknown_meta["lowpass_filter_applied"])
        self.assertEqual(unknown_meta["lowpass_filter_reason"], "unknown_model_family")

        _, nyquist_meta = lowpass_filter_for_model(
            waveform,
            model_name="audiomae_as2m",
            sample_rate=16_000,
        )
        self.assertFalse(nyquist_meta["lowpass_filter_applied"])
        self.assertEqual(
            nyquist_meta["lowpass_filter_reason"], "cutoff_at_or_above_saved_nyquist"
        )
        self.assertEqual(nyquist_meta["lowpass_filter_cutoff_hz"], 8_000)

        filtered, applied_meta = lowpass_filter_for_model(
            waveform,
            model_name="audiomae_as2m",
            sample_rate=32_000,
        )
        self.assertTrue(applied_meta["lowpass_filter_applied"])
        self.assertEqual(applied_meta["lowpass_filter_reason"], "applied")
        self.assertEqual(filtered.shape, waveform.shape)

    def test_finalize_filters_delta_and_uses_full_l2_budget(self):
        sample_rate = 32_000
        num_samples = sample_rate
        time = torch.arange(num_samples, dtype=torch.float32) / float(sample_rate)
        source = (0.1 * torch.sin(2.0 * torch.pi * 12_000.0 * time)).reshape(1, 1, -1)
        proposed_delta = (
            torch.sin(2.0 * torch.pi * 1_000.0 * time)
            + torch.sin(2.0 * torch.pi * 12_000.0 * time)
        ).reshape(1, 1, -1)
        epsilon = 1.0

        finalized, metadata = finalize_audio_adversarial_for_model(
            source=source,
            adversarial=source + proposed_delta,
            model_name="audiomae_as2m_ft_as20k",
            sample_rate=sample_rate,
            norm="l2",
            epsilon=epsilon,
        )

        delta = finalized - source
        self.assertAlmostEqual(float(delta.norm().item()), epsilon, places=4)
        self.assertTrue(torch.allclose(source, source.clone()))
        self.assertLessEqual(
            perturbation_high_frequency_energy_ratio(
                delta, sample_rate=sample_rate, cutoff_hz=8_000
            ),
            1e-6,
        )
        self.assertEqual(metadata["lowpass_filter_reason"], "applied_to_perturbation")

    def test_finalize_zero_budget_preserves_source(self):
        source = torch.linspace(-1.0, 1.0, 2_048).reshape(1, 1, -1)
        finalized, metadata = finalize_audio_adversarial_for_model(
            source=source,
            adversarial=torch.zeros_like(source),
            model_name="beats_iter3_plus_as2m",
            sample_rate=32_000,
            norm="l2",
            epsilon=0.0,
        )
        self.assertTrue(torch.equal(finalized, source))
        self.assertEqual(metadata["finalization_max_active_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
