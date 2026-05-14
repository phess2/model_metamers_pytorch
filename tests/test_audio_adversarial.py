import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

_AUDIO_BASE = importlib.import_module("models.audio.base")
_AUDIO_REGISTRY = importlib.import_module("models.audio.registry")
_AUDIO_ADVERSARIAL = importlib.import_module("analysis.audio_adversarial")
_SAVING = importlib.import_module("analysis.saving")

BaseAudioModelWrapper = _AUDIO_BASE.BaseAudioModelWrapper
get_audio_model = _AUDIO_REGISTRY.get_audio_model

AudioRepresentationAttacker = _AUDIO_ADVERSARIAL.AudioRepresentationAttacker
RepresentationAttackConfig = _AUDIO_ADVERSARIAL.RepresentationAttackConfig
audit_waveform_gradient = _AUDIO_ADVERSARIAL.audit_waveform_gradient
project_l2 = _AUDIO_ADVERSARIAL.project_l2
project_linf = _AUDIO_ADVERSARIAL.project_linf
representation_distance = _AUDIO_ADVERSARIAL.representation_distance

load_layer_metadata = _SAVING.load_layer_metadata
save_audio_adversarial_results = _SAVING.save_audio_adversarial_results

_RUN_AUDIO_MODEL_GPU_TESTS = os.environ.get("RUN_AUDIO_MODEL_GPU_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class _ToyAudioWrapper(BaseAudioModelWrapper):
    def __init__(self):
        super().__init__(model=nn.Identity())
        self.proj = nn.Conv1d(1, 2, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.proj.weight.fill_(0.25)
        self.metamer_layers = ["toy_layer"]

    @property
    def available_layers(self) -> list[str]:
        return ["toy_layer", "final"]

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        del sr
        if waveform.dim() == 3:
            return waveform
        if waveform.dim() == 2:
            return waveform.unsqueeze(1)
        if waveform.dim() == 1:
            return waveform.unsqueeze(0).unsqueeze(0)
        raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")

    def forward_waveform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        return self.forward_features(waveform, sr=sr, layer="final")

    def forward_features(
        self, waveform: torch.Tensor, sr: int, layer: str | None = None
    ) -> torch.Tensor:
        _, reps = self.forward_with_representations(waveform=waveform, sr=sr)
        selected_layer = layer or "final"
        return reps[selected_layer]

    def forward_with_representations(
        self, waveform: torch.Tensor, sr: int = 16_000, fake_relu: bool = False
    ):
        del sr, fake_relu
        x = self.preprocess(waveform, sr=16_000)
        toy_layer = torch.tanh(self.proj(x))
        final = toy_layer.mean(dim=2)
        reps = {"toy_layer": toy_layer, "final": final}
        return final, reps


class AudioAdversarialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.wrapper = _ToyAudioWrapper()
        self.source = torch.randn(1, 1, 512)
        self.target = torch.randn(1, 1, 512)

    def test_project_l2_enforces_radius(self):
        delta = torch.randn(2, 1, 128)
        epsilon = 0.5
        projected = project_l2(delta, epsilon=epsilon)
        norms = projected.view(projected.shape[0], -1).norm(p=2, dim=1)
        self.assertTrue(torch.all(norms <= epsilon + 1e-5).item())

    def test_project_linf_enforces_radius(self):
        delta = torch.randn(2, 1, 128)
        epsilon = 0.1
        projected = project_linf(delta, epsilon=epsilon)
        max_abs = projected.abs().amax()
        self.assertLessEqual(float(max_abs.item()), epsilon + 1e-6)

    def test_untargeted_attack_increases_distance(self):
        cfg = RepresentationAttackConfig(
            norm="l2",
            epsilon=0.5,
            step_size=0.05,
            num_steps=20,
            num_random_starts=1,
            loss_type="l2",
            clamp_range=(-2.0, 2.0),
            seed=123,
        )
        attacker = AudioRepresentationAttacker(
            model=self.wrapper,
            layer_name="toy_layer",
            sample_rate=16_000,
            config=cfg,
            device="cpu",
        )
        source_rep = attacker.extract_representation(self.source)
        adv, _ = attacker.attack_untargeted(self.source, source_rep=source_rep)
        adv_rep = attacker.extract_representation(adv)

        original_distance = representation_distance(
            source_rep, source_rep, loss_type="l2", reduction="mean"
        )
        attacked_distance = representation_distance(
            adv_rep, source_rep, loss_type="l2", reduction="mean"
        )
        self.assertGreater(float(attacked_distance.item()), float(original_distance.item()))

    def test_targeted_attack_reduces_target_distance(self):
        cfg = RepresentationAttackConfig(
            norm="linf",
            epsilon=0.2,
            step_size=0.01,
            num_steps=30,
            num_random_starts=1,
            loss_type="normalized_l2",
            clamp_range=(-2.0, 2.0),
            seed=42,
        )
        attacker = AudioRepresentationAttacker(
            model=self.wrapper,
            layer_name="toy_layer",
            sample_rate=16_000,
            config=cfg,
            device="cpu",
        )
        source_rep = attacker.extract_representation(self.source).detach()
        target_rep = attacker.extract_representation(self.target).detach()
        baseline = representation_distance(
            source_rep, target_rep, loss_type="normalized_l2", reduction="mean"
        )

        adv, _ = attacker.attack_targeted(
            source_waveform=self.source,
            target_rep=target_rep,
            source_rep=source_rep,
        )
        adv_rep = attacker.extract_representation(adv).detach()
        attacked = representation_distance(
            adv_rep, target_rep, loss_type="normalized_l2", reduction="mean"
        )
        self.assertLess(float(attacked.item()), float(baseline.item()))

    def test_save_audio_adversarial_results_outputs_files_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "layer0"
            metadata = {
                "sample_rate": 16_000,
                "attack_mode": "untargeted",
                "representation_distance_from_source": 1.23,
            }
            saved_paths = save_audio_adversarial_results(
                original=self.source,
                adversarial=self.source + 0.01,
                metadata=metadata,
                output_dir=output_dir,
                sample_idx=3,
                class_label="toy",
                include_spectrograms=True,
            )
            for save_path in saved_paths.values():
                self.assertTrue(Path(save_path).exists())
            loaded_meta = load_layer_metadata(output_dir)
            self.assertIn(3, loaded_meta)
            self.assertEqual(loaded_meta[3]["class_label"], "toy")
            self.assertIn("saved_paths", loaded_meta[3])

    def test_audit_waveform_gradient_reports_finite_nonzero(self):
        report = audit_waveform_gradient(
            model=self.wrapper,
            waveform=self.source.clone(),
            sample_rate=16_000,
            layer_name="toy_layer",
            device="cpu",
        )
        self.assertTrue(report["gradient_is_finite"])
        self.assertGreater(report["gradient_nonzero_count"], 0)


@unittest.skipUnless(
    _RUN_AUDIO_MODEL_GPU_TESTS and torch.cuda.is_available(),
    "Set RUN_AUDIO_MODEL_GPU_TESTS=1 and run on a CUDA host.",
)
class AudioAdversarialGpuSmokeTests(unittest.TestCase):
    def test_real_models_have_nonzero_finite_waveform_gradients(self):
        waveform = torch.randn(1, 1, 4_000, device="cuda")
        for model_name in ("audiomae_as2m_ft_as20k", "beats_iter3_plus_as2m"):
            with self.subTest(model_name=model_name):
                wrapper = get_audio_model(model_name, device="cuda", freeze=True)
                layer_name = list(getattr(wrapper, "metamer_layers"))[0]
                report = audit_waveform_gradient(
                    model=wrapper,
                    waveform=waveform.clone(),
                    sample_rate=16_000,
                    layer_name=layer_name,
                    device="cuda",
                )
                self.assertTrue(report["gradient_is_finite"])
                self.assertGreater(report["gradient_nonzero_count"], 0)


if __name__ == "__main__":
    unittest.main()
