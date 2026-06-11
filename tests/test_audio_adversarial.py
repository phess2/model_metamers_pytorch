import importlib
import importlib.util
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


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_AUDIO_FILTERING = _load_module_from_file(
    "analysis_audio_filtering_for_tests", _ROOT / "src" / "analysis" / "audio_filtering.py"
)
_AUDIO_ADVERSARIAL = _load_module_from_file(
    "analysis_audio_adversarial_for_tests", _ROOT / "src" / "analysis" / "audio_adversarial.py"
)
_SAVING = _load_module_from_file("analysis_saving_for_tests", _ROOT / "src" / "analysis" / "saving.py")

BaseAudioModelWrapper = _AUDIO_BASE.BaseAudioModelWrapper
get_audio_model = _AUDIO_REGISTRY.get_audio_model

AdversarialAttackConfig = _AUDIO_ADVERSARIAL.AdversarialAttackConfig
AudioAdversarialAttacker = _AUDIO_ADVERSARIAL.AudioAdversarialAttacker
AudioRepresentationAttacker = _AUDIO_ADVERSARIAL.AudioRepresentationAttacker
RepresentationAttackConfig = _AUDIO_ADVERSARIAL.RepresentationAttackConfig
audit_waveform_gradient = _AUDIO_ADVERSARIAL.audit_waveform_gradient
build_multihot_target = _AUDIO_ADVERSARIAL.build_multihot_target
classification_loss = _AUDIO_ADVERSARIAL.classification_loss
project_l2 = _AUDIO_ADVERSARIAL.project_l2
project_linf = _AUDIO_ADVERSARIAL.project_linf
representation_distance = _AUDIO_ADVERSARIAL.representation_distance
lowpass_filter_for_model = _AUDIO_FILTERING.lowpass_filter_for_model

load_layer_metadata = _SAVING.load_layer_metadata
save_audio_adversarial_results = _SAVING.save_audio_adversarial_results

_RUN_AUDIO_MODEL_GPU_TESTS = os.environ.get("RUN_AUDIO_MODEL_GPU_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class _ToyAudioWrapper(BaseAudioModelWrapper):
    def __init__(self, num_classes: int = 8):
        super().__init__(model=nn.Identity())
        self.num_classes = num_classes
        self.proj = nn.Conv1d(1, 2, kernel_size=5, padding=2, bias=False)
        self.classifier = nn.Linear(2, num_classes, bias=True)
        with torch.no_grad():
            self.proj.weight.fill_(0.25)
        self.metamer_layers = ["toy_layer"]

    @property
    def available_layers(self) -> list[str]:
        return ["toy_layer", "final", "logits"]

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
        logits = self.classifier(final)
        reps = {"toy_layer": toy_layer, "final": final, "logits": logits}
        return final, reps

    def get_classifier_logits(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        del sr
        _, reps = self.forward_with_representations(waveform=waveform, sr=16_000)
        return reps["logits"]


class AudioAdversarialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.wrapper = _ToyAudioWrapper()
        self.source = torch.randn(1, 1, 512)
        self.target = torch.randn(1, 1, 512)
        self.true_labels = [0, 3]

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

    def test_build_multihot_target(self):
        target = build_multihot_target(8, [0, 3], device="cpu")
        self.assertEqual(float(target.sum().item()), 2.0)
        self.assertEqual(float(target[0].item()), 1.0)
        self.assertEqual(float(target[3].item()), 1.0)

    def test_untargeted_attack_increases_classification_loss(self):
        cfg = AdversarialAttackConfig(
            norm="l2",
            epsilon=0.5,
            step_size=0.05,
            num_steps=20,
            num_random_starts=1,
            clamp_range=(-2.0, 2.0),
            seed=123,
        )
        attacker = AudioAdversarialAttacker(
            model=self.wrapper,
            sample_rate=16_000,
            config=cfg,
            device="cpu",
        )
        clean_logits = self.wrapper.get_classifier_logits(self.source, sr=16_000)
        target = build_multihot_target(
            num_classes=self.wrapper.num_classes,
            label_indices=self.true_labels,
            device="cpu",
        )
        clean_loss = classification_loss(clean_logits, target)

        adv, metadata = attacker.attack_untargeted(
            self.source, true_label_indices=self.true_labels
        )
        adv_logits = self.wrapper.get_classifier_logits(adv, sr=16_000)
        adv_loss = classification_loss(adv_logits, target)

        self.assertGreater(float(adv_loss.item()), float(clean_loss.item()))
        self.assertGreater(
            float(metadata["best_final_classification_loss"]),
            float(metadata["initial_classification_loss"]),
        )

    def test_representation_attacker_increases_distance(self):
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

    def test_save_audio_adversarial_results_outputs_files_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "layer0"
            filtered_adversarial, lowpass_metadata = lowpass_filter_for_model(
                self.source + 0.01,
                model_name="audiomae_as2m_ft_as20k",
                sample_rate=32_000,
            )
            metadata = {
                "sample_rate": 32_000,
                "attack_mode": "untargeted",
                "classification_loss": 1.23,
                **lowpass_metadata,
            }
            saved_paths = save_audio_adversarial_results(
                original=self.source,
                adversarial=filtered_adversarial,
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
            self.assertTrue(loaded_meta[3]["lowpass_filter_applied"])
            self.assertEqual(loaded_meta[3]["lowpass_filter_cutoff_hz"], 8_000)
            saved_adversarial = torch.load(saved_paths["adversarial_pt"])
            self.assertTrue(torch.allclose(saved_adversarial, filtered_adversarial.cpu()))

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

    def test_audit_waveform_gradient_logits_path(self):
        report = audit_waveform_gradient(
            model=self.wrapper,
            waveform=self.source.clone(),
            sample_rate=16_000,
            true_label_indices=self.true_labels,
            device="cpu",
            use_logits=True,
        )
        self.assertTrue(report["use_logits"])
        self.assertEqual(report["layer_name"], "logits")
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
                try:
                    logits = wrapper.get_classifier_logits(waveform.clone(), sr=16_000)
                except NotImplementedError:
                    self.skipTest(f"{model_name} has no classifier logits")
                num_classes = int(logits.shape[-1])
                true_labels = [0, 1]
                report = audit_waveform_gradient(
                    model=wrapper,
                    waveform=waveform.clone(),
                    sample_rate=16_000,
                    true_label_indices=true_labels,
                    device="cuda",
                    use_logits=True,
                )
                self.assertTrue(report["gradient_is_finite"])
                self.assertGreater(report["gradient_nonzero_count"], 0)
                del num_classes


if __name__ == "__main__":
    unittest.main()
