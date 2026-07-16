import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_VALIDATOR = _load_module_from_file(
    "validate_audio_adversarial_budgets_for_tests",
    _ROOT / "scripts" / "validate_audio_adversarial_budgets.py",
)
_ADV_GEN = _load_module_from_file(
    "generate_audio_adversarials_budget_tests",
    _ROOT / "scripts" / "generate_audio_adversarials.py",
)
_MAP_EVAL = _load_module_from_file(
    "evaluate_audio_adversarial_map_budget_tests",
    _ROOT / "scripts" / "evaluate_audio_adversarial_map.py",
)

parse_epsilon_from_attack_id = _VALIDATOR.parse_epsilon_from_attack_id
extract_budget_stats = _VALIDATOR.extract_budget_stats
summarize_attack_dir = _VALIDATOR.summarize_attack_dir
discover_attack_dirs = _VALIDATOR.discover_attack_dirs
recompute_saved_artifact_stats = _VALIDATOR.recompute_saved_artifact_stats
summarize_saved_artifacts = _VALIDATOR.summarize_saved_artifacts
compute_budget_utilization = _ADV_GEN.compute_budget_utilization
find_first_indices_without_label = _ADV_GEN._find_first_indices_without_label
mean_of_row_values = _ADV_GEN._mean_of_row_values
map_parse_epsilon_from_attack_id = _MAP_EVAL._parse_epsilon_from_attack_id


def _make_record(*, epsilon: float, pre_norm: float, post_norm: float) -> dict:
    return {
        "budget_verification_pre_lowpass": {
            "norm": "l2",
            "epsilon": epsilon,
            "max_active_norm": pre_norm,
        },
        "budget_verification_post_lowpass": {
            "norm": "l2",
            "epsilon": epsilon,
            "max_active_norm": post_norm,
        },
    }


class BudgetUtilizationTests(unittest.TestCase):
    def test_utilization_fraction(self):
        self.assertAlmostEqual(
            compute_budget_utilization(max_active_norm=1.2, epsilon=2.4), 0.5
        )

    def test_zero_epsilon_returns_none(self):
        self.assertIsNone(compute_budget_utilization(max_active_norm=0.0, epsilon=0.0))

    def test_mean_of_row_values_ignores_blanks(self):
        rows = [
            {"frac_true_labels_suppressed": 0.5},
            {"frac_true_labels_suppressed": ""},
            {"frac_true_labels_suppressed": 1.0},
            {},
        ]
        self.assertAlmostEqual(
            mean_of_row_values(rows, "frac_true_labels_suppressed"), 0.75
        )

    def test_mean_of_row_values_all_blank_is_nan(self):
        rows = [{"x": ""}, {}]
        self.assertTrue(math.isnan(mean_of_row_values(rows, "x")))

    def test_find_first_indices_without_target_label(self):
        examples = [
            {"labels": [0, 3]},
            {"labels": [2]},
            {"labels": [0]},
            {"labels": [4, 5]},
            {"labels": [6]},
        ]
        previous_loader = _ADV_GEN.example_loader
        _ADV_GEN.example_loader = lambda *args, **kwargs: iter(examples)
        try:
            selected = find_first_indices_without_label(
                tfrecord_path=Path("unused.tfrecords"),
                excluded_label=0,
                count=2,
                decode_labels_fn=lambda example: example["labels"],
            )
        finally:
            _ADV_GEN.example_loader = previous_loader
        self.assertEqual(selected, [1, 3])


class MapEvalParseTests(unittest.TestCase):
    def test_parse_epsilon_ignores_trailing_suffix(self):
        self.assertAlmostEqual(
            map_parse_epsilon_from_attack_id("adversarial_l2_eps_2p4_suppress"), 2.4
        )
        self.assertAlmostEqual(
            map_parse_epsilon_from_attack_id("adversarial_l2_eps_3p2"), 3.2
        )


class BudgetValidatorTests(unittest.TestCase):
    def test_parse_epsilon_from_attack_id(self):
        self.assertAlmostEqual(
            parse_epsilon_from_attack_id("adversarial_l2_eps_2p4"), 2.4
        )
        self.assertAlmostEqual(
            parse_epsilon_from_attack_id("adversarial_l2_eps_0"), 0.0
        )

    def test_parse_epsilon_ignores_trailing_suffix(self):
        self.assertAlmostEqual(
            parse_epsilon_from_attack_id("adversarial_l2_eps_2p4_suppress"), 2.4
        )
        self.assertAlmostEqual(
            parse_epsilon_from_attack_id("adversarial_l2_eps_0_suppress"), 0.0
        )
        self.assertAlmostEqual(
            parse_epsilon_from_attack_id("adversarial_l2_eps_1p6_single_label"), 1.6
        )

    def test_extract_budget_stats(self):
        record = _make_record(epsilon=2.4, pre_norm=2.4, post_norm=2.3)
        pre = extract_budget_stats(record, stage="pre_lowpass")
        assert pre is not None
        self.assertAlmostEqual(float(pre["utilization"]), 1.0)
        post = extract_budget_stats(record, stage="post_lowpass")
        assert post is not None
        self.assertAlmostEqual(float(post["max_active_norm"]), 2.3)
        self.assertIsNone(extract_budget_stats({}, stage="pre_lowpass"))

    def test_summarize_flags_overshoot_and_undershoot(self):
        records = [
            _make_record(epsilon=2.0, pre_norm=2.0, post_norm=1.95),
            _make_record(epsilon=2.0, pre_norm=2.5, post_norm=2.4),  # overshoot
            _make_record(epsilon=2.0, pre_norm=1.0, post_norm=0.9),  # undershoot
        ]
        summary = summarize_attack_dir(
            records=records,
            epsilon=2.0,
            stage="pre_lowpass",
            overshoot_tolerance=1e-4,
            undershoot_threshold=0.9,
        )
        self.assertEqual(summary["num_records"], 3)
        self.assertEqual(summary["num_overshoot"], 1)
        self.assertEqual(summary["num_undershoot"], 1)
        self.assertAlmostEqual(float(summary["max_norm"]), 2.5)
        self.assertAlmostEqual(float(summary["min_utilization"]), 0.5)

    def test_summarize_clean_run_passes(self):
        records = [
            _make_record(epsilon=2.0, pre_norm=2.0, post_norm=1.98),
            _make_record(epsilon=2.0, pre_norm=1.999999, post_norm=1.97),
        ]
        summary = summarize_attack_dir(
            records=records,
            epsilon=2.0,
            stage="pre_lowpass",
            overshoot_tolerance=1e-4,
            undershoot_threshold=0.9,
        )
        self.assertEqual(summary["num_overshoot"], 0)
        self.assertEqual(summary["num_undershoot"], 0)

    def test_zero_epsilon_has_no_undershoot(self):
        records = [_make_record(epsilon=0.0, pre_norm=0.0, post_norm=0.0)]
        summary = summarize_attack_dir(
            records=records,
            epsilon=0.0,
            stage="pre_lowpass",
            overshoot_tolerance=1e-4,
            undershoot_threshold=0.9,
        )
        self.assertEqual(summary["num_overshoot"], 0)
        self.assertEqual(summary["num_undershoot"], 0)

    def test_discover_attack_dirs(self):
        with TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            for model, eps_token in [("model_a", "2p4"), ("model_b", "0")]:
                logits_dir = (
                    run_root
                    / model
                    / "adversarial"
                    / f"adversarial_l2_eps_{eps_token}"
                    / "logits"
                )
                logits_dir.mkdir(parents=True)
                record = _make_record(epsilon=2.4, pre_norm=2.4, post_norm=2.3)
                (logits_dir / "metadata.jsonl").write_text(json.dumps(record) + "\n")
            found = discover_attack_dirs(run_root=run_root, norm="l2")
            self.assertEqual(len(found), 2)
            models = sorted(item[0] for item in found)
            self.assertEqual(models, ["model_a", "model_b"])
            epsilons = sorted(item[1] for item in found)
            self.assertEqual(epsilons, [0.0, 2.4])

    def test_discover_attack_dirs_with_suppress_suffix(self):
        with TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            logits_dir = (
                run_root
                / "beats_iter3_plus_as2m"
                / "adversarial"
                / "adversarial_l2_eps_2p4_suppress"
                / "logits"
            )
            logits_dir.mkdir(parents=True)
            record = _make_record(epsilon=2.4, pre_norm=2.4, post_norm=2.3)
            (logits_dir / "metadata.jsonl").write_text(json.dumps(record) + "\n")
            found = discover_attack_dirs(run_root=run_root, norm="l2")
            self.assertEqual(len(found), 1)
            model_name, epsilon, attack_dir = found[0]
            self.assertEqual(model_name, "beats_iter3_plus_as2m")
            self.assertAlmostEqual(epsilon, 2.4)
            self.assertTrue(attack_dir.name.endswith("_suppress"))

    def test_recompute_saved_artifacts_uses_tensor_files(self):
        with TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp)
            original = torch.zeros(1, 1, 32_000)
            delta = torch.zeros_like(original)
            delta[..., :4] = 0.5
            adversarial = original + delta
            original_path = metadata_dir / "original.pt"
            adversarial_path = metadata_dir / "adversarial.pt"
            torch.save(original, original_path)
            torch.save(adversarial, adversarial_path)
            record = {
                "sample_rate": 32_000,
                "saved_paths": {
                    "original_pt": str(original_path),
                    "adversarial_pt": str(adversarial_path),
                },
                "budget_verification_post_lowpass": {"max_active_norm": 1.0},
            }

            stats = recompute_saved_artifact_stats(
                record,
                metadata_dir=metadata_dir,
                model_name="panns_cnn14",
                norm="l2",
            )

            assert stats is not None
            self.assertAlmostEqual(float(stats["max_active_norm"]), 1.0)

    def test_saved_artifact_summary_flags_mismatch_and_nyquist_energy(self):
        with TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp)
            sample_rate = 32_000
            time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
            original = torch.zeros(1, 1, sample_rate)
            delta = torch.sin(2.0 * torch.pi * 12_000.0 * time).reshape(1, 1, -1)
            delta = delta / delta.norm()
            adversarial = original + delta
            original_path = metadata_dir / "original.pt"
            adversarial_path = metadata_dir / "adversarial.pt"
            torch.save(original, original_path)
            torch.save(adversarial, adversarial_path)
            records = [
                {
                    "sample_rate": sample_rate,
                    "saved_paths": {
                        "original_pt": str(original_path),
                        "adversarial_pt": str(adversarial_path),
                    },
                    "budget_verification_post_lowpass": {"max_active_norm": 0.5},
                }
            ]

            summary = summarize_saved_artifacts(
                records=records,
                metadata_dir=metadata_dir,
                model_name="beats_iter3_plus_as2m",
                norm="l2",
                epsilon=1.0,
                overshoot_tolerance=1e-4,
                undershoot_threshold=0.9,
                metadata_tolerance=1e-5,
                max_high_frequency_ratio=1e-6,
            )

            self.assertEqual(summary["num_metadata_mismatch"], 1)
            self.assertEqual(summary["num_nyquist_violation"], 1)
            self.assertEqual(summary["num_undershoot"], 0)


if __name__ == "__main__":
    unittest.main()
