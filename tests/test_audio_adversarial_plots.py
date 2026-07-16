import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_ROOT = Path(__file__).resolve().parents[1]


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_PLOTTER = _load_module_from_file(
    "plot_audio_adversarial_epsilon_curves_for_tests",
    _ROOT / "scripts" / "plot_audio_adversarial_epsilon_curves.py",
)


class AudioAdversarialPlotTests(unittest.TestCase):
    def test_parse_epsilon_ignores_attack_suffix(self):
        self.assertEqual(
            _PLOTTER.parse_epsilon_from_attack_id("adversarial_l2_eps_2p4_suppress"),
            2.4,
        )

    def test_collects_suppress_metrics_from_sweep_summary(self):
        with TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "model"
            layer_dir = (
                model_root
                / "adversarial"
                / "adversarial_l2_eps_2p4_suppress"
                / "logits"
            )
            layer_dir.mkdir(parents=True)
            sweep_summary = {
                "attack_mode": "suppress",
                "clean_map": 0.42,
                "adversarial_map": 0.03,
                "mean_true_label_prob_source": 0.45,
                "mean_true_label_prob_adversarial": 0.005,
                "mean_frac_true_labels_suppressed": 1.0,
            }
            (layer_dir / "sweep_summary.json").write_text(
                json.dumps(sweep_summary),
                encoding="utf-8",
            )

            points = _PLOTTER.collect_model_curve(
                model_root=model_root,
                norm="l2",
            )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["attack_mode"], "suppress")
        self.assertEqual(points[0]["adversarial_map"], 0.03)
        self.assertEqual(points[0]["mean_true_label_prob_adversarial"], 0.005)
        self.assertEqual(points[0]["mean_frac_true_labels_suppressed"], 1.0)

    def test_detects_available_suppress_metric(self):
        curves = {
            "model": [
                {
                    "mean_frac_true_labels_suppressed": 1.0,
                    "mean_true_label_prob_adversarial": float("nan"),
                }
            ]
        }

        self.assertTrue(
            _PLOTTER.metric_available(
                curves,
                "mean_frac_true_labels_suppressed",
            )
        )
        self.assertFalse(
            _PLOTTER.metric_available(
                curves,
                "mean_true_label_prob_adversarial",
            )
        )

    def test_collects_targeted_metrics_from_per_clip_summary(self):
        with TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "model"
            layer_dir = (
                model_root
                / "adversarial"
                / "adversarial_l2_eps_2p4_targeted_label0000"
                / "logits"
            )
            layer_dir.mkdir(parents=True)
            (layer_dir / "summary.csv").write_text(
                "target_source_probability,target_adversarial_probability,"
                "target_top1_hit,target_top5_hit,budget_utilization_post_lowpass\n"
                "0.1,0.8,True,True,1.0\n"
                "0.2,0.6,False,True,1.0\n",
                encoding="utf-8",
            )

            points = _PLOTTER.collect_model_curve(
                model_root=model_root,
                norm="l2",
            )

        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["mean_target_probability_source"], 0.15)
        self.assertAlmostEqual(points[0]["mean_target_probability_adversarial"], 0.7)
        self.assertAlmostEqual(points[0]["target_top1_hit_rate"], 0.5)
        self.assertAlmostEqual(points[0]["target_top5_hit_rate"], 1.0)
        self.assertAlmostEqual(points[0]["mean_budget_utilization"], 1.0)


if __name__ == "__main__":
    unittest.main()
