import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_MAP_EVAL = _load_module_from_file(
    "evaluate_audio_adversarial_map_for_tests",
    _ROOT / "scripts" / "evaluate_audio_adversarial_map.py",
)
_ADV_GEN = _load_module_from_file(
    "generate_audio_adversarials_for_tests",
    _ROOT / "scripts" / "generate_audio_adversarials.py",
)

compute_map_from_scores = _MAP_EVAL.compute_map_from_scores
_average_precision_binary = _MAP_EVAL._average_precision_binary
parse_sweep_attack_id = _MAP_EVAL.parse_sweep_attack_id
_compute_monotonicity_violations = _MAP_EVAL._compute_monotonicity_violations
select_best_by_epsilon = _MAP_EVAL.select_best_by_epsilon
select_best_m_by_ranked_vote = _MAP_EVAL.select_best_m_by_ranked_vote
select_best_by_epsilon_with_fixed_m = _MAP_EVAL.select_best_by_epsilon_with_fixed_m
resolve_attack_step_size = _ADV_GEN.resolve_attack_step_size


class AudioMapEvalTests(unittest.TestCase):
    def test_resolve_attack_step_size_uses_epsilon_scaled_default(self):
        alpha, source = resolve_attack_step_size(
            norm="l2",
            epsilon=4.0,
            num_steps=40,
            step_size=None,
            step_size_multiplier=2.0,
        )
        self.assertEqual(source, "epsilon_scaled")
        self.assertAlmostEqual(alpha, 0.2, places=7)

    def test_resolve_attack_step_size_handles_zero_epsilon(self):
        alpha, source = resolve_attack_step_size(
            norm="l2",
            epsilon=0.0,
            num_steps=40,
            step_size=None,
            step_size_multiplier=2.0,
        )
        self.assertEqual(source, "epsilon_scaled")
        self.assertEqual(alpha, 0.0)

    def test_resolve_attack_step_size_rejects_nonpositive_num_steps(self):
        with self.assertRaises(ValueError):
            resolve_attack_step_size(
                norm="l2",
                epsilon=1.0,
                num_steps=0,
                step_size=None,
                step_size_multiplier=2.0,
            )

    def test_parse_sweep_attack_id(self):
        parsed = parse_sweep_attack_id("adversarial_l2_eps_1p6_steps_100_m_0p5")
        self.assertEqual(parsed["norm"], "l2")
        self.assertAlmostEqual(float(parsed["epsilon"]), 1.6, places=7)
        self.assertEqual(int(parsed["num_steps"]), 100)
        self.assertAlmostEqual(float(parsed["step_size_multiplier"]), 0.5, places=7)

    def test_monotonicity_violations_detect_increase(self):
        violations = _compute_monotonicity_violations(
            [(0.0, 0.45), (0.8, 0.10), (1.6, 0.11), (2.4, 0.08)]
        )
        self.assertEqual(len(violations), 1)
        self.assertAlmostEqual(violations[0]["epsilon_prev"], 0.8, places=7)
        self.assertAlmostEqual(violations[0]["epsilon_curr"], 1.6, places=7)

    def test_select_best_by_epsilon_chooses_lowest_adversarial_map(self):
        rows = [
            {
                "epsilon": 0.8,
                "adversarial_map": 0.11,
                "mean_adversarial_loss": 0.9,
                "num_steps": 20,
                "step_size_multiplier": 1.0,
            },
            {
                "epsilon": 0.8,
                "adversarial_map": 0.07,
                "mean_adversarial_loss": 0.8,
                "num_steps": 40,
                "step_size_multiplier": 2.0,
            },
            {
                "epsilon": 1.6,
                "adversarial_map": 0.05,
                "mean_adversarial_loss": 1.1,
                "num_steps": 100,
                "step_size_multiplier": 4.0,
            },
        ]
        best = select_best_by_epsilon(rows)
        self.assertEqual(len(best), 2)
        self.assertAlmostEqual(float(best[0]["epsilon"]), 0.8, places=7)
        self.assertAlmostEqual(float(best[0]["adversarial_map"]), 0.07, places=7)

    def test_select_best_m_by_ranked_vote_picks_global_winner(self):
        rows = [
            {
                "epsilon": 0.8,
                "adversarial_map": 0.10,
                "mean_adversarial_loss": 0.9,
                "num_steps": 20,
                "step_size_multiplier": 0.5,
                "alpha": 0.02,
            },
            {
                "epsilon": 0.8,
                "adversarial_map": 0.07,
                "mean_adversarial_loss": 0.8,
                "num_steps": 40,
                "step_size_multiplier": 2.0,
                "alpha": 0.04,
            },
            {
                "epsilon": 0.8,
                "adversarial_map": 0.09,
                "mean_adversarial_loss": 0.85,
                "num_steps": 100,
                "step_size_multiplier": 4.0,
                "alpha": 0.016,
            },
            {
                "epsilon": 1.6,
                "adversarial_map": 0.08,
                "mean_adversarial_loss": 1.0,
                "num_steps": 20,
                "step_size_multiplier": 0.5,
                "alpha": 0.04,
            },
            {
                "epsilon": 1.6,
                "adversarial_map": 0.06,
                "mean_adversarial_loss": 1.1,
                "num_steps": 40,
                "step_size_multiplier": 2.0,
                "alpha": 0.08,
            },
            {
                "epsilon": 1.6,
                "adversarial_map": 0.07,
                "mean_adversarial_loss": 1.05,
                "num_steps": 100,
                "step_size_multiplier": 4.0,
                "alpha": 0.032,
            },
        ]
        vote = select_best_m_by_ranked_vote(rows)
        self.assertAlmostEqual(float(vote["best_step_size_multiplier"]), 2.0, places=7)
        totals = {
            float(item["step_size_multiplier"]): float(item["total_points"])
            for item in vote["vote_totals"]
        }
        self.assertEqual(totals[2.0], 4.0)
        self.assertEqual(totals[4.0], 2.0)
        self.assertEqual(totals[0.5], 0.0)

    def test_select_best_by_epsilon_with_fixed_m_picks_best_num_steps(self):
        rows = [
            {
                "epsilon": 0.8,
                "adversarial_map": 0.10,
                "mean_adversarial_loss": 0.9,
                "num_steps": 20,
                "step_size_multiplier": 2.0,
                "alpha": 0.08,
            },
            {
                "epsilon": 0.8,
                "adversarial_map": 0.07,
                "mean_adversarial_loss": 0.8,
                "num_steps": 40,
                "step_size_multiplier": 2.0,
                "alpha": 0.04,
            },
            {
                "epsilon": 1.6,
                "adversarial_map": 0.06,
                "mean_adversarial_loss": 1.1,
                "num_steps": 100,
                "step_size_multiplier": 2.0,
                "alpha": 0.032,
            },
        ]
        best = select_best_by_epsilon_with_fixed_m(rows, step_size_multiplier=2.0)
        self.assertEqual(len(best), 2)
        self.assertEqual(int(best[0]["num_steps"]), 40)
        self.assertEqual(int(best[1]["num_steps"]), 100)

    def test_compute_map_excludes_classes_without_positives(self):
        scores = np.array(
            [
                [0.9, 0.8, 0.1],
                [0.1, 0.7, 0.2],
                [0.8, 0.2, 0.3],
            ],
            dtype=np.float32,
        )
        targets = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        result = compute_map_from_scores(scores, targets)
        self.assertEqual(result["num_positive_classes"], 2)
        self.assertTrue(np.isnan(result["per_class_ap"][2]))
        self.assertGreater(result["map"], 0.9)

    def test_average_precision_fallback_works_without_sklearn(self):
        old_impl = _MAP_EVAL._sk_average_precision_score
        try:
            setattr(_MAP_EVAL, "_sk_average_precision_score", None)
            y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
            y_score = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
            ap = _average_precision_binary(y_true, y_score)
            self.assertGreater(ap, 0.75)
            self.assertLessEqual(ap, 1.0)
        finally:
            setattr(_MAP_EVAL, "_sk_average_precision_score", old_impl)

    def test_shape_mismatch_raises(self):
        scores = np.zeros((2, 4), dtype=np.float32)
        targets = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_map_from_scores(scores, targets)


if __name__ == "__main__":
    unittest.main()
