import json
import tempfile
import unittest
from pathlib import Path

from src.robustness_search.compare import compare_against_baseline, load_pgd_curve
from src.robustness_search.config_builder import build_trial_config
from src.robustness_search.search import propose_next_trial
from src.robustness_search.state import trial_signature
from src.utils.config import load_config


class NightlyBoundSearchTests(unittest.TestCase):
    def test_build_trial_config_sets_bound_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "trial.json"
            build_trial_config(
                template_config_path="configs/vision/lipsresnet50_v1/lipsresnet50_w_max_0_SGD.json",
                output_path=output,
                trial_id="test_trial",
                hparams={
                    "model_name": "lipsresnet",
                    "bound_method": "modular_linf",
                    "projection": "modular_linf_cap",
                    "w_max": 2.0,
                    "optimizer_lr": 0.2,
                },
            )
            config = load_config(output)
            self.assertEqual(config["hparams"]["bound_method"], "modular_linf")
            self.assertEqual(config["hparams"]["projection"], "modular_linf_cap")
            self.assertEqual(config["hparams"]["w_max"], 2.0)

    def test_trial_signature_is_stable(self):
        sig = trial_signature(
            {
                "model_name": "lipsresnet",
                "bound_method": "modular_linf",
                "projection": "modular_linf_cap",
                "w_max": 2.0,
                "optimizer_lr": 0.2,
            }
        )
        self.assertIn("bound_method=modular_linf", sig)

    def test_propose_next_trial_writes_ready_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            search_space = {
                "goal": {
                    "baseline_config": "configs/vision/robust_v1/resnet50_l2_3_robust.json",
                    "clean_acc_tolerance": 0.02,
                    "pgd_epsilons": [0, 1, 2, 3],
                    "beat_epsilons": [1, 2, 3],
                    "require_beat_at_all_eps": True,
                },
                "template_config": "configs/vision/lipsresnet50_v1/lipsresnet50_w_max_0_SGD.json",
                "search_space": {
                    "model_name": ["lipsresnet"],
                    "bound_method": ["modular_linf"],
                    "projection": {"modular_linf": ["modular_linf_cap"]},
                    "w_max": [1.0],
                    "optimizer_lr": [0.2],
                },
                "pgd_eval": {
                    "epsilons": [0, 1],
                    "num_steps": 5,
                    "step_size": 0.1,
                    "indices": "0-1",
                },
                "output": {
                    "generated_config_dir": str(tmp / "generated"),
                    "state_dir": str(tmp / "state"),
                    "ready_for_morning_file": str(tmp / "state" / "READY_FOR_MORNING.json"),
                },
            }
            search_path = tmp / "search_space.json"
            search_path.write_text(json.dumps(search_space), encoding="utf-8")

            result = propose_next_trial(search_path)
            self.assertEqual(result["status"], "ready_for_morning")
            self.assertTrue(Path(result["config_path"]).exists())
            self.assertTrue(Path(search_space["output"]["ready_for_morning_file"]).exists())

    def test_compare_against_baseline_with_mock_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            candidate_config = tmp / "candidate.json"
            candidate_config.write_text(
                json.dumps({"model_name": "lipsresnet", "hparams": {}}),
                encoding="utf-8",
            )
            baseline_config = tmp / "baseline.json"
            baseline_config.write_text(
                json.dumps({"model_name": "robustresnet50", "hparams": {}}),
                encoding="utf-8",
            )

            def write_csv(config_path: Path, epsilon: float, accuracy: float) -> None:
                from src.robustness_search.compare import adversarial_csv_path

                csv_path = adversarial_csv_path(
                    config_path,
                    epsilon,
                    exp_dir=str(tmp / "experiments"),
                    config_root=str(tmp),
                )
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_path.write_text(
                    "sample_idx,is_correct,true_class_softmax\n"
                    f"0,{int(accuracy >= 0.5)},{accuracy}\n"
                    f"1,{int(accuracy >= 0.5)},{accuracy}\n",
                    encoding="utf-8",
                )

            for eps, acc in [(0.0, 0.8), (1.0, 0.5), (2.0, 0.4), (3.0, 0.3)]:
                write_csv(baseline_config, eps, acc)
            for eps, acc in [(0.0, 0.79), (1.0, 0.55), (2.0, 0.45), (3.0, 0.35)]:
                write_csv(candidate_config, eps, acc)

            search_space = {
                "goal": {
                    "baseline_config": str(baseline_config),
                    "clean_acc_tolerance": 0.02,
                    "beat_epsilons": [1.0, 2.0, 3.0],
                    "require_beat_at_all_eps": True,
                },
                "pgd_eval": {"epsilons": [0, 1, 2, 3]},
            }
            search_path = tmp / "search_space.json"
            search_path.write_text(json.dumps(search_space), encoding="utf-8")

            report = compare_against_baseline(
                candidate_config,
                search_space_path=search_path,
                exp_dir=str(tmp / "experiments"),
            )
            self.assertTrue(report["passed"])

            curve = load_pgd_curve(
                candidate_config,
                [0.0, 1.0],
                exp_dir=str(tmp / "experiments"),
                config_root=str(tmp),
            )
            self.assertIn(0.0, curve)


if __name__ == "__main__":
    unittest.main()
