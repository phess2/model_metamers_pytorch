"""Compare PGD robustness curves against a robust baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.utils.config import load_config


def derive_experiment_root(
    config_path: str | Path,
    exp_dir: str | Path = "experiments",
    config_root: str | Path = "configs",
) -> Path:
    """Build experiment root path using config-relative structure when possible."""
    config_path_obj = Path(config_path).resolve()
    exp_dir_obj = Path(exp_dir)
    config_root_obj = Path(config_root).resolve()
    try:
        rel = config_path_obj.relative_to(config_root_obj)
        return exp_dir_obj / rel.with_suffix("")
    except ValueError:
        return exp_dir_obj / config_path_obj.stem


def epsilon_to_filename_token(epsilon: float) -> str:
    return format(epsilon, "g").replace(".", "p")


def adversarial_csv_path(
    config_path: str | Path,
    epsilon: float,
    *,
    exp_dir: str = "experiments",
    config_root: str = "configs",
) -> Path:
    exp_root = derive_experiment_root(
        config_path=str(config_path),
        exp_dir=exp_dir,
        config_root=config_root,
    )
    token = epsilon_to_filename_token(epsilon)
    return exp_root / "adversarial" / f"adversarial_l2_eps_{token}.csv"


def load_pgd_curve(
    config_path: str | Path,
    epsilons: list[float],
    *,
    exp_dir: str = "experiments",
    config_root: str = "configs",
) -> dict[float, dict[str, float]]:
    curve: dict[float, dict[str, float]] = {}
    for epsilon in epsilons:
        csv_path = adversarial_csv_path(
            config_path,
            epsilon,
            exp_dir=exp_dir,
            config_root=config_root,
        )
        if not csv_path.exists():
            continue
        rows = []
        with open(csv_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        correct = [float(row["is_correct"]) for row in rows]
        confidence = [float(row["true_class_softmax"]) for row in rows]
        curve[float(epsilon)] = {
            "accuracy": sum(correct) / len(correct),
            "mean_confidence": sum(confidence) / len(confidence),
            "num_samples": len(rows),
            "csv_path": str(csv_path),
        }
    return curve


def compare_against_baseline(
    candidate_config_path: str | Path,
    *,
    search_space_path: str | Path,
    exp_dir: str = "experiments",
) -> dict[str, Any]:
    """Check whether a trained candidate beats the PGD baseline at matched clean accuracy."""
    spec = load_config(search_space_path)
    goal = spec["goal"]
    epsilons = [float(x) for x in spec["pgd_eval"]["epsilons"]]
    beat_epsilons = [float(x) for x in goal["beat_epsilons"]]
    tolerance = float(goal["clean_acc_tolerance"])

    baseline_config = goal["baseline_config"]
    baseline_curve = load_pgd_curve(baseline_config, epsilons, exp_dir=exp_dir)
    candidate_curve = load_pgd_curve(candidate_config_path, epsilons, exp_dir=exp_dir)

    baseline_clean = baseline_curve.get(0.0, {}).get("accuracy")
    candidate_clean = candidate_curve.get(0.0, {}).get("accuracy")

    result: dict[str, Any] = {
        "baseline_config": baseline_config,
        "candidate_config": str(candidate_config_path),
        "baseline_curve": baseline_curve,
        "candidate_curve": candidate_curve,
        "baseline_clean_accuracy": baseline_clean,
        "candidate_clean_accuracy": candidate_clean,
        "clean_acc_tolerance": tolerance,
        "beat_epsilons": beat_epsilons,
        "passed": False,
        "reasons": [],
    }

    if baseline_clean is None:
        result["reasons"].append("Missing baseline PGD results at epsilon=0.")
        return result
    if candidate_clean is None:
        result["reasons"].append("Missing candidate PGD results at epsilon=0.")
        return result

    clean_gap = abs(candidate_clean - baseline_clean)
    result["clean_accuracy_gap"] = clean_gap
    if clean_gap > tolerance:
        result["reasons"].append(
            f"Clean accuracy gap {clean_gap:.4f} exceeds tolerance {tolerance:.4f}."
        )
        return result

    beats = []
    for epsilon in beat_epsilons:
        baseline_acc = baseline_curve.get(epsilon, {}).get("accuracy")
        candidate_acc = candidate_curve.get(epsilon, {}).get("accuracy")
        if baseline_acc is None:
            result["reasons"].append(f"Missing baseline PGD results at epsilon={epsilon}.")
            return result
        if candidate_acc is None:
            result["reasons"].append(f"Missing candidate PGD results at epsilon={epsilon}.")
            return result
        beat = candidate_acc >= baseline_acc
        beats.append(
            {
                "epsilon": epsilon,
                "baseline_accuracy": baseline_acc,
                "candidate_accuracy": candidate_acc,
                "beat_baseline": beat,
            }
        )

    result["epsilon_comparisons"] = beats
    if goal.get("require_beat_at_all_eps", True):
        passed = all(item["beat_baseline"] for item in beats)
    else:
        passed = any(item["beat_baseline"] for item in beats)

    result["passed"] = passed
    if passed:
        result["reasons"].append("Candidate beats baseline at required epsilons with similar clean accuracy.")
    else:
        result["reasons"].append("Candidate does not beat baseline at all required epsilons.")
    return result


def save_comparison_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path
