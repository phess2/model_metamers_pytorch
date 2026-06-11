#!/usr/bin/env python
"""
Nightly automation for modular L-infinity bound search.

Subcommands:
  propose  - pick the next untried bound config and write READY_FOR_MORNING.json
  evaluate - run PGD sweep + compare a trained trial against the robust baseline
  status   - print search progress
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.robustness_search.compare import compare_against_baseline, save_comparison_report
from src.robustness_search.search import propose_next_trial
from src.robustness_search.state import SearchState, load_state, save_state
from src.utils.config import load_config


DEFAULT_SEARCH_SPACE = "configs/nightly/search_space.json"


def _derive_experiment_root(config_path: str, exp_dir: str = "experiments") -> Path:
    from src.analysis.metamer import derive_experiment_root

    return derive_experiment_root(
        config_path=config_path,
        exp_dir=exp_dir,
        config_root="configs",
    )


def _default_ckpt_path(config_path: str, exp_dir: str = "experiments") -> Path:
    return _derive_experiment_root(config_path, exp_dir=exp_dir) / "checkpoints" / "last.ckpt"


def _run_pgd_sweep(
    config_path: str,
    ckpt_path: str,
    *,
    search_space_path: str,
    device: str,
) -> None:
    spec = load_config(search_space_path)
    pgd = spec["pgd_eval"]
    epsilons = [str(x) for x in pgd["epsilons"]]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sweep_pgd_eval.py"),
        "--config",
        config_path,
        "--ckpt_path",
        ckpt_path,
        "--epsilons",
        *epsilons,
        "--num_steps",
        str(pgd.get("num_steps", 100)),
        "--step_size",
        str(pgd.get("step_size", 0.1)),
        "--indices",
        str(pgd.get("indices", "0-399")),
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def phase_propose(args: argparse.Namespace) -> int:
    result = propose_next_trial(args.search_space, force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "ready_for_morning":
        print("\nMorning training command:")
        print(result["train_command"])
        return 0
    return 1 if result.get("status") == "search_exhausted" else 0


def phase_evaluate(args: argparse.Namespace) -> int:
    spec = load_config(args.search_space)
    state_path = Path(spec["output"]["state_dir"]) / "state.json"
    state = load_state(state_path)

    if args.trial_id:
        trial = state.get_trial(args.trial_id)
        if trial is None:
            print(f"ERROR: Unknown trial_id '{args.trial_id}'", file=sys.stderr)
            return 1
        config_path = trial.config_path
        trial_id = trial.trial_id
    elif args.config:
        config_path = args.config
        trial_id = Path(config_path).stem
    else:
        print("ERROR: provide --trial-id or --config", file=sys.stderr)
        return 1

    ckpt_path = args.ckpt_path or str(_default_ckpt_path(config_path, args.exp_dir))
    if not Path(ckpt_path).exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    print(f"Running PGD sweep for {config_path}")
    _run_pgd_sweep(
        config_path,
        ckpt_path,
        search_space_path=args.search_space,
        device=args.device,
    )

    baseline_config = spec["goal"]["baseline_config"]
    baseline_ckpt = args.baseline_ckpt
    if baseline_ckpt is None:
        baseline_ckpt = str(_default_ckpt_path(baseline_config, args.exp_dir))
    if Path(baseline_ckpt).exists():
        print(f"Running PGD sweep for baseline {baseline_config}")
        _run_pgd_sweep(
            baseline_config,
            baseline_ckpt,
            search_space_path=args.search_space,
            device=args.device,
        )
    else:
        print(
            f"WARNING: baseline checkpoint missing at {baseline_ckpt}; "
            "comparison will fail unless baseline CSVs already exist."
        )

    report = compare_against_baseline(
        config_path,
        search_space_path=args.search_space,
        exp_dir=args.exp_dir,
    )
    report_path = (
        Path(spec["output"]["state_dir"])
        / "results"
        / f"{trial_id}_comparison.json"
    )
    save_comparison_report(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nSaved comparison report to {report_path}")

    trial = state.get_trial(trial_id)
    if trial is not None:
        trial.status = "evaluated"
        trial.evaluated_at = datetime.now(timezone.utc).isoformat()
        trial.metrics = {
            "comparison_report": str(report_path),
            "passed_goal": report["passed"],
            "candidate_clean_accuracy": report.get("candidate_clean_accuracy"),
            "baseline_clean_accuracy": report.get("baseline_clean_accuracy"),
        }
        trial.passed_goal = bool(report["passed"])
        if report["passed"]:
            state.goal_met = True
            state.best_trial_id = trial_id
        save_state(state_path, state)

    return 0 if report["passed"] else 2


def phase_status(args: argparse.Namespace) -> int:
    spec = load_config(args.search_space)
    state_path = Path(spec["output"]["state_dir"]) / "state.json"
    state: SearchState = load_state(state_path)
    ready_path = Path(spec["output"]["ready_for_morning_file"])

    summary = {
        "goal_met": state.goal_met,
        "best_trial_id": state.best_trial_id,
        "num_trials": len(state.trials),
        "num_evaluated": sum(1 for t in state.trials if t.status == "evaluated"),
        "num_passed": sum(1 for t in state.trials if t.passed_goal),
        "ready_for_morning_exists": ready_path.exists(),
    }
    if ready_path.exists():
        with open(ready_path, "r", encoding="utf-8") as handle:
            summary["ready_for_morning"] = json.load(handle)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nightly modular L-inf bound search")
    parser.add_argument(
        "--search-space",
        default=DEFAULT_SEARCH_SPACE,
        help="Path to search space JSON",
    )
    parser.add_argument(
        "--exp_dir",
        default="experiments",
        help="Experiment root directory",
    )
    parser.add_argument("--device", default="cuda", help="Device for PGD evaluation")

    subparsers = parser.add_subparsers(dest="phase", required=True)

    propose_parser = subparsers.add_parser(
        "propose", help="Propose the next trial and write READY_FOR_MORNING.json"
    )
    propose_parser.add_argument(
        "--force",
        action="store_true",
        help="Propose a new trial even if the search goal was already met",
    )

    eval_parser = subparsers.add_parser(
        "evaluate", help="Run PGD sweep and compare a trained trial"
    )
    eval_parser.add_argument("--trial-id", type=str, default=None)
    eval_parser.add_argument("--config", type=str, default=None)
    eval_parser.add_argument("--ckpt_path", type=str, default=None)
    eval_parser.add_argument("--baseline-ckpt", type=str, default=None)

    subparsers.add_parser("status", help="Print search progress")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.phase == "propose":
        return phase_propose(args)
    if args.phase == "evaluate":
        return phase_evaluate(args)
    if args.phase == "status":
        return phase_status(args)
    parser.error(f"Unknown phase: {args.phase}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
