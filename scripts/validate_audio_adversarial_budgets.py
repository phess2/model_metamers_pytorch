"""Validate perturbation budgets across saved audio adversarial runs.

Scans ``<run_root>/<model>/adversarial/adversarial_<norm>_eps_*/logits/metadata.jsonl``
and checks, per model x epsilon cell, that every attack:

  - does not exceed its epsilon budget (overshoot; norm > epsilon + tolerance), and
  - does not undershoot it (budget utilization below a threshold, epsilon > 0 only).

Writes ``<run_root>/budget_validation.csv`` and exits non-zero on any violation so
it can gate downstream analysis.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch


def parse_epsilon_from_attack_id(attack_id: str) -> float:
    # The epsilon token follows "_eps_" and ends at the next underscore, so any
    # trailing attack-scope/mode suffix (e.g. "_suppress", "_single_label") is ignored.
    token = attack_id.split("_eps_")[-1].split("_")[0]
    return float(token.replace("p", "."))


def load_metadata_records(metadata_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_model_cutoff_hz(model_name: str) -> int | None:
    """Resolve the model-observable Nyquist cutoff without loading a model."""
    normalized = model_name.strip().lower()
    if normalized.startswith(("audiomae", "beats")):
        return 8_000
    if normalized.startswith("panns"):
        return 16_000
    if normalized.startswith("clap"):
        return 24_000
    return None


def _resolve_saved_path(raw_path: object, metadata_dir: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    local_path = metadata_dir / path.name
    return local_path if local_path.exists() else None


def _active_norm(delta: torch.Tensor, norm: str) -> float:
    flattened = delta.reshape(delta.shape[0], -1)
    if norm == "l2":
        values = torch.linalg.vector_norm(flattened, ord=2, dim=1)
    elif norm == "linf":
        values = flattened.abs().amax(dim=1)
    else:
        raise ValueError(f"Unsupported norm={norm!r}.")
    return float(values.max().item())


def _high_frequency_energy_ratio(
    delta: torch.Tensor,
    *,
    sample_rate: int,
    cutoff_hz: float,
) -> float:
    if delta.shape[-1] <= 1 or cutoff_hz >= sample_rate / 2.0:
        return 0.0
    flattened = delta.reshape(-1, delta.shape[-1])
    spectrum = torch.fft.rfft(flattened, dim=-1)
    energy = spectrum.abs().square()
    frequencies = torch.fft.rfftfreq(delta.shape[-1], d=1.0 / float(sample_rate))
    total = energy.sum()
    if float(total.item()) == 0.0:
        return 0.0
    return float((energy[:, frequencies > cutoff_hz].sum() / total).item())


def recompute_saved_artifact_stats(
    record: dict[str, Any],
    *,
    metadata_dir: Path,
    model_name: str,
    norm: str,
) -> dict[str, float | bool] | None:
    """Recompute budget and spectral statistics from saved PT tensors."""
    saved_paths = record.get("saved_paths")
    if not isinstance(saved_paths, dict):
        return None
    original_path = _resolve_saved_path(saved_paths.get("original_pt"), metadata_dir)
    adversarial_path = _resolve_saved_path(
        saved_paths.get("adversarial_pt"), metadata_dir
    )
    if original_path is None or adversarial_path is None:
        return None

    original = torch.load(original_path, map_location="cpu", weights_only=True)
    adversarial = torch.load(adversarial_path, map_location="cpu", weights_only=True)
    if not isinstance(original, torch.Tensor) or not isinstance(
        adversarial, torch.Tensor
    ):
        raise TypeError("Saved original/adversarial artifacts must be tensors.")
    if original.shape != adversarial.shape:
        raise ValueError(
            f"Saved tensor shape mismatch: {original.shape} vs {adversarial.shape}"
        )

    delta = adversarial - original
    max_norm = _active_norm(delta, norm)
    sample_rate = int(record["sample_rate"])
    cutoff_hz = resolve_model_cutoff_hz(model_name)
    high_frequency_ratio = (
        _high_frequency_energy_ratio(
            delta,
            sample_rate=sample_rate,
            cutoff_hz=float(cutoff_hz),
        )
        if cutoff_hz is not None
        else 0.0
    )
    recorded_budget = record.get("budget_verification_post_lowpass")
    recorded_norm = (
        float(recorded_budget["max_active_norm"])
        if isinstance(recorded_budget, dict) and "max_active_norm" in recorded_budget
        else float("nan")
    )
    return {
        "max_active_norm": max_norm,
        "recorded_max_active_norm": recorded_norm,
        "high_frequency_energy_ratio": high_frequency_ratio,
    }


def extract_budget_stats(
    record: dict[str, Any],
    *,
    stage: str,
) -> dict[str, float | None] | None:
    """Pull max_active_norm/epsilon for one record at stage 'pre_lowpass' or 'post_lowpass'."""
    budget = record.get(f"budget_verification_{stage}")
    if not isinstance(budget, dict) or "max_active_norm" not in budget:
        return None
    epsilon = float(budget["epsilon"])
    max_norm = float(budget["max_active_norm"])
    utilization = (max_norm / epsilon) if epsilon > 0.0 else None
    return {
        "epsilon": epsilon,
        "max_active_norm": max_norm,
        "utilization": utilization,
    }


def summarize_attack_dir(
    *,
    records: list[dict[str, Any]],
    epsilon: float,
    stage: str,
    overshoot_tolerance: float,
    undershoot_threshold: float,
) -> dict[str, Any]:
    """Aggregate budget stats for one attack directory at one filtering stage."""
    norms: list[float] = []
    utilizations: list[float] = []
    num_overshoot = 0
    num_undershoot = 0
    num_missing = 0
    for record in records:
        stats = extract_budget_stats(record, stage=stage)
        if stats is None:
            num_missing += 1
            continue
        max_norm = float(stats["max_active_norm"])  # type: ignore[arg-type]
        norms.append(max_norm)
        if max_norm > epsilon + overshoot_tolerance:
            num_overshoot += 1
        if stats["utilization"] is not None:
            utilization = float(stats["utilization"])
            utilizations.append(utilization)
            if utilization < undershoot_threshold:
                num_undershoot += 1

    def _agg(values: list[float], fn: Any) -> float:
        return float(fn(values)) if values else float("nan")

    return {
        "stage": stage,
        "num_records": len(records),
        "num_missing_budget": num_missing,
        "num_overshoot": num_overshoot,
        "num_undershoot": num_undershoot,
        "min_norm": _agg(norms, min),
        "max_norm": _agg(norms, max),
        "mean_norm": _agg(norms, lambda v: sum(v) / len(v)),
        "min_utilization": _agg(utilizations, min),
        "max_utilization": _agg(utilizations, max),
        "mean_utilization": _agg(utilizations, lambda v: sum(v) / len(v)),
    }


def summarize_saved_artifacts(
    *,
    records: list[dict[str, Any]],
    metadata_dir: Path,
    model_name: str,
    norm: str,
    epsilon: float,
    overshoot_tolerance: float,
    undershoot_threshold: float,
    metadata_tolerance: float,
    max_high_frequency_ratio: float,
) -> dict[str, Any]:
    """Aggregate independently recomputed saved-tensor budget statistics."""
    norms: list[float] = []
    utilizations: list[float] = []
    high_frequency_ratios: list[float] = []
    num_overshoot = 0
    num_undershoot = 0
    num_missing = 0
    num_metadata_mismatch = 0
    num_nyquist_violation = 0

    for record in records:
        stats = recompute_saved_artifact_stats(
            record,
            metadata_dir=metadata_dir,
            model_name=model_name,
            norm=norm,
        )
        if stats is None:
            num_missing += 1
            continue
        max_norm = float(stats["max_active_norm"])
        recorded_norm = float(stats["recorded_max_active_norm"])
        high_frequency_ratio = float(stats["high_frequency_energy_ratio"])
        norms.append(max_norm)
        high_frequency_ratios.append(high_frequency_ratio)
        if max_norm > epsilon + overshoot_tolerance:
            num_overshoot += 1
        if epsilon > 0.0:
            utilization = max_norm / epsilon
            utilizations.append(utilization)
            if utilization < undershoot_threshold:
                num_undershoot += 1
        if (
            not math.isfinite(recorded_norm)
            or abs(max_norm - recorded_norm) > metadata_tolerance
        ):
            num_metadata_mismatch += 1
        if high_frequency_ratio > max_high_frequency_ratio:
            num_nyquist_violation += 1

    def _agg(values: list[float], fn: Any) -> float:
        return float(fn(values)) if values else float("nan")

    return {
        "stage": "saved_artifact",
        "num_records": len(records),
        "num_missing_budget": num_missing,
        "num_overshoot": num_overshoot,
        "num_undershoot": num_undershoot,
        "num_metadata_mismatch": num_metadata_mismatch,
        "num_nyquist_violation": num_nyquist_violation,
        "min_norm": _agg(norms, min),
        "max_norm": _agg(norms, max),
        "mean_norm": _agg(norms, lambda v: sum(v) / len(v)),
        "min_utilization": _agg(utilizations, min),
        "max_utilization": _agg(utilizations, max),
        "mean_utilization": _agg(utilizations, lambda v: sum(v) / len(v)),
        "max_high_frequency_energy_ratio": _agg(high_frequency_ratios, max),
    }


def discover_attack_dirs(
    *,
    run_root: Path,
    norm: str,
) -> list[tuple[str, float, Path]]:
    """Find (model_name, epsilon, attack_dir) triples under a run root."""
    discovered: list[tuple[str, float, Path]] = []
    for metadata_path in sorted(
        run_root.glob(f"*/adversarial/adversarial_{norm}_eps_*/logits/metadata.jsonl")
    ):
        attack_dir = metadata_path.parent.parent
        model_name = attack_dir.parent.parent.name
        try:
            epsilon = parse_epsilon_from_attack_id(attack_dir.name)
        except ValueError:
            continue
        discovered.append((model_name, epsilon, attack_dir))
    return discovered


def main() -> None:
    parser = ArgumentParser(
        description="Validate audio adversarial perturbation budgets."
    )
    parser.add_argument(
        "--run_root",
        type=str,
        required=True,
        help="Run root containing <model>/adversarial/adversarial_<norm>_eps_*/ dirs.",
    )
    parser.add_argument("--norm", type=str, choices=["l2", "linf"], default="l2")
    parser.add_argument(
        "--overshoot_tolerance",
        type=float,
        default=1e-4,
        help="Absolute norm tolerance above epsilon before flagging an overshoot.",
    )
    parser.add_argument(
        "--undershoot_threshold",
        type=float,
        default=0.9,
        help="Minimum budget utilization (norm/epsilon) before flagging an undershoot.",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=["saved_artifact"],
        choices=["pre_lowpass", "post_lowpass", "saved_artifact"],
    )
    parser.add_argument(
        "--fail_stage",
        type=str,
        default="saved_artifact",
        choices=["pre_lowpass", "post_lowpass", "saved_artifact"],
        help=(
            "Stage whose violations cause a non-zero exit. The default independently "
            "recomputes statistics from saved tensor artifacts."
        ),
    )
    parser.add_argument(
        "--metadata_tolerance",
        type=float,
        default=1e-5,
        help="Maximum absolute mismatch between recomputed and recorded norm.",
    )
    parser.add_argument(
        "--max_high_frequency_ratio",
        type=float,
        default=1e-6,
        help="Maximum perturbation FFT energy fraction above model cutoff.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path (defaults to <run_root>/budget_validation.csv).",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    attack_dirs = discover_attack_dirs(run_root=run_root, norm=args.norm)
    if not attack_dirs:
        raise SystemExit(f"No attack metadata found under {run_root}")

    rows: list[dict[str, Any]] = []
    num_failures = 0
    for model_name, epsilon, attack_dir in attack_dirs:
        records = load_metadata_records(attack_dir / "logits" / "metadata.jsonl")
        for stage in args.stages:
            if stage == "saved_artifact":
                summary = summarize_saved_artifacts(
                    records=records,
                    metadata_dir=attack_dir / "logits",
                    model_name=model_name,
                    norm=args.norm,
                    epsilon=epsilon,
                    overshoot_tolerance=args.overshoot_tolerance,
                    undershoot_threshold=args.undershoot_threshold,
                    metadata_tolerance=args.metadata_tolerance,
                    max_high_frequency_ratio=args.max_high_frequency_ratio,
                )
            else:
                summary = summarize_attack_dir(
                    records=records,
                    epsilon=epsilon,
                    stage=stage,
                    overshoot_tolerance=args.overshoot_tolerance,
                    undershoot_threshold=args.undershoot_threshold,
                )
            has_violation = summary["num_overshoot"] > 0 or (
                epsilon > 0.0 and summary["num_undershoot"] > 0
            )
            has_violation = has_violation or any(
                int(summary.get(key, 0)) > 0
                for key in (
                    "num_missing_budget",
                    "num_metadata_mismatch",
                    "num_nyquist_violation",
                )
            )
            if stage == args.fail_stage and has_violation:
                num_failures += 1
            row = {
                "model_name": model_name,
                "attack_id": attack_dir.name,
                "epsilon": epsilon,
                "has_violation": has_violation,
                **summary,
            }
            rows.append(row)
            print(
                f"[{'FAIL' if has_violation else 'ok'}] {model_name} eps={epsilon:g} "
                f"stage={stage}: overshoot={summary['num_overshoot']}, "
                f"undershoot={summary['num_undershoot']}, "
                f"missing={summary['num_missing_budget']}, "
                f"metadata_mismatch={summary.get('num_metadata_mismatch', 0)}, "
                f"nyquist={summary.get('num_nyquist_violation', 0)}, "
                f"mean_util="
                + (
                    f"{summary['mean_utilization']:.4f}"
                    if not math.isnan(float(summary["mean_utilization"]))
                    else "n/a"
                )
            )

    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else run_root / "budget_validation.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "attack_id",
        "epsilon",
        "stage",
        "has_violation",
        "num_records",
        "num_missing_budget",
        "num_overshoot",
        "num_undershoot",
        "num_metadata_mismatch",
        "num_nyquist_violation",
        "min_norm",
        "max_norm",
        "mean_norm",
        "min_utilization",
        "max_utilization",
        "mean_utilization",
        "max_high_frequency_energy_ratio",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")

    if num_failures > 0:
        print(
            f"ERROR: {num_failures} model x epsilon cells violated the budget "
            f"at stage {args.fail_stage}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("All budgets validated.")


if __name__ == "__main__":
    main()
