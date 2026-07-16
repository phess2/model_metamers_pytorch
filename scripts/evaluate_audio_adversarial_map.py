"""Evaluate mAP from saved audio adversarial outputs without rerunning attacks."""

from __future__ import annotations

import csv
import json
import math
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

try:
    from sklearn.metrics import average_precision_score as _sk_average_precision_score
except ModuleNotFoundError:  # pragma: no cover - exercised by fallback path tests
    _sk_average_precision_score = None


def _parse_epsilon_from_attack_id(attack_id: str) -> float:
    # Take only the leading numeric token after "_eps_" so trailing suffixes such
    # as "_suppress" or "_single_label" do not break epsilon parsing/discovery.
    token = attack_id.split("_eps_")[-1].split("_")[0]
    return float(token.replace("p", "."))


def _load_metadata_records(metadata_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _resolve_saved_path(path_value: str, workspace_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return workspace_root / candidate


def _extract_true_labels(
    *,
    record: dict[str, Any],
    model_label_mids: list[str] | None,
) -> list[int]:
    from src.analysis.audio_classification import remap_audioset_labels_to_model_indices

    source_info = record.get("source_info", {})
    explicit = source_info.get("labels_for_scoring")
    if explicit:
        return [int(x) for x in explicit]
    labels = source_info.get("labels", [])
    if not labels:
        return []
    return remap_audioset_labels_to_model_indices(
        true_labels=[int(x) for x in labels],
        model_label_mids=model_label_mids,
    )


def _sigmoid_logits(logits: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)


def _average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AP for one class; assumes at least one positive label exists."""
    if _sk_average_precision_score is not None:
        return float(_sk_average_precision_score(y_true, y_score))

    # Fallback implementation equivalent to rank-based area under PR curve.
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order].astype(np.float64, copy=False)
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1.0 - y_true_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    total_positives = float(np.sum(y_true_sorted))
    if total_positives <= 0.0:
        raise ValueError("AP requested with zero positives.")
    return float(np.sum(precision * y_true_sorted) / total_positives)


def compute_map_from_scores(
    scores: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    """
    Compute per-class AP and mAP.

    Excludes classes with zero positives from the mean.
    """
    if scores.shape != targets.shape:
        raise ValueError(
            f"scores/targets shape mismatch: {scores.shape} vs {targets.shape}"
        )
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D score matrix, got shape {scores.shape}")

    per_class_ap = np.full((scores.shape[1],), np.nan, dtype=np.float64)
    positive_mask = targets.sum(axis=0) > 0
    for class_idx in np.where(positive_mask)[0]:
        per_class_ap[class_idx] = _average_precision_binary(
            y_true=targets[:, class_idx],
            y_score=scores[:, class_idx],
        )

    valid = np.isfinite(per_class_ap)
    map_value = float(np.mean(per_class_ap[valid])) if np.any(valid) else float("nan")
    return {
        "map": map_value,
        "per_class_ap": per_class_ap,
        "num_samples": int(scores.shape[0]),
        "num_classes": int(scores.shape[1]),
        "num_positive_classes": int(np.sum(positive_mask)),
    }


def _mean_true_label_probability(scores: np.ndarray, targets: np.ndarray) -> float:
    positives = targets > 0.0
    if not np.any(positives):
        return float("nan")
    per_sample: list[float] = []
    for row_idx in range(scores.shape[0]):
        row_mask = positives[row_idx]
        if np.any(row_mask):
            per_sample.append(float(np.max(scores[row_idx, row_mask])))
    if not per_sample:
        return float("nan")
    return float(np.mean(np.asarray(per_sample, dtype=np.float64)))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _iter_attack_dirs(
    *,
    attack_root: Path,
    norm: str,
    epsilons: list[float] | None,
) -> list[tuple[float, Path]]:
    if epsilons:
        items: list[tuple[float, Path]] = []
        for epsilon in epsilons:
            token = format(epsilon, "g").replace(".", "p")
            path = attack_root / f"adversarial_{norm}_eps_{token}"
            if not path.exists():
                raise FileNotFoundError(f"Expected attack directory not found: {path}")
            items.append((float(epsilon), path))
        return sorted(items, key=lambda pair: pair[0])

    discovered: list[tuple[float, Path]] = []
    for path in sorted(attack_root.glob(f"adversarial_{norm}_eps_*")):
        if not path.is_dir():
            continue
        try:
            epsilon = _parse_epsilon_from_attack_id(path.name)
        except ValueError:
            continue
        discovered.append((epsilon, path))
    return sorted(discovered, key=lambda pair: pair[0])


_SWEEP_ATTACK_RE = re.compile(
    r"^adversarial_(?P<norm>l2|linf)_eps_(?P<eps>[0-9p\.\-]+)_steps_(?P<steps>\d+)_m_"
    r"(?P<mult>[0-9p\.\-]+)$"
)


def parse_sweep_attack_id(attack_id: str) -> dict[str, float | int | str]:
    match = _SWEEP_ATTACK_RE.match(attack_id)
    if match is None:
        raise ValueError(f"Unrecognized sweep attack id: {attack_id}")
    epsilon = float(match.group("eps").replace("p", "."))
    num_steps = int(match.group("steps"))
    multiplier = float(match.group("mult").replace("p", "."))
    return {
        "attack_id": attack_id,
        "norm": match.group("norm"),
        "epsilon": epsilon,
        "num_steps": num_steps,
        "step_size_multiplier": multiplier,
    }


def _iter_sweep_dirs(
    *,
    sweep_root: Path,
    norm: str,
    epsilons: list[float] | None,
) -> list[Path]:
    wanted = set(float(eps) for eps in epsilons) if epsilons else None
    discovered: list[tuple[float, Path]] = []
    for path in sorted(sweep_root.glob(f"adversarial_{norm}_eps_*_steps_*_m_*")):
        if not path.is_dir():
            continue
        try:
            meta = parse_sweep_attack_id(path.name)
        except ValueError:
            continue
        epsilon = float(meta["epsilon"])
        if wanted is not None and epsilon not in wanted:
            continue
        discovered.append((epsilon, path))
    return [
        path for _, path in sorted(discovered, key=lambda pair: (pair[0], pair[1].name))
    ]


def _compute_monotonicity_violations(
    points: list[tuple[float, float]],
    tolerance: float = 1e-12,
) -> list[dict[str, float]]:
    if not points:
        return []
    points = sorted(points, key=lambda pair: pair[0])
    violations: list[dict[str, float]] = []
    for idx in range(1, len(points)):
        prev_eps, prev_val = points[idx - 1]
        eps, val = points[idx]
        if val > prev_val + tolerance:
            violations.append(
                {
                    "epsilon_prev": float(prev_eps),
                    "epsilon_curr": float(eps),
                    "map_prev": float(prev_val),
                    "map_curr": float(val),
                    "increase": float(val - prev_val),
                }
            )
    return violations


def _attack_strength_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    """Lower adversarial mAP is stronger; tie-break on higher adversarial loss."""
    mean_adv_loss = float(row["mean_adversarial_loss"])
    loss_tiebreak = -mean_adv_loss if not math.isnan(mean_adv_loss) else float("inf")
    return float(row["adversarial_map"]), loss_tiebreak


def select_best_by_epsilon(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_epsilon: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_epsilon[float(row["epsilon"])].append(row)
    best_rows: list[dict[str, Any]] = []
    for epsilon in sorted(by_epsilon.keys()):
        candidates = by_epsilon[epsilon]
        best = min(candidates, key=_attack_strength_sort_key)
        best_rows.append(best)
    return best_rows


def _best_row_per_epsilon_and_m(
    rows: list[dict[str, Any]],
) -> dict[tuple[float, float], dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (float(row["epsilon"]), float(row["step_size_multiplier"]))
        grouped[key].append(row)
    return {
        key: min(candidates, key=_attack_strength_sort_key)
        for key, candidates in grouped.items()
    }


def select_best_m_by_ranked_vote(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Pick one global step-size multiplier ``m`` via Borda-style ranked voting.

    For each epsilon:
      1. Collapse num_steps by keeping the strongest row per ``m``.
      2. Rank all ``m`` values (lower adversarial mAP is better).
      3. Award Borda points: rank 1 gets ``n-1`` points, rank ``n`` gets 0.

    The ``m`` with the highest total points across epsilons wins. Ties break on
    lower mean adversarial mAP (averaged over epsilons, using each epsilon's
    best num_steps for that ``m``).
    """
    best_per_eps_m = _best_row_per_epsilon_and_m(rows)
    by_epsilon: dict[float, dict[float, dict[str, Any]]] = defaultdict(dict)
    for (epsilon, multiplier), row in best_per_eps_m.items():
        by_epsilon[epsilon][multiplier] = row

    vote_totals: dict[float, float] = defaultdict(float)
    per_epsilon_rankings: list[dict[str, Any]] = []
    for epsilon in sorted(by_epsilon.keys()):
        multiplier_to_row = by_epsilon[epsilon]
        ranked_multipliers = sorted(
            multiplier_to_row.keys(),
            key=lambda multiplier: _attack_strength_sort_key(
                multiplier_to_row[multiplier]
            ),
        )
        num_candidates = len(ranked_multipliers)
        epsilon_detail: dict[str, Any] = {
            "epsilon": float(epsilon),
            "ranked_multipliers": [float(m) for m in ranked_multipliers],
            "multiplier_details": {},
        }
        for rank_idx, multiplier in enumerate(ranked_multipliers):
            points = float(num_candidates - 1 - rank_idx)
            vote_totals[multiplier] += points
            row = multiplier_to_row[multiplier]
            epsilon_detail["multiplier_details"][str(multiplier)] = {
                "rank": int(rank_idx + 1),
                "points": points,
                "adversarial_map": float(row["adversarial_map"]),
                "best_num_steps": int(row["num_steps"]),
                "alpha": float(row["alpha"]),
            }
        per_epsilon_rankings.append(epsilon_detail)

    def _mean_adversarial_map_for_multiplier(multiplier: float) -> float:
        maps = [
            float(best_per_eps_m[(epsilon, multiplier)]["adversarial_map"])
            for epsilon in by_epsilon
            if multiplier in by_epsilon[epsilon]
        ]
        return float(np.mean(np.asarray(maps, dtype=np.float64)))

    best_multiplier = max(
        vote_totals.keys(),
        key=lambda multiplier: (
            vote_totals[multiplier],
            -_mean_adversarial_map_for_multiplier(multiplier),
        ),
    )
    vote_summary = [
        {
            "step_size_multiplier": float(multiplier),
            "total_points": float(vote_totals[multiplier]),
            "mean_adversarial_map": _mean_adversarial_map_for_multiplier(multiplier),
        }
        for multiplier in sorted(vote_totals.keys())
    ]
    vote_summary.sort(
        key=lambda item: (
            -float(item["total_points"]),
            float(item["mean_adversarial_map"]),
        )
    )
    return {
        "method": "borda_ranked_vote_per_epsilon",
        "best_step_size_multiplier": float(best_multiplier),
        "vote_totals": vote_summary,
        "per_epsilon_rankings": per_epsilon_rankings,
    }


def select_best_by_epsilon_with_fixed_m(
    rows: list[dict[str, Any]],
    step_size_multiplier: float,
) -> list[dict[str, Any]]:
    """For a fixed ``m``, pick the strongest num_steps per epsilon."""
    filtered = [
        row
        for row in rows
        if float(row["step_size_multiplier"]) == float(step_size_multiplier)
    ]
    by_epsilon: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        by_epsilon[float(row["epsilon"])].append(row)
    return [
        min(candidates, key=_attack_strength_sort_key)
        for _, candidates in sorted(by_epsilon.items())
    ]


def _plot_sweep_results(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "matplotlib is required to generate diagnostic sweep plots."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    by_config: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[(int(row["num_steps"]), float(row["step_size_multiplier"]))].append(
            row
        )
    for items in by_config.values():
        items.sort(key=lambda item: float(item["epsilon"]))

    plot_paths: list[Path] = []
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for (steps, mult), items in sorted(by_config.items()):
        eps = [float(item["epsilon"]) for item in items]
        adv_map = [float(item["adversarial_map"]) for item in items]
        ax1.plot(eps, adv_map, "o-", label=f"steps={steps}, m={mult:g}")
    ax1.set_xlabel("L2 epsilon")
    ax1.set_ylabel("Adversarial mAP")
    ax1.set_title("Adversarial mAP vs epsilon by attack config")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    map_path = output_dir / "adversarial_map_vs_epsilon_by_config.png"
    fig1.savefig(map_path, dpi=160)
    plt.close(fig1)
    plot_paths.append(map_path)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for (steps, mult), items in sorted(by_config.items()):
        eps = [float(item["epsilon"]) for item in items]
        adv_loss = [float(item["mean_adversarial_loss"]) for item in items]
        ax2.plot(eps, adv_loss, "o-", label=f"steps={steps}, m={mult:g}")
    ax2.set_xlabel("L2 epsilon")
    ax2.set_ylabel("Mean adversarial BCE loss")
    ax2.set_title("Adversarial loss vs epsilon by attack config")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    loss_path = output_dir / "adversarial_loss_vs_epsilon_by_config.png"
    fig2.savefig(loss_path, dpi=160)
    plt.close(fig2)
    plot_paths.append(loss_path)

    return plot_paths


def _plot_best_m_curve(
    *,
    best_rows: list[dict[str, Any]],
    step_size_multiplier: float,
    output_dir: Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "matplotlib is required to generate diagnostic sweep plots."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(best_rows, key=lambda row: float(row["epsilon"]))
    eps = [float(row["epsilon"]) for row in ordered]
    adv_map = [float(row["adversarial_map"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        eps, adv_map, "o-", label=f"m={step_size_multiplier:g} (ranked-vote winner)"
    )
    ax.set_xlabel("L2 epsilon")
    ax.set_ylabel("Adversarial mAP")
    ax.set_title("Best adversarial mAP vs epsilon (fixed m, best num_steps per ε)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "best_adversarial_map_vs_epsilon.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _run_inference(
    *,
    model: torch.nn.Module,
    waveform: torch.Tensor,
    sample_rate: int,
    device: str,
) -> np.ndarray:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    logits = cast(Any, model).get_classifier_logits(waveform.to(device), sr=sample_rate)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return _sigmoid_logits(logits[0])


def run_sweep_aggregation(
    *,
    sweep_root: Path,
    norm: str,
    epsilons: list[float] | None,
) -> None:
    sweep_dirs = _iter_sweep_dirs(sweep_root=sweep_root, norm=norm, epsilons=epsilons)
    if not sweep_dirs:
        raise SystemExit(f"No sweep directories found under {sweep_root}")

    result_rows: list[dict[str, Any]] = []
    for sweep_dir in sweep_dirs:
        parsed = parse_sweep_attack_id(sweep_dir.name)
        layer_dir = sweep_dir / "logits"
        scores_path = layer_dir / "scores.npz"
        summary_path = layer_dir / "summary.csv"
        if not scores_path.exists():
            print(f"[skip] missing scores: {scores_path}")
            continue
        scores_npz = np.load(scores_path)
        original_scores = np.asarray(scores_npz["original_scores"], dtype=np.float32)
        adversarial_scores = np.asarray(
            scores_npz["adversarial_scores"], dtype=np.float32
        )
        targets = np.asarray(scores_npz["targets"], dtype=np.float32)
        if (
            original_scores.size == 0
            or adversarial_scores.size == 0
            or targets.size == 0
        ):
            print(f"[skip] empty arrays in {scores_path}")
            continue
        clean_metrics = compute_map_from_scores(original_scores, targets)
        adversarial_metrics = compute_map_from_scores(adversarial_scores, targets)

        mean_adv_loss = float("nan")
        mean_clean_loss = float("nan")
        mean_delta_l2 = float("nan")
        median_delta_l2 = float("nan")
        mean_delta_linf = float("nan")
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                summary_rows = list(reader)
            if summary_rows:
                adv_loss_values = [
                    float(row["classification_loss"])
                    for row in summary_rows
                    if row.get("classification_loss", "") not in {"", "nan", "None"}
                ]
                clean_loss_values = [
                    float(row["clean_classification_loss"])
                    for row in summary_rows
                    if row.get("clean_classification_loss", "")
                    not in {"", "nan", "None"}
                ]
                delta_l2_values = [
                    float(row["delta_l2"])
                    for row in summary_rows
                    if row.get("delta_l2", "") not in {"", "nan", "None"}
                ]
                delta_linf_values = [
                    float(row["delta_linf"])
                    for row in summary_rows
                    if row.get("delta_linf", "") not in {"", "nan", "None"}
                ]
                if adv_loss_values:
                    mean_adv_loss = float(
                        np.mean(np.asarray(adv_loss_values, dtype=np.float64))
                    )
                if clean_loss_values:
                    mean_clean_loss = float(
                        np.mean(np.asarray(clean_loss_values, dtype=np.float64))
                    )
                if delta_l2_values:
                    l2_arr = np.asarray(delta_l2_values, dtype=np.float64)
                    mean_delta_l2 = float(np.mean(l2_arr))
                    median_delta_l2 = float(np.median(l2_arr))
                if delta_linf_values:
                    mean_delta_linf = float(
                        np.mean(np.asarray(delta_linf_values, dtype=np.float64))
                    )

        alpha = float(
            np.asarray(
                scores_npz.get("alpha", np.asarray(float("nan"), dtype=np.float32))
            ).reshape(-1)[0]
        )
        row = {
            "attack_id": sweep_dir.name,
            "epsilon": float(parsed["epsilon"]),
            "num_steps": int(parsed["num_steps"]),
            "step_size_multiplier": float(parsed["step_size_multiplier"]),
            "alpha": alpha,
            "clean_map": float(clean_metrics["map"]),
            "adversarial_map": float(adversarial_metrics["map"]),
            "mean_adversarial_loss": mean_adv_loss,
            "mean_clean_loss": mean_clean_loss,
            "mean_delta_l2": mean_delta_l2,
            "median_delta_l2": median_delta_l2,
            "mean_delta_linf": mean_delta_linf,
            "num_samples": int(clean_metrics["num_samples"]),
            "num_classes": int(clean_metrics["num_classes"]),
            "num_positive_classes": int(clean_metrics["num_positive_classes"]),
        }
        result_rows.append(row)
        print(
            f"[ok] {sweep_dir.name}: adv mAP={row['adversarial_map']:.4f}, "
            f"clean mAP={row['clean_map']:.4f}, mean adv loss={row['mean_adversarial_loss']:.4f}"
        )

    if not result_rows:
        raise SystemExit("No valid sweep results found.")

    result_rows.sort(
        key=lambda row: (
            float(row["epsilon"]),
            int(row["num_steps"]),
            float(row["step_size_multiplier"]),
        )
    )
    sweep_results_path = sweep_root / "sweep_results.csv"
    _write_csv(
        sweep_results_path,
        result_rows,
        fieldnames=[
            "attack_id",
            "epsilon",
            "num_steps",
            "step_size_multiplier",
            "alpha",
            "clean_map",
            "adversarial_map",
            "mean_adversarial_loss",
            "mean_clean_loss",
            "mean_delta_l2",
            "median_delta_l2",
            "mean_delta_linf",
            "num_samples",
            "num_classes",
            "num_positive_classes",
        ],
    )

    best_rows_unconstrained = select_best_by_epsilon(result_rows)
    unconstrained_path = sweep_root / "best_by_epsilon_unconstrained.csv"
    _write_csv(
        unconstrained_path,
        best_rows_unconstrained,
        fieldnames=[
            "attack_id",
            "epsilon",
            "num_steps",
            "step_size_multiplier",
            "alpha",
            "clean_map",
            "adversarial_map",
            "mean_adversarial_loss",
            "mean_clean_loss",
            "mean_delta_l2",
            "median_delta_l2",
            "mean_delta_linf",
            "num_samples",
            "num_classes",
            "num_positive_classes",
        ],
    )

    m_vote = select_best_m_by_ranked_vote(result_rows)
    best_multiplier = float(m_vote["best_step_size_multiplier"])
    best_rows = select_best_by_epsilon_with_fixed_m(result_rows, best_multiplier)
    best_path = sweep_root / "best_by_epsilon.csv"
    _write_csv(
        best_path,
        best_rows,
        fieldnames=[
            "attack_id",
            "epsilon",
            "num_steps",
            "step_size_multiplier",
            "alpha",
            "clean_map",
            "adversarial_map",
            "mean_adversarial_loss",
            "mean_clean_loss",
            "mean_delta_l2",
            "median_delta_l2",
            "mean_delta_linf",
            "num_samples",
            "num_classes",
            "num_positive_classes",
        ],
    )

    m_vote_path = sweep_root / "best_m_ranked_vote.json"
    with m_vote_path.open("w", encoding="utf-8") as handle:
        json.dump(m_vote, handle, indent=2, sort_keys=True)
    m_votes_csv_path = sweep_root / "best_m_votes.csv"
    _write_csv(
        m_votes_csv_path,
        list(m_vote["vote_totals"]),
        fieldnames=[
            "step_size_multiplier",
            "total_points",
            "mean_adversarial_map",
        ],
    )

    violations = _compute_monotonicity_violations(
        [(float(row["epsilon"]), float(row["adversarial_map"])) for row in best_rows]
    )
    unconstrained_violations = _compute_monotonicity_violations(
        [
            (float(row["epsilon"]), float(row["adversarial_map"]))
            for row in best_rows_unconstrained
        ]
    )
    monotonicity = {
        "selection_method": "fixed_m_ranked_vote_best_num_steps_per_epsilon",
        "best_step_size_multiplier": best_multiplier,
        "is_non_increasing": len(violations) == 0,
        "num_points": len(best_rows),
        "violations": violations,
        "unconstrained_per_epsilon_best": {
            "is_non_increasing": len(unconstrained_violations) == 0,
            "violations": unconstrained_violations,
        },
    }
    monotonicity_path = sweep_root / "monotonicity_report.json"
    with monotonicity_path.open("w", encoding="utf-8") as handle:
        json.dump(monotonicity, handle, indent=2, sort_keys=True)

    plots_dir = sweep_root / "plots"
    plot_paths = _plot_sweep_results(result_rows, plots_dir)
    plot_paths.append(
        _plot_best_m_curve(
            best_rows=best_rows,
            step_size_multiplier=best_multiplier,
            output_dir=plots_dir,
        )
    )
    print(f"[done] wrote sweep results: {sweep_results_path}")
    print(f"[done] wrote unconstrained best-by-epsilon: {unconstrained_path}")
    print(
        f"[done] selected global m={best_multiplier:g} via ranked vote; "
        f"wrote best-by-epsilon: {best_path}"
    )
    print(f"[done] wrote ranked-vote breakdown: {m_vote_path}")
    print(f"[done] wrote ranked-vote summary: {m_votes_csv_path}")
    print(f"[done] wrote monotonicity report: {monotonicity_path}")
    for path in plot_paths:
        print(f"[done] wrote plot: {path}")


def main() -> None:
    from src.models.audio import get_audio_model

    parser = ArgumentParser(
        description="Compute mAP from saved audio adversarial tensors."
    )
    parser.add_argument("--audio_model_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--tokenizer_checkpoint_path", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("--attack_root", type=str, default=None)
    parser.add_argument(
        "--sweep_root",
        type=str,
        default=None,
        help=(
            "Root directory for compact sweep outputs. Defaults to "
            "experiments/audio/<model>/adversarial_sweeps."
        ),
    )
    parser.add_argument("--norm", type=str, choices=["l2", "linf"], default="l2")
    parser.add_argument(
        "--epsilons",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of epsilons. If omitted, discover all saved attack dirs.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="Save compact scores/targets arrays as NPZ in each epsilon dir.",
    )
    parser.add_argument(
        "--write_aggregate_summary",
        action="store_true",
        help=(
            "Write attack_root/map_summary_all_eps.csv. "
            "Disable for Slurm array jobs (one epsilon per task) to avoid races."
        ),
    )
    parser.add_argument(
        "--aggregate_sweeps",
        action="store_true",
        help=(
            "Aggregate compact sweep outputs from scores.npz/summary.csv and write "
            "sweep diagnostics (CSV, JSON, plots)."
        ),
    )
    args = parser.parse_args()

    workspace_root = Path.cwd()
    sweep_root = (
        Path(args.sweep_root)
        if args.sweep_root is not None
        else Path(args.exp_dir) / "audio" / args.audio_model_name / "adversarial_sweeps"
    )
    if args.aggregate_sweeps:
        run_sweep_aggregation(
            sweep_root=sweep_root,
            norm=args.norm,
            epsilons=args.epsilons,
        )
        return

    attack_root = (
        Path(args.attack_root)
        if args.attack_root is not None
        else Path(args.exp_dir) / "audio" / args.audio_model_name / "adversarial"
    )
    layer_name = "logits"
    attack_items = _iter_attack_dirs(
        attack_root=attack_root,
        norm=args.norm,
        epsilons=args.epsilons,
    )
    if not attack_items:
        raise SystemExit(f"No attack directories found under {attack_root}")

    model = get_audio_model(
        args.audio_model_name,
        checkpoint_path=args.checkpoint_path,
        tokenizer_checkpoint_path=args.tokenizer_checkpoint_path,
        device=args.device,
        freeze=True,
    )
    model_label_mids = getattr(model, "classifier_label_mids", None)

    aggregate_rows: list[dict[str, Any]] = []
    for epsilon, attack_dir in attack_items:
        layer_dir = attack_dir / layer_name
        metadata_path = layer_dir / "metadata.jsonl"
        if not metadata_path.exists():
            print(f"[skip] missing metadata: {metadata_path}")
            continue
        records = _load_metadata_records(metadata_path)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        if not records:
            print(f"[skip] no records: {metadata_path}")
            continue

        score_original: list[np.ndarray] = []
        score_adversarial: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        sample_indices: list[int] = []

        num_classes: int | None = None
        skipped = 0
        for record in records:
            sample_idx_value = record.get("sample_idx")
            if sample_idx_value is None:
                skipped += 1
                continue
            sample_idx = int(sample_idx_value)
            sample_rate = int(record.get("sample_rate", 16_000))
            label_indices = _extract_true_labels(
                record=record, model_label_mids=model_label_mids
            )
            if not label_indices:
                skipped += 1
                continue
            saved_paths = record.get("saved_paths", {})
            orig_path_value = saved_paths.get("original_pt")
            adv_path_value = saved_paths.get("adversarial_pt")
            if not orig_path_value or not adv_path_value:
                skipped += 1
                continue
            original_path = _resolve_saved_path(orig_path_value, workspace_root)
            adversarial_path = _resolve_saved_path(adv_path_value, workspace_root)
            if not original_path.exists() or not adversarial_path.exists():
                skipped += 1
                continue

            original = torch.load(original_path, map_location="cpu")
            adversarial = torch.load(adversarial_path, map_location="cpu")
            original_scores = _run_inference(
                model=model,
                waveform=original,
                sample_rate=sample_rate,
                device=args.device,
            )
            adversarial_scores = _run_inference(
                model=model,
                waveform=adversarial,
                sample_rate=sample_rate,
                device=args.device,
            )
            if num_classes is None:
                num_classes = int(original_scores.shape[0])
            if int(original_scores.shape[0]) != num_classes:
                raise RuntimeError(
                    f"Inconsistent class count in {attack_dir}: "
                    f"{original_scores.shape[0]} vs expected {num_classes}"
                )

            target = np.zeros((num_classes,), dtype=np.float32)
            for label_idx in label_indices:
                idx = int(label_idx)
                if 0 <= idx < num_classes:
                    target[idx] = 1.0

            if float(target.sum()) <= 0.0:
                skipped += 1
                continue

            sample_indices.append(sample_idx)
            score_original.append(original_scores)
            score_adversarial.append(adversarial_scores)
            targets.append(target)

        if not targets:
            print(f"[skip] no valid labeled samples in {attack_dir}")
            continue

        original_arr = np.stack(score_original, axis=0)
        adversarial_arr = np.stack(score_adversarial, axis=0)
        target_arr = np.stack(targets, axis=0)

        original_metrics = compute_map_from_scores(original_arr, target_arr)
        adversarial_metrics = compute_map_from_scores(adversarial_arr, target_arr)
        original_mean_true = _mean_true_label_probability(original_arr, target_arr)
        adversarial_mean_true = _mean_true_label_probability(
            adversarial_arr, target_arr
        )

        summary_rows = [
            {
                "attack_id": attack_dir.name,
                "epsilon": epsilon,
                "split": "original",
                "map": original_metrics["map"],
                "num_samples": original_metrics["num_samples"],
                "num_classes": original_metrics["num_classes"],
                "num_positive_classes": original_metrics["num_positive_classes"],
                "mean_true_label_probability": original_mean_true,
                "num_skipped_samples": skipped,
            },
            {
                "attack_id": attack_dir.name,
                "epsilon": epsilon,
                "split": "adversarial",
                "map": adversarial_metrics["map"],
                "num_samples": adversarial_metrics["num_samples"],
                "num_classes": adversarial_metrics["num_classes"],
                "num_positive_classes": adversarial_metrics["num_positive_classes"],
                "mean_true_label_probability": adversarial_mean_true,
                "num_skipped_samples": skipped,
            },
        ]
        summary_path = layer_dir / "map_summary.csv"
        _write_csv(
            summary_path,
            summary_rows,
            fieldnames=[
                "attack_id",
                "epsilon",
                "split",
                "map",
                "num_samples",
                "num_classes",
                "num_positive_classes",
                "mean_true_label_probability",
                "num_skipped_samples",
            ],
        )

        per_class_rows: list[dict[str, Any]] = []
        per_class_original = original_metrics["per_class_ap"]
        per_class_adversarial = adversarial_metrics["per_class_ap"]
        positives_per_class = target_arr.sum(axis=0).astype(int)
        for class_idx in range(target_arr.shape[1]):
            if positives_per_class[class_idx] <= 0:
                continue
            per_class_rows.append(
                {
                    "class_idx": int(class_idx),
                    "num_positives": int(positives_per_class[class_idx]),
                    "ap_original": float(per_class_original[class_idx]),
                    "ap_adversarial": float(per_class_adversarial[class_idx]),
                    "ap_delta": float(
                        per_class_adversarial[class_idx] - per_class_original[class_idx]
                    ),
                }
            )
        _write_csv(
            layer_dir / "map_per_class.csv",
            per_class_rows,
            fieldnames=[
                "class_idx",
                "num_positives",
                "ap_original",
                "ap_adversarial",
                "ap_delta",
            ],
        )

        if args.save_scores:
            np.savez_compressed(
                layer_dir / "map_scores_original.npz",
                scores=original_arr.astype(np.float32),
                targets=target_arr.astype(np.float32),
                sample_idx=np.asarray(sample_indices, dtype=np.int64),
            )
            np.savez_compressed(
                layer_dir / "map_scores_adversarial.npz",
                scores=adversarial_arr.astype(np.float32),
                targets=target_arr.astype(np.float32),
                sample_idx=np.asarray(sample_indices, dtype=np.int64),
            )

        aggregate_rows.extend(summary_rows)
        print(
            f"[ok] {attack_dir.name}: "
            f"original mAP={original_metrics['map']:.4f}, "
            f"adversarial mAP={adversarial_metrics['map']:.4f}, "
            f"samples={int(target_arr.shape[0])}, skipped={skipped}"
        )

    if aggregate_rows and args.write_aggregate_summary:
        aggregate_path = attack_root / "map_summary_all_eps.csv"
        _write_csv(
            aggregate_path,
            aggregate_rows,
            fieldnames=[
                "attack_id",
                "epsilon",
                "split",
                "map",
                "num_samples",
                "num_classes",
                "num_positive_classes",
                "mean_true_label_probability",
                "num_skipped_samples",
            ],
        )
        print(f"[done] wrote aggregate summary: {aggregate_path}")
    elif aggregate_rows:
        print("[done] per-epsilon map_summary.csv written (aggregate summary skipped).")
    else:
        print("[done] no summaries written.")


if __name__ == "__main__":
    main()
