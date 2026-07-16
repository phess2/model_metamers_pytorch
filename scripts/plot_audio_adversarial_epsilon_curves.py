"""Plot aggregate performance-vs-epsilon curves across audio models.

Reads per-epsilon sweep and mAP summaries plus per-sample attack summaries for
each model under a run root laid out as
``<run_root>/<model>/adversarial/adversarial_<norm>_eps_*/``. Produces available
multi-model classification and suppression figures plus a merged CSV under
``<run_root>/plots/``.
"""

from __future__ import annotations

import csv
import json
import math
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TRUTHY = {"true", "1"}


def parse_epsilon_from_attack_id(attack_id: str) -> float:
    # Only the leading numeric token after "_eps_" is the epsilon; ignore any
    # trailing suffix such as "_suppress".
    token = attack_id.split("_eps_")[-1].split("_")[0]
    return float(token.replace("p", "."))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fraction_true(rows: list[dict[str, str]], column: str) -> float:
    values = [row[column].strip().lower() for row in rows if row.get(column, "") != ""]
    if not values:
        return float("nan")
    return sum(1 for value in values if value in _TRUTHY) / len(values)


def _mean_float(rows: list[dict[str, str]], column: str) -> float:
    values = [float(row[column]) for row in rows if row.get(column, "") != ""]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def collect_model_curve(
    *,
    model_root: Path,
    norm: str,
) -> list[dict[str, Any]]:
    """Collect per-epsilon metrics for one model's adversarial attack dirs."""
    points: list[dict[str, Any]] = []
    for attack_dir in sorted(
        (model_root / "adversarial").glob(f"adversarial_{norm}_eps_*")
    ):
        if not attack_dir.is_dir():
            continue
        try:
            epsilon = parse_epsilon_from_attack_id(attack_dir.name)
        except ValueError:
            continue
        layer_dir = attack_dir / "logits"
        point: dict[str, Any] = {
            "attack_id": attack_dir.name,
            "attack_mode": "",
            "epsilon": epsilon,
            "clean_map": float("nan"),
            "adversarial_map": float("nan"),
            "clean_top1": float("nan"),
            "adversarial_top1": float("nan"),
            "clean_top5": float("nan"),
            "adversarial_top5": float("nan"),
            "mean_true_label_prob_source": float("nan"),
            "mean_true_label_prob_adversarial": float("nan"),
            "mean_frac_true_labels_suppressed": float("nan"),
            "mean_target_probability_source": float("nan"),
            "mean_target_probability_adversarial": float("nan"),
            "target_top1_hit_rate": float("nan"),
            "target_top5_hit_rate": float("nan"),
            "mean_budget_utilization": float("nan"),
        }

        sweep_summary_path = layer_dir / "sweep_summary.json"
        if sweep_summary_path.exists():
            with sweep_summary_path.open("r", encoding="utf-8") as handle:
                sweep_summary = json.load(handle)
            for key in (
                "attack_mode",
                "clean_map",
                "adversarial_map",
                "mean_true_label_prob_source",
                "mean_true_label_prob_adversarial",
                "mean_frac_true_labels_suppressed",
            ):
                if key in sweep_summary:
                    point[key] = sweep_summary[key]

        map_summary_path = layer_dir / "map_summary.csv"
        if map_summary_path.exists():
            for row in _read_csv_rows(map_summary_path):
                if row.get("split") == "original":
                    point["clean_map"] = float(row["map"])
                elif row.get("split") == "adversarial":
                    point["adversarial_map"] = float(row["map"])

        summary_path = layer_dir / "summary.csv"
        if summary_path.exists():
            rows = _read_csv_rows(summary_path)
            point["clean_top1"] = _fraction_true(rows, "source_is_correct")
            point["adversarial_top1"] = _fraction_true(rows, "is_correct")
            point["clean_top5"] = _fraction_true(rows, "source_top5_is_correct")
            point["adversarial_top5"] = _fraction_true(rows, "top5_is_correct")
            point["mean_target_probability_source"] = _mean_float(
                rows, "target_source_probability"
            )
            point["mean_target_probability_adversarial"] = _mean_float(
                rows, "target_adversarial_probability"
            )
            point["target_top1_hit_rate"] = _fraction_true(rows, "target_top1_hit")
            point["target_top5_hit_rate"] = _fraction_true(rows, "target_top5_hit")
            point["mean_budget_utilization"] = _mean_float(
                rows, "budget_utilization_post_lowpass"
            )

        points.append(point)
    return sorted(points, key=lambda item: float(item["epsilon"]))


def _plot_metric(
    *,
    curves: dict[str, list[dict[str, Any]]],
    metric: str,
    clean_metric: str | None,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, points in sorted(curves.items()):
        eps = [float(point["epsilon"]) for point in points]
        values = [float(point[metric]) for point in points]
        line = ax.plot(eps, values, "o-", label=model_name)[0]
        clean_values = (
            [float(point[clean_metric]) for point in points]
            if clean_metric is not None
            else []
        )
        if clean_values and math.isfinite(clean_values[0]):
            ax.axhline(
                clean_values[0],
                color=line.get_color(),
                linestyle=":",
                alpha=0.5,
            )
    ax.set_xlabel("L2 epsilon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def metric_available(
    curves: dict[str, list[dict[str, Any]]],
    metric: str,
) -> bool:
    """Return whether at least one collected point has a finite metric value."""
    return any(
        math.isfinite(float(point[metric]))
        for points in curves.values()
        for point in points
    )


def main() -> None:
    parser = ArgumentParser(description="Plot multi-model adversarial epsilon curves.")
    parser.add_argument(
        "--run_root",
        type=str,
        required=True,
        help="Run root containing <model>/adversarial/adversarial_<norm>_eps_*/ dirs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names to include (defaults to all subdirectories with attacks).",
    )
    parser.add_argument("--norm", type=str, choices=["l2", "linf"], default="l2")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to <run_root>/plots).",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if args.models:
        model_names = list(args.models)
    else:
        model_names = sorted(
            path.parent.name for path in run_root.glob("*/adversarial") if path.is_dir()
        )
    if not model_names:
        raise SystemExit(f"No model adversarial directories found under {run_root}")

    curves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model_name in model_names:
        points = collect_model_curve(model_root=run_root / model_name, norm=args.norm)
        if not points:
            print(f"[skip] no attack dirs for model {model_name}")
            continue
        curves[model_name] = points

    if not curves:
        raise SystemExit("No curves collected; run attacks and mAP evaluation first.")

    output_dir = (
        Path(args.output_dir) if args.output_dir is not None else run_root / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, Any]] = []
    for model_name, points in sorted(curves.items()):
        for point in points:
            combined_rows.append({"model_name": model_name, **point})
    combined_csv = output_dir / "performance_vs_epsilon.csv"
    fieldnames = [
        "model_name",
        "attack_id",
        "attack_mode",
        "epsilon",
        "clean_map",
        "adversarial_map",
        "clean_top1",
        "adversarial_top1",
        "clean_top5",
        "adversarial_top5",
        "mean_true_label_prob_source",
        "mean_true_label_prob_adversarial",
        "mean_frac_true_labels_suppressed",
        "mean_target_probability_source",
        "mean_target_probability_adversarial",
        "target_top1_hit_rate",
        "target_top5_hit_rate",
        "mean_budget_utilization",
    ]
    with combined_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"Wrote {combined_csv}")

    plot_specs = [
        (
            "adversarial_map",
            "clean_map",
            "mAP",
            "Adversarial mAP vs L2 epsilon (dotted: clean mAP)",
            "adversarial_map_vs_epsilon.png",
        ),
        (
            "adversarial_top1",
            "clean_top1",
            "Top-1 accuracy",
            "Adversarial Top-1 accuracy vs L2 epsilon (dotted: clean)",
            "adversarial_top1_vs_epsilon.png",
        ),
        (
            "adversarial_top5",
            "clean_top5",
            "Top-5 accuracy",
            "Adversarial Top-5 accuracy vs L2 epsilon (dotted: clean)",
            "adversarial_top5_vs_epsilon.png",
        ),
        (
            "mean_true_label_prob_adversarial",
            "mean_true_label_prob_source",
            "Mean true-label probability",
            "True-label probability vs L2 epsilon (dotted: source)",
            "true_label_probability_vs_epsilon.png",
        ),
        (
            "mean_frac_true_labels_suppressed",
            None,
            "Fraction suppressed",
            "Fraction of true labels suppressed vs L2 epsilon",
            "fraction_true_labels_suppressed_vs_epsilon.png",
        ),
        (
            "mean_target_probability_adversarial",
            "mean_target_probability_source",
            "Mean target-label probability",
            "Target-label probability vs L2 epsilon (dotted: source)",
            "target_probability_vs_epsilon.png",
        ),
        (
            "target_top1_hit_rate",
            None,
            "Target top-1 hit rate",
            "Target-label top-1 hit rate vs L2 epsilon",
            "target_top1_hit_rate_vs_epsilon.png",
        ),
        (
            "target_top5_hit_rate",
            None,
            "Target top-5 hit rate",
            "Target-label top-5 hit rate vs L2 epsilon",
            "target_top5_hit_rate_vs_epsilon.png",
        ),
        (
            "mean_budget_utilization",
            None,
            "Mean budget utilization",
            "Saved perturbation budget utilization vs L2 epsilon",
            "budget_utilization_vs_epsilon.png",
        ),
    ]
    for metric, clean_metric, ylabel, title, filename in plot_specs:
        if not metric_available(curves, metric):
            continue
        _plot_metric(
            curves=curves,
            metric=metric,
            clean_metric=clean_metric,
            ylabel=ylabel,
            title=title,
            output_path=output_dir / filename,
        )
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
