"""Run untargeted classification (BCE) audio adversarial evaluation."""

from __future__ import annotations

import io
import json
import sys
import csv
from collections.abc import Sequence
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy.io import wavfile
try:
    from sklearn.metrics import average_precision_score as _sk_average_precision_score
except ModuleNotFoundError:  # pragma: no cover - exercised by fallback path tests
    _sk_average_precision_score = None
try:
    from tfrecord.reader import example_loader
except ModuleNotFoundError:
    example_loader = None


def parse_indices(spec: str) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def epsilon_to_filename_token(epsilon: float) -> str:
    return format(epsilon, "g").replace(".", "p")


def _format_float_token(value: float) -> str:
    return format(float(value), "g").replace(".", "p")


def resolve_attack_step_size(
    *,
    norm: str,
    epsilon: float,
    num_steps: int,
    step_size: float | None,
    step_size_multiplier: float,
) -> tuple[float, str]:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if step_size is not None:
        return float(step_size), "explicit_step_size"
    if norm == "l2":
        return float(step_size_multiplier * float(epsilon) / float(num_steps)), "epsilon_scaled"
    return 0.002, "legacy_default"


def _average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if _sk_average_precision_score is not None:
        return float(_sk_average_precision_score(y_true, y_score))

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order].astype(np.float64, copy=False)
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1.0 - y_true_sorted)
    precision = tp / np.maximum(tp + fp, 1e-12)
    total_positives = float(np.sum(y_true_sorted))
    if total_positives <= 0.0:
        raise ValueError("AP requested with zero positives.")
    return float(np.sum(precision * y_true_sorted) / total_positives)


def _compute_map(scores: np.ndarray, targets: np.ndarray) -> float:
    if scores.shape != targets.shape:
        raise ValueError(f"scores/targets mismatch: {scores.shape} vs {targets.shape}")
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D scores, got {scores.shape}")
    per_class_ap = np.full((scores.shape[1],), np.nan, dtype=np.float64)
    positive_mask = targets.sum(axis=0) > 0
    for class_idx in np.where(positive_mask)[0]:
        per_class_ap[class_idx] = _average_precision_binary(
            y_true=targets[:, class_idx], y_score=scores[:, class_idx]
        )
    valid = np.isfinite(per_class_ap)
    return float(np.mean(per_class_ap[valid])) if np.any(valid) else float("nan")


def _write_vision_style_adversarial_csv(
    *,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_idx",
        "dataset_idx",
        "true_label_idx",
        "predicted_label_idx",
        "true_class_label",
        "predicted_class_label",
        "is_correct",
        "true_class_softmax",
        "epsilon_l2",
    ]
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_audio_waveform(waveform_np: np.ndarray) -> torch.Tensor:
    if waveform_np.ndim == 1:
        waveform_np = waveform_np[:, None]

    if np.issubdtype(waveform_np.dtype, np.integer):
        waveform = torch.from_numpy(waveform_np.astype(np.float32))
        max_int = float(np.iinfo(waveform_np.dtype).max)
        waveform = waveform / max_int
    else:
        waveform = torch.from_numpy(waveform_np.astype(np.float32))

    return waveform.transpose(0, 1).contiguous()


def _sanitize_label(text: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in text)


def _slice_clip(
    waveform: torch.Tensor,
    sr: int,
    start_seconds: float,
    clip_seconds: float,
) -> torch.Tensor:
    start_sample = int(start_seconds * sr)
    clip_samples = int(clip_seconds * sr)
    end_sample = start_sample + clip_samples
    waveform_clip = waveform[:, start_sample:end_sample]
    if waveform_clip.shape[-1] == 0:
        raise ValueError("Empty clip after slicing. Adjust --start_seconds/--clip_seconds.")
    return waveform_clip


def _rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2) + eps)


def _snr_db(signal: torch.Tensor, noise: torch.Tensor, eps: float = 1e-12) -> float:
    signal_rms = _rms(signal, eps=eps)
    noise_rms = _rms(noise, eps=eps)
    return float((20.0 * torch.log10((signal_rms + eps) / (noise_rms + eps))).item())


def _load_selected_examples(
    tfrecord_path: Path,
    sample_indices: list[int],
) -> dict[int, dict]:
    if example_loader is None:
        raise ModuleNotFoundError(
            "Missing dependency 'tfrecord'. Install it to load AudioSet TFRecord examples."
        )
    max_requested_idx = max(sample_indices)
    tfrecord_examples = example_loader(
        str(tfrecord_path),
        index_path=None,
        description=None,
        compression_type="gzip",
    )
    selected_examples = {}
    for example_idx, example in enumerate(tfrecord_examples):
        if example_idx in sample_indices:
            selected_examples[example_idx] = example
        if example_idx >= max_requested_idx and len(selected_examples) == len(sample_indices):
            break
    return selected_examples


def _format_label_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    return "|".join(str(int(idx)) for idx in indices)


def _load_audioset_label_names(csv_path: str | None) -> dict[int, str]:
    if csv_path is None:
        return {}
    path = Path(csv_path)
    if not path.exists():
        return {}
    label_names: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index = row.get("index")
            display_name = row.get("display_name")
            if index is None or display_name is None:
                continue
            try:
                label_names[int(index)] = str(display_name)
            except ValueError:
                continue
    return label_names


def _resolve_single_label_targets(
    *,
    labels_list: Sequence[int],
    model_label_mids: Sequence[str] | None,
    remap_fn: Any,
) -> list[dict[str, int]]:
    targets: list[dict[str, int]] = []
    for audioset_label in labels_list:
        mapped = remap_fn(
            true_labels=[int(audioset_label)],
            model_label_mids=model_label_mids,
        )
        if not mapped:
            continue
        targets.append(
            {
                "target_audioset_label": int(audioset_label),
                "target_model_label": int(mapped[0]),
            }
        )
    return targets


def _extract_model_logits(model: torch.nn.Module, waveform: torch.Tensor, sr: int) -> torch.Tensor | None:
    try:
        logits = cast(Any, model).get_classifier_logits(waveform, sr=sr)
        return logits.detach().cpu()
    except NotImplementedError:
        return None


def _verify_model_supports_classification_attack(
    model: torch.nn.Module, waveform: torch.Tensor, sr: int
) -> None:
    try:
        cast(Any, model).get_classifier_logits(waveform, sr=sr)
    except NotImplementedError as exc:
        print(
            "ERROR: Model does not expose classifier logits. "
            "Classification BCE attacks require a fine-tuned head (e.g. beats_iter3_plus_as2m).",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def main() -> None:
    from src.analysis.audio_adversarial import (
        AdversarialAttackConfig,
        AudioAdversarialAttacker,
        classification_loss,
        perturbation_norms,
    )
    from src.analysis.audio_classification import (
        aggregate_multilabel_topk,
        decode_audioset_labels,
        remap_audioset_labels_to_model_indices,
        remap_model_indices_to_audioset_labels,
        summarize_multilabel_logits,
    )
    from src.analysis.audio_filtering import lowpass_filter_for_model
    from src.analysis.saving import save_adversarial_results_csv, save_audio_adversarial_results
    from src.models.audio import get_audio_model

    parser = ArgumentParser(
        description="Evaluate untargeted audio adversarial robustness (BCE on logits)"
    )
    parser.add_argument("--audio_model_name", type=str, default="audiomae_as2m_ft_as20k")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--tokenizer_checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="Ignored except 'logits'; classification attacks always target classifier logits.",
    )
    parser.add_argument("--list_layers", action="store_true")
    parser.add_argument("--indices", type=str, default="0-9")
    parser.add_argument("--norm", type=str, choices=["l2", "linf"], default="l2")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument(
        "--step_size",
        type=float,
        default=None,
        help=(
            "Explicit PGD step size alpha. If omitted for L2 attacks, uses "
            "alpha = step_size_multiplier * epsilon / num_steps."
        ),
    )
    parser.add_argument(
        "--step_size_multiplier",
        type=float,
        default=2.0,
        help="Multiplier m for epsilon-scaled L2 step size alpha = m * epsilon / num_steps.",
    )
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--num_random_starts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip_seconds", type=float, default=2.0)
    parser.add_argument("--start_seconds", type=float, default=0.0)
    parser.add_argument("--clamp_min", type=float, default=-1.0)
    parser.add_argument("--clamp_max", type=float, default=1.0)
    parser.add_argument("--include_spectrograms", action="store_true")
    parser.add_argument(
        "--audioset_root",
        type=str,
        default="/home/rphess/orcd/datasets/AudioSet",
    )
    parser.add_argument("--audioset_split_glob", type=str, default="audioset-2m-part*")
    parser.add_argument("--tfrecord_glob", type=str, default="*.tfrecords")
    parser.add_argument("--input_sample_rate_override", type=int, default=None)
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Optional output root override. When set, outputs are written under "
            "<output_root>/<audio_model_name>/..."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--single_label_targets",
        action="store_true",
        help=(
            "Attack one ground-truth label at a time (fan out one adversarial "
            "example per source label)."
        ),
    )
    parser.add_argument(
        "--audioset_label_map_csv",
        type=str,
        default="/home/rphess/orcd/datasets/AudioSet/class_labels_indices.csv",
        help=(
            "Optional AudioSet class_labels_indices.csv path for readable label names "
            "in metadata (falls back to label indices when unavailable)."
        ),
    )
    parser.add_argument(
        "--sweep_scores_only",
        action="store_true",
        help=(
            "Run attacks and save compact scores/metrics only (no per-sample "
            "audio/PT outputs). Intended for hyperparameter sweeps."
        ),
    )
    args = parser.parse_args()
    if args.num_steps <= 0:
        raise SystemExit("--num_steps must be positive.")

    effective_step_size, step_size_source = resolve_attack_step_size(
        norm=args.norm,
        epsilon=args.epsilon,
        num_steps=args.num_steps,
        step_size=args.step_size,
        step_size_multiplier=args.step_size_multiplier,
    )

    sample_indices = parse_indices(args.indices)
    if len(sample_indices) == 0:
        print("ERROR: --indices resolved to an empty set.", file=sys.stderr)
        sys.exit(1)

    model = get_audio_model(
        args.audio_model_name,
        checkpoint_path=args.checkpoint_path,
        tokenizer_checkpoint_path=args.tokenizer_checkpoint_path,
        device=args.device,
        freeze=True,
    )
    available_layers = list(getattr(model, "metamer_layers"))
    if args.list_layers:
        print("\nAvailable metamer layers (for metamers; attacks use logits only):")
        for layer in available_layers:
            print(f"  - {layer}")
        if hasattr(model, "get_classifier_logits"):
            print("  - logits (classification attack target)")
        sys.exit(0)

    if args.layers is not None:
        unsupported = [layer for layer in args.layers if layer != "logits"]
        if unsupported:
            print(
                "WARNING: Classification attacks only support layer 'logits'. "
                f"Ignoring: {unsupported}",
                file=sys.stderr,
            )

    layer_name = "logits"
    model_label_mids = getattr(model, "classifier_label_mids", None)
    audioset_root = Path(args.audioset_root)
    split_dirs = sorted(audioset_root.glob(args.audioset_split_glob))
    if len(split_dirs) == 0:
        print(
            f"ERROR: No split dirs matched {args.audioset_split_glob!r} under {audioset_root}",
            file=sys.stderr,
        )
        sys.exit(1)
    first_split_dir = split_dirs[0]
    tfrecord_paths = sorted(first_split_dir.glob(args.tfrecord_glob))
    if len(tfrecord_paths) == 0:
        print(
            f"ERROR: No TFRecords matched {args.tfrecord_glob!r} under {first_split_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    first_tfrecord = tfrecord_paths[0]

    selected_examples = _load_selected_examples(
        tfrecord_path=first_tfrecord,
        sample_indices=sample_indices,
    )
    missing = sorted(set(sample_indices) - set(selected_examples.keys()))
    if missing:
        print(f"ERROR: Missing TFRecord example indices: {missing}", file=sys.stderr)
        sys.exit(1)

    epsilon_token = epsilon_to_filename_token(args.epsilon)
    step_multiplier_token = _format_float_token(args.step_size_multiplier)
    attack_scope_suffix = "_single_label" if args.single_label_targets else ""
    output_base = (
        Path(args.output_root) / args.audio_model_name
        if args.output_root is not None
        else Path(args.exp_dir) / "audio" / args.audio_model_name
    )
    if args.sweep_scores_only:
        attack_id = (
            f"adversarial_{args.norm}_eps_{epsilon_token}_steps_{args.num_steps}_m_"
            f"{step_multiplier_token}{attack_scope_suffix}"
        )
        exp_root = output_base / "adversarial_sweeps" / attack_id
    else:
        attack_id = f"adversarial_{args.norm}_eps_{epsilon_token}{attack_scope_suffix}"
        exp_root = output_base / "adversarial" / attack_id
    layer_output_dir = exp_root / layer_name
    top_level_csv_path = output_base / "adversarial" / f"{attack_id}.csv"
    audioset_label_names = _load_audioset_label_names(args.audioset_label_map_csv)

    print(
        f"Running untargeted {args.norm.upper()} BCE classification attack on "
        f"{len(sample_indices)} samples "
        f"(epsilon={args.epsilon}, steps={args.num_steps}, alpha={effective_step_size}, "
        f"alpha_source={step_size_source}, m={args.step_size_multiplier}, "
        f"single_label_targets={args.single_label_targets}, "
        f"sweep_scores_only={args.sweep_scores_only})"
    )

    rows = []
    vision_style_rows = []
    source_summaries = []
    adversarial_summaries = []
    score_original: list[np.ndarray] = []
    score_adversarial: list[np.ndarray] = []
    score_targets: list[np.ndarray] = []
    sweep_metrics_rows: list[dict[str, object]] = []
    sweep_delta_l2_values: list[float] = []
    sweep_delta_linf_values: list[float] = []
    model_verified = False

    for sample_idx in sample_indices:
        source_example = selected_examples[sample_idx]
        file_sr, waveform_np = wavfile.read(io.BytesIO(source_example["audio"]))
        if args.input_sample_rate_override is not None:
            file_sr = int(args.input_sample_rate_override)
        waveform = _normalize_audio_waveform(waveform_np)
        waveform_mono = waveform.mean(dim=0, keepdim=True)
        source_clip = _slice_clip(
            waveform_mono,
            sr=file_sr,
            start_seconds=args.start_seconds,
            clip_seconds=args.clip_seconds,
        )
        source_waveform = source_clip.unsqueeze(0).to(args.device)

        if not model_verified:
            _verify_model_supports_classification_attack(
                model, source_waveform, sr=int(file_sr)
            )
            model_verified = True

        labels_list = decode_audioset_labels(source_example)
        if not labels_list:
            print(
                f"ERROR: sample {sample_idx} has no AudioSet labels; cannot run BCE attack.",
                file=sys.stderr,
            )
            sys.exit(1)
        labels_for_scoring = remap_audioset_labels_to_model_indices(
            true_labels=labels_list,
            model_label_mids=model_label_mids,
        )
        if not labels_for_scoring:
            print(
                f"ERROR: sample {sample_idx} has no remapped model labels for scoring.",
                file=sys.stderr,
            )
            sys.exit(1)

        source_logits = _extract_model_logits(model, source_waveform, sr=int(file_sr))
        ytid_value = source_example.get("ytid", b"")
        if isinstance(ytid_value, bytes):
            ytid_value = ytid_value.decode("utf-8", errors="replace")

        if args.single_label_targets:
            target_specs = _resolve_single_label_targets(
                labels_list=labels_list,
                model_label_mids=model_label_mids,
                remap_fn=remap_audioset_labels_to_model_indices,
            )
            if not target_specs:
                print(
                    f"ERROR: sample {sample_idx} has no single-label remapped targets.",
                    file=sys.stderr,
                )
                sys.exit(1)
            target_batches = [
                (
                    int(item["target_audioset_label"]),
                    [int(item["target_model_label"])],
                )
                for item in target_specs
            ]
        else:
            target_batches = [(int(labels_list[0]), [int(x) for x in labels_for_scoring])]

        base_class_label = _sanitize_label(str(ytid_value)) if ytid_value else "audioset"
        if not base_class_label:
            base_class_label = "audioset"
        all_source_label_names = [
            audioset_label_names.get(int(idx), f"label_{int(idx)}") for idx in labels_list
        ]

        for target_audioset_label, target_model_labels in target_batches:
            target_model_label = int(target_model_labels[0])
            target_label_name = audioset_label_names.get(
                target_audioset_label, f"label_{target_audioset_label}"
            )
            target_label_token = f"label{target_audioset_label:04d}"
            class_label = (
                f"{base_class_label}_{target_label_token}"
                if args.single_label_targets
                else base_class_label
            )

            attack_config = AdversarialAttackConfig(
                norm=args.norm,
                epsilon=args.epsilon,
                step_size=effective_step_size,
                num_steps=args.num_steps,
                num_random_starts=args.num_random_starts,
                clamp_range=(args.clamp_min, args.clamp_max),
                seed=args.seed + int(sample_idx) + (10_000 * target_model_label),
            )
            attacker = AudioAdversarialAttacker(
                model=model,
                sample_rate=int(file_sr),
                config=attack_config,
                device=args.device,
            )
            adversarial, attack_metadata = attacker.attack_untargeted(
                source_waveform=source_waveform,
                true_label_indices=target_model_labels,
            )

            adversarial, lowpass_metadata = lowpass_filter_for_model(
                adversarial,
                model_name=args.audio_model_name,
                sample_rate=int(file_sr),
            )
            delta = adversarial - source_waveform
            norms = perturbation_norms(delta)
            adv_logits = _extract_model_logits(model, adversarial, sr=int(file_sr))
            final_classification_loss = None
            clean_classification_loss = None
            if adv_logits is not None and source_logits is not None:
                target = torch.zeros_like(adv_logits)
                for label_idx in target_model_labels:
                    target[..., int(label_idx)] = 1.0
                clean_classification_loss = float(
                    classification_loss(source_logits.to(args.device), target.to(args.device)).item()
                )
                final_classification_loss = float(
                    classification_loss(adv_logits.to(args.device), target.to(args.device)).item()
                )
                clean_probs = torch.sigmoid(source_logits).detach().cpu().numpy().astype(np.float32)
                adv_probs = torch.sigmoid(adv_logits).detach().cpu().numpy().astype(np.float32)
                target_np = np.zeros_like(clean_probs, dtype=np.float32)
                for label_idx in target_model_labels:
                    target_np[..., int(label_idx)] = 1.0
                score_original.append(clean_probs.reshape(-1))
                score_adversarial.append(adv_probs.reshape(-1))
                score_targets.append(target_np.reshape(-1))

            source_logits_summary = (
                summarize_multilabel_logits(source_logits, true_labels=target_model_labels)
                if source_logits is not None
                else None
            )
            adv_logits_summary = (
                summarize_multilabel_logits(adv_logits, true_labels=target_model_labels)
                if adv_logits is not None
                else None
            )
            if source_logits_summary is not None:
                source_summaries.append(source_logits_summary)
            if adv_logits_summary is not None:
                adversarial_summaries.append(adv_logits_summary)
            mapped_predicted_label_idx = ""
            mapped_predicted_top5_label_indices: list[int] = []
            if adv_logits_summary is not None:
                mapped_predicted_label_idx = int(
                    remap_model_indices_to_audioset_labels(
                        predicted_indices=[int(adv_logits_summary["predicted_label_idx"])],
                        model_label_mids=model_label_mids,
                    )[0]
                )
                mapped_predicted_top5_label_indices = remap_model_indices_to_audioset_labels(
                    predicted_indices=adv_logits_summary["predicted_top5_label_indices"],
                    model_label_mids=model_label_mids,
                )

            metadata = {
                "model_name": args.audio_model_name,
                "layer_name": layer_name,
                "attack_mode": "untargeted",
                "attack_id": attack_id,
                "sample_rate": int(file_sr),
                "single_label_attack": bool(args.single_label_targets),
                "target_audioset_label": target_audioset_label,
                "target_model_label": target_model_label,
                "target_label_name": target_label_name,
                "all_source_labels": [int(x) for x in labels_list],
                "all_source_label_names": all_source_label_names,
                "source_info": {
                    "tfrecord_example_index": int(sample_idx),
                    "ytid": ytid_value,
                    "labels": labels_list,
                    "labels_for_scoring": target_model_labels,
                },
                "attack_config": {
                    "norm": args.norm,
                    "epsilon": args.epsilon,
                    "step_size": effective_step_size,
                    "step_size_input": args.step_size,
                    "step_size_effective": effective_step_size,
                    "step_size_source": step_size_source,
                    "step_size_multiplier": args.step_size_multiplier,
                    "num_steps": args.num_steps,
                    "num_random_starts": args.num_random_starts,
                    "loss": "bce_multihot",
                    "clamp_range": [args.clamp_min, args.clamp_max],
                },
                "classification_loss": final_classification_loss,
                "initial_classification_loss": float(
                    attack_metadata["initial_classification_loss"]
                ),
                "delta_l2": float(norms["l2"].mean().item()),
                "delta_linf": float(norms["linf"].mean().item()),
                "snr_db": _snr_db(signal=source_waveform, noise=delta),
                "attack_summary": {
                    "best_start_idx": int(attack_metadata["best_start_idx"]),
                    "best_final_classification_loss": float(
                        attack_metadata["best_final_classification_loss"]
                    ),
                    "initial_classification_loss": float(
                        attack_metadata["initial_classification_loss"]
                    ),
                    "best_delta_l2_mean": float(attack_metadata["best_delta_l2_mean"]),
                    "best_delta_linf_mean": float(attack_metadata["best_delta_linf_mean"]),
                    "num_random_starts": int(attack_metadata["num_random_starts"]),
                },
                **lowpass_metadata,
            }
            if source_logits_summary is not None:
                metadata["source_classification"] = source_logits_summary
            if adv_logits_summary is not None:
                metadata["adversarial_classification"] = adv_logits_summary

            if not args.sweep_scores_only:
                save_audio_adversarial_results(
                    original=source_waveform,
                    adversarial=adversarial,
                    metadata=metadata,
                    output_dir=layer_output_dir,
                    sample_idx=sample_idx,
                    class_label=class_label,
                    include_spectrograms=args.include_spectrograms,
                )

            true_label_idx = target_audioset_label
            predicted_label_idx = (
                int(mapped_predicted_label_idx)
                if adv_logits_summary is not None
                else ""
            )
            vision_style_rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "dataset_idx": int(sample_idx),
                    "true_label_idx": true_label_idx,
                    "predicted_label_idx": predicted_label_idx,
                    "true_class_label": str(true_label_idx),
                    "predicted_class_label": (
                        str(predicted_label_idx) if predicted_label_idx != "" else ""
                    ),
                    "is_correct": (
                        bool(adv_logits_summary["top1_hit"])
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "true_class_softmax": (
                        float(adv_logits_summary["true_label_max_softmax"])
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "epsilon_l2": float(args.epsilon) if args.norm == "l2" else "",
                }
            )

            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "dataset_idx": int(sample_idx),
                    "true_label_idx": true_label_idx,
                    "true_label_indices": _format_label_indices(labels_list),
                    "target_audioset_label": target_audioset_label,
                    "target_model_label": target_model_label,
                    "target_label_name": target_label_name,
                    "predicted_label_idx": (
                        int(mapped_predicted_label_idx)
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "predicted_top5_label_indices": (
                        _format_label_indices(mapped_predicted_top5_label_indices)
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "true_class_label": class_label,
                    "predicted_class_label": (
                        str(int(mapped_predicted_label_idx))
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "is_correct": (
                        bool(adv_logits_summary["top1_hit"])
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "top5_is_correct": (
                        bool(adv_logits_summary["top5_hit"])
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "true_class_softmax": (
                        float(adv_logits_summary["true_label_max_softmax"])
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "source_is_correct": (
                        bool(source_logits_summary["top1_hit"])
                        if source_logits_summary is not None
                        else ""
                    ),
                    "source_top5_is_correct": (
                        bool(source_logits_summary["top5_hit"])
                        if source_logits_summary is not None
                        else ""
                    ),
                    "attack_mode": "untargeted",
                    "single_label_attack": bool(args.single_label_targets),
                    "layer_name": layer_name,
                    "delta_l2": float(norms["l2"].mean().item()),
                    "delta_linf": float(norms["linf"].mean().item()),
                    "snr_db": metadata["snr_db"],
                    "classification_loss": final_classification_loss,
                    "clean_classification_loss": clean_classification_loss,
                    "initial_classification_loss": metadata["initial_classification_loss"],
                    "epsilon": float(args.epsilon),
                    "alpha": float(effective_step_size),
                    "num_steps": int(args.num_steps),
                    "step_size_multiplier": float(args.step_size_multiplier),
                    "lowpass_filter_applied": lowpass_metadata["lowpass_filter_applied"],
                    "lowpass_filter_cutoff_hz": lowpass_metadata["lowpass_filter_cutoff_hz"],
                    "lowpass_filter_effective_cutoff_hz": lowpass_metadata[
                        "lowpass_filter_effective_cutoff_hz"
                    ],
                    "lowpass_filter_reason": lowpass_metadata["lowpass_filter_reason"],
                }
            )
            sweep_metrics_rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "target_audioset_label": target_audioset_label,
                    "target_model_label": target_model_label,
                    "target_label_name": target_label_name,
                    "epsilon": float(args.epsilon),
                    "alpha": float(effective_step_size),
                    "num_steps": int(args.num_steps),
                    "step_size_multiplier": float(args.step_size_multiplier),
                    "clean_classification_loss": clean_classification_loss,
                    "classification_loss": final_classification_loss,
                    "delta_l2": float(norms["l2"].mean().item()),
                    "delta_linf": float(norms["linf"].mean().item()),
                }
            )
            sweep_delta_l2_values.append(float(norms["l2"].mean().item()))
            sweep_delta_linf_values.append(float(norms["linf"].mean().item()))

            loss_str = (
                f"{final_classification_loss:.6f}"
                if final_classification_loss is not None
                else "n/a"
            )
            print(
                f"sample={sample_idx} layer={layer_name} mode=untargeted "
                f"target={target_audioset_label}:{target_label_name} "
                f"bce={loss_str} delta_l2={float(norms['l2'].mean().item()):.6f}"
            )

    summary_path = layer_output_dir / "summary.csv"
    if args.sweep_scores_only:
        layer_output_dir.mkdir(parents=True, exist_ok=True)
        save_adversarial_results_csv(rows=sweep_metrics_rows, output_path=summary_path)
    else:
        save_adversarial_results_csv(rows=rows, output_path=summary_path)
        _write_vision_style_adversarial_csv(rows=vision_style_rows, output_path=top_level_csv_path)
    source_metrics = aggregate_multilabel_topk(source_summaries)
    adversarial_metrics = aggregate_multilabel_topk(adversarial_summaries)
    if adversarial_summaries:
        print(f"Layer={layer_name} source Top-1 accuracy: {source_metrics['top1_acc']:.4f}")
        print(f"Layer={layer_name} source Top-5 accuracy: {source_metrics['top5_acc']:.4f}")
        print(
            f"Layer={layer_name} adversarial Top-1 accuracy: {adversarial_metrics['top1_acc']:.4f}"
        )
        print(
            f"Layer={layer_name} adversarial Top-5 accuracy: {adversarial_metrics['top5_acc']:.4f}"
        )
    else:
        print(f"Layer={layer_name} classification metrics unavailable (no classifier head).")
    if score_targets:
        original_arr = np.stack(score_original, axis=0).astype(np.float32)
        adversarial_arr = np.stack(score_adversarial, axis=0).astype(np.float32)
        target_arr = np.stack(score_targets, axis=0).astype(np.float32)
        clean_map = _compute_map(original_arr, target_arr)
        adversarial_map = _compute_map(adversarial_arr, target_arr)
        np.savez_compressed(
            layer_output_dir / "scores.npz",
            original_scores=original_arr,
            adversarial_scores=adversarial_arr,
            targets=target_arr,
            epsilon=np.asarray(args.epsilon, dtype=np.float32),
            alpha=np.asarray(effective_step_size, dtype=np.float32),
            num_steps=np.asarray(args.num_steps, dtype=np.int32),
            step_size_multiplier=np.asarray(args.step_size_multiplier, dtype=np.float32),
        )
        sweep_summary = {
            "attack_id": attack_id,
            "epsilon": float(args.epsilon),
            "alpha": float(effective_step_size),
            "alpha_source": step_size_source,
            "num_steps": int(args.num_steps),
            "step_size_multiplier": float(args.step_size_multiplier),
            "num_samples": int(target_arr.shape[0]),
            "clean_map": float(clean_map),
            "adversarial_map": float(adversarial_map),
            "mean_adversarial_loss": float(
                np.nanmean(
                    np.asarray(
                        [row.get("classification_loss", np.nan) for row in sweep_metrics_rows],
                        dtype=np.float64,
                    )
                )
            ),
            "mean_clean_loss": float(
                np.nanmean(
                    np.asarray(
                        [row.get("clean_classification_loss", np.nan) for row in sweep_metrics_rows],
                        dtype=np.float64,
                    )
                )
            ),
            "mean_delta_linf": float(
                np.mean(np.asarray(sweep_delta_linf_values, dtype=np.float64))
            ),
            "mean_delta_l2": float(
                np.mean(np.asarray(sweep_delta_l2_values, dtype=np.float64))
            ),
        }
        with (layer_output_dir / "sweep_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(sweep_summary, handle, indent=2, sort_keys=True)
        print(
            f"Sweep metrics: clean mAP={clean_map:.4f}, adversarial mAP={adversarial_map:.4f}, "
            f"saved scores={layer_output_dir / 'scores.npz'}"
        )
    print(f"Saved layer summary: {summary_path}")
    if not args.sweep_scores_only:
        print(f"Saved vision-style adversarial CSV: {top_level_csv_path}")
    print(f"Done. Results in {exp_root}")


if __name__ == "__main__":
    main()
