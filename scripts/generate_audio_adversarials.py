"""Run untargeted classification (BCE) audio adversarial evaluation."""

from __future__ import annotations

import io
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy.io import wavfile
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
        import csv

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
    parser.add_argument("--step_size", type=float, default=0.002)
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
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

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
    attack_id = f"adversarial_{args.norm}_eps_{epsilon_token}"
    exp_root = Path(args.exp_dir) / "audio" / args.audio_model_name / "adversarial" / attack_id
    layer_output_dir = exp_root / layer_name
    top_level_csv_path = (
        Path(args.exp_dir)
        / "audio"
        / args.audio_model_name
        / "adversarial"
        / f"adversarial_{args.norm}_eps_{epsilon_token}.csv"
    )

    print(
        f"Running untargeted {args.norm.upper()} BCE classification attack on "
        f"{len(sample_indices)} samples "
        f"(epsilon={args.epsilon}, steps={args.num_steps}, step_size={args.step_size})"
    )

    rows = []
    vision_style_rows = []
    source_summaries = []
    adversarial_summaries = []
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

        attack_config = AdversarialAttackConfig(
            norm=args.norm,
            epsilon=args.epsilon,
            step_size=args.step_size,
            num_steps=args.num_steps,
            num_random_starts=args.num_random_starts,
            clamp_range=(args.clamp_min, args.clamp_max),
            seed=args.seed + int(sample_idx),
        )
        attacker = AudioAdversarialAttacker(
            model=model,
            sample_rate=int(file_sr),
            config=attack_config,
            device=args.device,
        )
        source_logits = _extract_model_logits(model, source_waveform, sr=int(file_sr))
        adversarial, attack_metadata = attacker.attack_untargeted(
            source_waveform=source_waveform,
            true_label_indices=labels_for_scoring,
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
        if adv_logits is not None and source_logits is not None:
            target = torch.zeros_like(adv_logits)
            for label_idx in labels_for_scoring:
                target[..., int(label_idx)] = 1.0
            final_classification_loss = float(
                classification_loss(adv_logits.to(args.device), target.to(args.device)).item()
            )

        ytid_value = source_example.get("ytid", b"")
        if isinstance(ytid_value, bytes):
            ytid_value = ytid_value.decode("utf-8", errors="replace")
        class_label = _sanitize_label(str(ytid_value)) if ytid_value else "audioset"
        if not class_label:
            class_label = "audioset"

        source_logits_summary = (
            summarize_multilabel_logits(source_logits, true_labels=labels_for_scoring)
            if source_logits is not None
            else None
        )
        adv_logits_summary = (
            summarize_multilabel_logits(adv_logits, true_labels=labels_for_scoring)
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
            "source_info": {
                "tfrecord_example_index": int(sample_idx),
                "ytid": ytid_value,
                "labels": labels_list,
                "labels_for_scoring": labels_for_scoring,
            },
            "attack_config": {
                "norm": args.norm,
                "epsilon": args.epsilon,
                "step_size": args.step_size,
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

        save_audio_adversarial_results(
            original=source_waveform,
            adversarial=adversarial,
            metadata=metadata,
            output_dir=layer_output_dir,
            sample_idx=sample_idx,
            class_label=class_label,
            include_spectrograms=args.include_spectrograms,
        )

        true_label_idx = int(labels_list[0]) if labels_list else ""
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
                "true_class_label": str(true_label_idx) if true_label_idx != "" else "",
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
                "true_label_idx": int(labels_list[0]) if labels_list else "",
                "true_label_indices": _format_label_indices(labels_list),
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
                "layer_name": layer_name,
                "delta_l2": float(norms["l2"].mean().item()),
                "delta_linf": float(norms["linf"].mean().item()),
                "snr_db": metadata["snr_db"],
                "classification_loss": final_classification_loss,
                "initial_classification_loss": metadata["initial_classification_loss"],
                "lowpass_filter_applied": lowpass_metadata["lowpass_filter_applied"],
                "lowpass_filter_cutoff_hz": lowpass_metadata["lowpass_filter_cutoff_hz"],
                "lowpass_filter_effective_cutoff_hz": lowpass_metadata[
                    "lowpass_filter_effective_cutoff_hz"
                ],
                "lowpass_filter_reason": lowpass_metadata["lowpass_filter_reason"],
            }
        )

        loss_str = (
            f"{final_classification_loss:.6f}"
            if final_classification_loss is not None
            else "n/a"
        )
        print(
            f"sample={sample_idx} layer={layer_name} mode=untargeted "
            f"bce={loss_str} delta_l2={float(norms['l2'].mean().item()):.6f}"
        )

    summary_path = layer_output_dir / "summary.csv"
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
    print(f"Saved layer summary: {summary_path}")
    print(f"Saved vision-style adversarial CSV: {top_level_csv_path}")
    print(f"Done. Results in {exp_root}")


if __name__ == "__main__":
    main()
