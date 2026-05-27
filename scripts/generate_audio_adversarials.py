"""Generate representation-space audio adversarial examples."""

from __future__ import annotations

import io
import json
import math
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch
from scipy.io import wavfile
from tfrecord.reader import example_loader

from src.analysis.audio_adversarial import (
    AudioRepresentationAttacker,
    RepresentationAttackConfig,
    perturbation_norms,
    representation_distance,
)
from src.analysis.audio_classification import (
    aggregate_multilabel_topk,
    decode_audioset_labels,
    summarize_multilabel_logits,
)
from src.analysis.audio_filtering import lowpass_filter_for_model
from src.analysis.saving import save_adversarial_results_csv, save_audio_adversarial_results
from src.models.audio import get_audio_model


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
) -> Dict[int, dict]:
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


def _resolve_target_indices(
    source_indices: list[int],
    target_indices: list[int] | None,
) -> Dict[int, int]:
    if target_indices is None:
        if len(source_indices) < 2:
            raise ValueError(
                "Targeted attack requires at least two source examples when --target_indices is omitted."
            )
        return {
            source_idx: source_indices[(idx + 1) % len(source_indices)]
            for idx, source_idx in enumerate(source_indices)
        }

    if len(target_indices) == 1:
        return {source_idx: target_indices[0] for source_idx in source_indices}
    if len(target_indices) != len(source_indices):
        raise ValueError(
            "When --target_indices has multiple values, it must match --indices length."
        )
    return {source_idx: target_idx for source_idx, target_idx in zip(source_indices, target_indices)}


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


def main() -> None:
    parser = ArgumentParser(description="Generate representation-space audio adversarial examples")
    parser.add_argument("--audio_model_name", type=str, default="audiomae_as2m_ft_as20k")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--tokenizer_checkpoint_path", type=str, default=None)
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--list_layers", action="store_true")
    parser.add_argument("--attack_mode", type=str, choices=["untargeted", "targeted"], default="untargeted")
    parser.add_argument("--indices", type=str, default="0-9")
    parser.add_argument("--target_indices", type=str, default=None)
    parser.add_argument("--loss_type", type=str, choices=["normalized_l2", "l2", "squared_l2", "cosine"], default="normalized_l2")
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
    parser.add_argument("--classification_only", action="store_true")
    args = parser.parse_args()

    sample_indices = parse_indices(args.indices)
    if len(sample_indices) == 0:
        print("ERROR: --indices resolved to an empty set.", file=sys.stderr)
        sys.exit(1)

    target_indices = parse_indices(args.target_indices) if args.target_indices else None
    if args.attack_mode == "targeted":
        target_index_map = _resolve_target_indices(sample_indices, target_indices)
    else:
        target_index_map = {}

    model = get_audio_model(
        args.audio_model_name,
        checkpoint_path=args.checkpoint_path,
        tokenizer_checkpoint_path=args.tokenizer_checkpoint_path,
        device=args.device,
        freeze=True,
    )
    available_layers = list(getattr(model, "metamer_layers"))
    if args.list_layers:
        print("\nAvailable metamer layers:")
        for layer in available_layers:
            print(f"  - {layer}")
        sys.exit(0)

    layers = args.layers if args.layers else available_layers
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

    required_indices = set(sample_indices)
    required_indices.update(target_index_map.values())
    selected_examples = _load_selected_examples(
        tfrecord_path=first_tfrecord,
        sample_indices=sorted(required_indices),
    )
    missing = sorted(required_indices - set(selected_examples.keys()))
    if missing:
        print(f"ERROR: Missing TFRecord example indices: {missing}", file=sys.stderr)
        sys.exit(1)

    epsilon_token = epsilon_to_filename_token(args.epsilon)
    attack_id = (
        f"{args.attack_mode}_{args.norm}_eps_{epsilon_token}_steps_{args.num_steps}"
        f"_loss_{args.loss_type}"
    )
    exp_root = Path(args.exp_dir) / "audio" / args.audio_model_name / "adversarial" / attack_id
    classification_root = Path(args.exp_dir) / "audio" / args.audio_model_name / "classification" / "raw"

    if args.classification_only:
        rows = []
        classification_summaries = []
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

            logits = _extract_model_logits(model, source_waveform, sr=int(file_sr))
            if logits is None:
                raise RuntimeError(
                    f"Model '{args.audio_model_name}' does not expose classifier logits. "
                    "--classification_only requires a classifier head."
                )
            labels_list = decode_audioset_labels(source_example)
            logits_summary = summarize_multilabel_logits(logits, true_labels=labels_list)
            classification_summaries.append(logits_summary)

            ytid_value = source_example.get("ytid", b"")
            if isinstance(ytid_value, bytes):
                ytid_value = ytid_value.decode("utf-8", errors="replace")
            class_label = _sanitize_label(str(ytid_value)) if ytid_value else "audioset"
            if not class_label:
                class_label = "audioset"

            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "dataset_idx": int(sample_idx),
                    "true_label_idx": int(labels_list[0]) if labels_list else "",
                    "true_label_indices": _format_label_indices(labels_list),
                    "predicted_label_idx": int(logits_summary["predicted_label_idx"]),
                    "predicted_top5_label_indices": _format_label_indices(
                        logits_summary["predicted_top5_label_indices"]
                    ),
                    "true_class_label": class_label,
                    "predicted_class_label": str(int(logits_summary["predicted_label_idx"])),
                    "is_correct": bool(logits_summary["top1_hit"]),
                    "top5_is_correct": bool(logits_summary["top5_hit"]),
                    "true_class_softmax": float(logits_summary["true_label_max_softmax"]),
                    "num_true_labels": int(logits_summary["num_true_labels"]),
                }
            )

        metrics = aggregate_multilabel_topk(classification_summaries)
        csv_path = save_adversarial_results_csv(
            rows=rows, output_path=classification_root / "summary.csv"
        )
        payload = {
            "metrics": metrics,
            "num_examples": len(rows),
            "sample_indices": [int(idx) for idx in sample_indices],
            "audio_model_name": args.audio_model_name,
            "classification_mode": "raw_source_only",
            "csv_path": str(csv_path),
        }
        with open(classification_root / "summary.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Top-1 accuracy: {metrics['top1_acc']:.4f}")
        print(f"Top-5 accuracy: {metrics['top5_acc']:.4f}")
        print(f"Saved classification summary: {csv_path}")
        print(f"Done. Results in {classification_root}")
        return

    for layer_name in layers:
        layer_output_dir = exp_root / layer_name
        rows = []
        source_classification_summaries = []
        adversarial_classification_summaries = []
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

            attack_config = RepresentationAttackConfig(
                norm=args.norm,
                epsilon=args.epsilon,
                step_size=args.step_size,
                num_steps=args.num_steps,
                num_random_starts=args.num_random_starts,
                loss_type=args.loss_type,
                clamp_range=(args.clamp_min, args.clamp_max),
                seed=args.seed + int(sample_idx),
            )
            attacker = AudioRepresentationAttacker(
                model=model,
                layer_name=layer_name,
                sample_rate=int(file_sr),
                config=attack_config,
                device=args.device,
            )
            source_rep = attacker.extract_representation(source_waveform).detach()
            source_logits = _extract_model_logits(model, source_waveform, sr=int(file_sr))

            attack_metadata: dict
            target_idx = None
            target_distance = math.nan
            target_labels: list[int] = []
            if args.attack_mode == "targeted":
                target_idx = target_index_map[sample_idx]
                target_example = selected_examples[target_idx]
                target_sr, target_waveform_np = wavfile.read(io.BytesIO(target_example["audio"]))
                if args.input_sample_rate_override is not None:
                    target_sr = int(args.input_sample_rate_override)
                target_waveform = _normalize_audio_waveform(target_waveform_np).mean(
                    dim=0, keepdim=True
                )
                target_clip = _slice_clip(
                    target_waveform,
                    sr=target_sr,
                    start_seconds=args.start_seconds,
                    clip_seconds=args.clip_seconds,
                )
                target_clip_batch = target_clip.unsqueeze(0).to(args.device)
                target_rep = attacker.extract_representation(target_clip_batch).detach()
                adversarial, attack_metadata = attacker.attack_targeted(
                    source_waveform=source_waveform,
                    target_rep=target_rep,
                    source_rep=source_rep,
                )
                target_distance = float(attack_metadata["adv_to_target_distance"])
                target_labels = decode_audioset_labels(target_example)
            else:
                adversarial, attack_metadata = attacker.attack_untargeted(
                    source_waveform=source_waveform, source_rep=source_rep
                )

            adversarial, lowpass_metadata = lowpass_filter_for_model(
                adversarial,
                model_name=args.audio_model_name,
                sample_rate=int(file_sr),
            )
            delta = adversarial - source_waveform
            norms = perturbation_norms(delta)
            adversarial_rep = attacker.extract_representation(adversarial).detach()
            source_distance = float(
                representation_distance(
                    adversarial_rep,
                    source_rep,
                    loss_type=args.loss_type,
                    reduction="mean",
                ).item()
            )
            adv_logits = _extract_model_logits(model, adversarial, sr=int(file_sr))

            ytid_value = source_example.get("ytid", b"")
            if isinstance(ytid_value, bytes):
                ytid_value = ytid_value.decode("utf-8", errors="replace")
            labels_list = decode_audioset_labels(source_example)
            class_label = _sanitize_label(str(ytid_value)) if ytid_value else "audioset"
            if not class_label:
                class_label = "audioset"

            source_logits_summary = (
                summarize_multilabel_logits(source_logits, true_labels=labels_list)
                if source_logits is not None
                else None
            )
            adv_logits_summary = (
                summarize_multilabel_logits(adv_logits, true_labels=labels_list)
                if adv_logits is not None
                else None
            )
            if source_logits_summary is not None:
                source_classification_summaries.append(source_logits_summary)
            if adv_logits_summary is not None:
                adversarial_classification_summaries.append(adv_logits_summary)

            metadata = {
                "model_name": args.audio_model_name,
                "layer_name": layer_name,
                "attack_mode": args.attack_mode,
                "attack_id": attack_id,
                "sample_rate": int(file_sr),
                "source_info": {
                    "tfrecord_example_index": int(sample_idx),
                    "ytid": ytid_value,
                    "labels": labels_list,
                },
                "target_info": {
                    "tfrecord_example_index": int(target_idx) if target_idx is not None else None,
                    "labels": target_labels,
                },
                "attack_config": {
                    "norm": args.norm,
                    "epsilon": args.epsilon,
                    "step_size": args.step_size,
                    "num_steps": args.num_steps,
                    "num_random_starts": args.num_random_starts,
                    "loss_type": args.loss_type,
                    "clamp_range": [args.clamp_min, args.clamp_max],
                },
                "representation_distance_from_source": source_distance,
                "representation_distance_to_target": target_distance,
                "delta_l2": float(norms["l2"].mean().item()),
                "delta_linf": float(norms["linf"].mean().item()),
                "snr_db": _snr_db(signal=source_waveform, noise=delta),
                "attack_trace": attack_metadata,
                **lowpass_metadata,
            }
            if source_logits is not None:
                metadata["source_logits"] = source_logits.numpy().tolist()
            if source_logits_summary is not None:
                metadata["source_classification"] = source_logits_summary
            if adv_logits is not None:
                metadata["adversarial_logits"] = adv_logits.numpy().tolist()
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

            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "dataset_idx": int(sample_idx),
                    "true_label_idx": int(labels_list[0]) if labels_list else "",
                    "true_label_indices": _format_label_indices(labels_list),
                    "predicted_label_idx": (
                        int(adv_logits_summary["predicted_label_idx"])
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "predicted_top5_label_indices": (
                        _format_label_indices(adv_logits_summary["predicted_top5_label_indices"])
                        if adv_logits_summary is not None
                        else ""
                    ),
                    "true_class_label": class_label,
                    "predicted_class_label": (
                        str(int(adv_logits_summary["predicted_label_idx"]))
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
                    "attack_mode": args.attack_mode,
                    "layer_name": layer_name,
                    "delta_l2": float(norms["l2"].mean().item()),
                    "delta_linf": float(norms["linf"].mean().item()),
                    "snr_db": metadata["snr_db"],
                    "representation_distance_from_source": source_distance,
                    "representation_distance_to_target": target_distance,
                    "target_sample_idx": int(target_idx) if target_idx is not None else "",
                    "lowpass_filter_applied": lowpass_metadata["lowpass_filter_applied"],
                    "lowpass_filter_cutoff_hz": lowpass_metadata["lowpass_filter_cutoff_hz"],
                    "lowpass_filter_effective_cutoff_hz": lowpass_metadata[
                        "lowpass_filter_effective_cutoff_hz"
                    ],
                    "lowpass_filter_reason": lowpass_metadata["lowpass_filter_reason"],
                }
            )

            print(
                f"sample={sample_idx} layer={layer_name} mode={args.attack_mode} "
                f"dist_src={source_distance:.6f} delta_l2={float(norms['l2'].mean().item()):.6f}"
            )

        summary_path = layer_output_dir / "summary.csv"
        save_adversarial_results_csv(rows=rows, output_path=summary_path)
        metrics_payload = {
            "source_metrics": aggregate_multilabel_topk(source_classification_summaries),
            "adversarial_metrics": aggregate_multilabel_topk(adversarial_classification_summaries),
        }
        with open(layer_output_dir / "classification_metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)
        with open(layer_output_dir / "summary.json", "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved layer summary: {summary_path}")

    print(f"Done. Results in {exp_root}")


if __name__ == "__main__":
    main()
