"""
CLI script for generating model metamers.

Loads a trained model from a config + checkpoint, extracts target
representations from dataset samples, and optimises noise inputs to
match those representations at specified layers for vision or audio models.

Example::

    python scripts/generate_metamers.py \\
        --config configs/vision/lipsalexnet_v1/lipsalexnet_w_max_9.json \\
        --ckpt_path experiments/lipsalexnet_w_max_9/checkpoints/epoch=25-val/loss=2.7389.ckpt \\
        --layers relu0 relu2 relu4 fc0_relu fc1_relu final \\
        --dataset imagenet \\
        --split val \\
        --indices 0-4 \\
        --num_rounds 8 \\
        --steps_per_round 3000 \\
        --seed 42
"""

import io
import pathlib
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from src.analysis.metamer import (
    MetamerGenerator,
    derive_experiment_root,
    load_eval_dataset_with_subset,
    load_model_from_checkpoint,
    resolve_model_normalize_fn,
)
from src.analysis.saving import save_metamer_results
from src.models.audio import get_audio_model

# ---------------------------------------------------------------------------
# Index parsing (supports "0", "0,1,2", "0-9", "0-4,10,20-24")
# ---------------------------------------------------------------------------


def parse_indices(spec: str) -> list[int]:
    """Parse a comma-separated list of indices or ranges.

    Examples::

        "0"       -> [0]
        "0,1,2"   -> [0, 1, 2]
        "0-4"     -> [0, 1, 2, 3, 4]
        "0-2,5,8-10" -> [0, 1, 2, 5, 8, 9, 10]
    """
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def _normalize_audio_waveform(waveform_np: np.ndarray) -> torch.Tensor:
    """Convert scipy waveform output [T, C] or [T] to torch [C, T] float32 in [-1, 1]."""
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


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(description="Generate model metamers")
    parser.add_argument(
        "--modality", type=str, default="vision", choices=["vision", "audio"]
    )
    parser.add_argument("--config", type=str, default=None, help="Path to model config")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="Layer names (space-separated). Defaults to all metamer_layers.",
    )
    parser.add_argument(
        "--list_layers",
        action="store_true",
        help="Print available layers and exit",
    )
    parser.add_argument(
        "--audio_model_name",
        type=str,
        default="clap",
        help="Audio model registry name (used when --modality audio)",
    )
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset root directory (defaults to value from config)",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--indices",
        type=str,
        default="0",
        help='Sample indices, e.g. "0,1,2" or "0-9". '
        "When --imagenet_subset is set, these index into that subset ordering.",
    )
    parser.add_argument(
        "--imagenet_subset",
        type=str,
        default="none",
        choices=["none", "imagenet_400_val"],
        help="Optional ImageNet subset selection mode.",
    )
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument(
        "--audioset_root",
        type=str,
        default="/home/rphess/orcd/datasets/AudioSet",
        help="AudioSet root directory containing audioset-2m-part* splits",
    )
    parser.add_argument(
        "--audioset_split_glob",
        type=str,
        default="audioset-2m-part*",
        help="Glob to locate AudioSet split dirs under --audioset_root",
    )
    parser.add_argument(
        "--tfrecord_glob",
        type=str,
        default="*.tfrecords",
        help="Glob to locate TFRecords in the selected split dir",
    )
    parser.add_argument(
        "--clip_seconds",
        type=float,
        default=2.0,
        help="Audio clip duration in seconds (for --modality audio)",
    )
    parser.add_argument(
        "--start_seconds",
        type=float,
        default=0.0,
        help="Audio clip start offset in seconds (for --modality audio)",
    )
    parser.add_argument(
        "--input_sample_rate_override",
        type=int,
        default=None,
        help="Optional sample-rate override for decoded TFRecord audio",
    )
    parser.add_argument("--lr", type=float, default=1.0, help="Initial learning rate")
    parser.add_argument(
        "--num_rounds", type=int, default=8, help="Number of optimisation rounds"
    )
    parser.add_argument(
        "--steps_per_round", type=int, default=3000, help="Gradient steps per round"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="LR decay factor per round"
    )
    parser.add_argument(
        "--fake_relu",
        type=str,
        default="true",
        help="Use fake ReLU for gradient flow (true/false)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="normalized_l2",
        choices=["normalized_l2", "l2", "cosine"],
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.05,
        help="Scale of initial Gaussian noise",
    )
    parser.add_argument(
        "--noise_mean",
        type=float,
        default=0.5,
        help="Mean of initial noise (0.5 = gray)",
    )
    parser.add_argument(
        "--lambda_tv",
        type=float,
        default=0.0,
        help="Weight for TV smoothness regularizer (e.g. 5e-6, 5e-5, 5e-4)",
    )
    parser.add_argument(
        "--lambda_range",
        type=float,
        default=0.0,
        help="Weight for Lp range regularizer (e.g. 0.005)",
    )
    parser.add_argument(
        "--range_norm_p",
        type=int,
        default=6,
        help="Norm order p for the range regularizer (default 6)",
    )
    parser.add_argument(
        "--clamp_min",
        type=float,
        default=0.0,
        help="Lower clamp bound for the optimized metamer",
    )
    parser.add_argument(
        "--clamp_max",
        type=float,
        default=1.0,
        help="Upper clamp bound for the optimized metamer",
    )

    args = parser.parse_args()

    if args.modality == "audio" and args.indices == "0":
        # Keep vision default lightweight, but use the standard AudioSet example set.
        args.indices = "0-9"
        print("No audio --indices provided; defaulting to AudioSet examples 0-9.")

    # Parse boolean
    fake_relu = args.fake_relu.lower() in ("true", "1", "yes")

    sample_indices = parse_indices(args.indices)
    print(f"Sample indices (requested): {sample_indices}")

    if args.modality == "vision":
        if args.config is None or args.ckpt_path is None:
            print("ERROR: --config and --ckpt_path are required for --modality vision.")
            sys.exit(1)

        print(f"Loading model from {args.config} + {args.ckpt_path}")
        model, config = load_model_from_checkpoint(
            args.config, args.ckpt_path, device=args.device
        )
        print(f"Model: {model}")
        normalize_fn = resolve_model_normalize_fn(model, config)

        available_layers = list(getattr(model, "metamer_layers"))
        if args.list_layers:
            print("\nAvailable metamer layers:")
            for layer in available_layers:
                print(f"  - {layer}")
            sys.exit(0)

        layers = args.layers if args.layers else available_layers
        print(f"Generating metamers for layers: {layers}")

        exp_root = derive_experiment_root(
            config_path=args.config, exp_dir=args.exp_dir, config_root="configs"
        )

        data_settings = config.get("data_settings", {})
        data_dir = args.data_dir or data_settings.get("data_dir")
        if data_dir is None:
            print(
                "ERROR: --data_dir not provided and not found in config.", file=sys.stderr
            )
            sys.exit(1)
        print(f"Loading {args.dataset} dataset from {data_dir} (split={args.split})")
        dataset, selected_subset_indices, use_subset_indices = load_eval_dataset_with_subset(
            config=config,
            dataset_name=args.dataset,
            split=args.split,
            data_dir=args.data_dir,
            imagenet_subset=args.imagenet_subset,
        )

        selected_dataset_indices: list[int] = []
        if args.imagenet_subset != "none":
            if args.dataset != "imagenet":
                print(
                    "ERROR: --imagenet_subset is only supported with --dataset imagenet.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if args.split != "val":
                print(
                    "ERROR: --imagenet_subset imagenet_400_val requires --split val.",
                    file=sys.stderr,
                )
                sys.exit(1)

            if args.imagenet_subset == "imagenet_400_val":
                print("Resolving ImageNet subset: imagenet_400_val")
                selected_dataset_indices = selected_subset_indices
                print(
                    f"Resolved {len(selected_dataset_indices)} images for imagenet_400_val subset."
                )
                dataset_obj = getattr(dataset, "dataset", dataset)
                subset_samples = getattr(dataset_obj, "samples")
                preview = [
                    pathlib.Path(subset_samples[i][0]).name
                    for i in selected_dataset_indices[:5]
                ]
                print(f"Subset reproducibility preview (first 5 filenames): {preview}")

        if use_subset_indices:
            subset_max_index = len(selected_dataset_indices) - 1
            if any(i < 0 or i > subset_max_index for i in sample_indices):
                print(
                    "ERROR: At least one requested --indices value is out of range for the "
                    f"imagenet_400_val subset [0, {subset_max_index}].",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(
                "Interpreting requested indices in imagenet_400_val ordering "
                f"(size={len(selected_dataset_indices)})."
            )
        else:
            print(f"Sample indices (dataset indices): {sample_indices}")

        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

        for layer_name in layers:
            metamer_dir = exp_root / "metamers" / layer_name
            print(f"\n{'=' * 60}")
            print(f"Layer: {layer_name}  ->  {metamer_dir}")
            print(f"{'=' * 60}")

            generator = MetamerGenerator(
                model=model,
                layer_name=layer_name,
                lr=args.lr,
                num_rounds=args.num_rounds,
                steps_per_round=args.steps_per_round,
                lr_decay=args.lr_decay,
                fake_relu=fake_relu,
                loss_type=args.loss_type,
                clamp_range=(args.clamp_min, args.clamp_max),
                lambda_tv=args.lambda_tv,
                lambda_range=args.lambda_range,
                range_norm_p=args.range_norm_p,
                normalize_fn=normalize_fn,
                noise_scale=args.noise_scale,
                noise_mean=args.noise_mean,
                device=args.device,
            )

            for idx in sample_indices:
                dataset_idx = selected_dataset_indices[idx] if use_subset_indices else idx
                image, label = dataset[dataset_idx]
                class_label = idx_to_class.get(label, str(label))
                if use_subset_indices:
                    print(
                        f"\n  Sample subset_idx={idx} -> dataset_idx={dataset_idx}: "
                        f"class={class_label} (label={label})"
                    )
                else:
                    print(f"\n  Sample {idx}: class={class_label} (label={label})")

                image_batch = image.unsqueeze(0)
                target_rep = generator.extract_target(image_batch)
                metamer, metadata = generator.generate(
                    target_rep, shape=image_batch.shape, seed=args.seed + idx
                )
                metadata["layer_name"] = layer_name
                metadata["model_name"] = config.get("model_name", "unknown")
                metadata["config_path"] = str(args.config)
                metadata["ckpt_path"] = str(args.ckpt_path)

                save_metamer_results(
                    metamer=metamer,
                    original=image_batch,
                    metadata=metadata,
                    output_dir=metamer_dir,
                    sample_idx=idx,
                    class_label=class_label,
                    modality="vision",
                )
                print(f"  Saved to {metamer_dir}")

        print(f"\nDone! Results in {exp_root / 'metamers'}")
        return

    supported_audio_model_names = {
        "audiomae_as2m",
        "audiomae_as2m_ft_as20k",
        "beats",
        "beats_iter3",
        "beats_iter3_plus_as2m",
        "clap",
    }
    if args.audio_model_name not in supported_audio_model_names:
        print(
            "ERROR: This audio metamer path currently supports only "
            f"--audio_model_name in {sorted(supported_audio_model_names)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = get_audio_model(args.audio_model_name, device=args.device, freeze=True)
    print(f"Loaded audio model: {args.audio_model_name}")
    available_layers = list(getattr(model, "metamer_layers"))
    if args.list_layers:
        print("\nAvailable metamer layers:")
        for layer in available_layers:
            print(f"  - {layer}")
        sys.exit(0)

    layers = args.layers if args.layers else available_layers
    print(f"Generating metamers for layers: {layers}")

    audioset_root = Path(args.audioset_root)
    if not audioset_root.exists():
        print(f"ERROR: AudioSet root not found: {audioset_root}", file=sys.stderr)
        sys.exit(1)
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
    print(f"Using first split: {first_split_dir}")
    print(f"Using first TFRecord: {first_tfrecord.name}")

    from tfrecord.reader import example_loader

    max_requested_idx = max(sample_indices)
    tfrecord_examples = example_loader(
        str(first_tfrecord),
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
    if len(selected_examples) != len(sample_indices):
        missing = sorted(set(sample_indices) - set(selected_examples.keys()))
        print(
            f"ERROR: Missing TFRecord example indices in {first_tfrecord}: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    exp_root = Path(args.exp_dir) / "audio" / args.audio_model_name

    for layer_name in layers:
        metamer_dir = exp_root / "metamers" / layer_name
        print(f"\n{'=' * 60}")
        print(f"Layer: {layer_name}  ->  {metamer_dir}")
        print(f"{'=' * 60}")

        generator = MetamerGenerator(
            model=model,
            layer_name=layer_name,
            lr=args.lr,
            num_rounds=args.num_rounds,
            steps_per_round=args.steps_per_round,
            lr_decay=args.lr_decay,
            fake_relu=fake_relu,
            loss_type=args.loss_type,
            clamp_range=(args.clamp_min, args.clamp_max),
            lambda_tv=args.lambda_tv,
            lambda_range=args.lambda_range,
            range_norm_p=args.range_norm_p,
            normalize_fn=lambda x: x,
            noise_scale=args.noise_scale,
            noise_mean=args.noise_mean,
            device=args.device,
        )

        for idx in sample_indices:
            example = selected_examples[idx]
            if "audio" not in example:
                print(
                    f"ERROR: Expected 'audio' bytes in TFRecord example {idx}. "
                    f"Keys: {sorted(example.keys())}",
                    file=sys.stderr,
                )
                sys.exit(1)

            file_sr, waveform_np = wavfile.read(io.BytesIO(example["audio"]))
            if args.input_sample_rate_override is not None:
                file_sr = int(args.input_sample_rate_override)

            waveform = _normalize_audio_waveform(waveform_np)
            if waveform.dim() != 2:
                print(
                    f"ERROR: Expected waveform shape [C, T], got {tuple(waveform.shape)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            waveform_mono = waveform.mean(dim=0, keepdim=True)

            start_sample = int(args.start_seconds * file_sr)
            clip_samples = int(args.clip_seconds * file_sr)
            end_sample = start_sample + clip_samples
            waveform_clip = waveform_mono[:, start_sample:end_sample]
            if waveform_clip.shape[-1] == 0:
                print(
                    "ERROR: Empty clip after slicing. Adjust --start_seconds/--clip_seconds.",
                    file=sys.stderr,
                )
                sys.exit(1)

            original_waveform = waveform_clip.unsqueeze(0).to(args.device)  # [B,1,T]
            original_waveform_model = original_waveform.squeeze(1)  # [B,T]

            ytid_value = example.get("ytid", b"")
            if isinstance(ytid_value, bytes):
                ytid_value = ytid_value.decode("utf-8", errors="replace")
            labels_value = example.get("labels")
            labels_list = []
            if labels_value is not None:
                if hasattr(labels_value, "tolist"):
                    labels_list = [int(x) for x in labels_value.tolist()]
                elif isinstance(labels_value, (list, tuple)):
                    labels_list = [int(x) for x in labels_value]
                else:
                    labels_list = [int(labels_value)]
            class_label = _sanitize_label(str(ytid_value)) if ytid_value else "audioset"
            if not class_label:
                class_label = "audioset"

            print(
                f"\n  Sample {idx}: ytid={ytid_value} labels={labels_list} "
                f"sr={file_sr} shape={tuple(original_waveform_model.shape)}"
            )

            generator.forward_kwargs = {"sr": int(file_sr)}
            target_rep = generator.extract_target(original_waveform_model)
            metamer, metadata = generator.generate(
                target_rep, shape=original_waveform_model.shape, seed=args.seed + idx
            )

            metadata["layer_name"] = layer_name
            metadata["model_name"] = args.audio_model_name
            metadata["sample_rate"] = int(file_sr)
            metadata["source_info"] = {
                "audioset_root": str(audioset_root),
                "first_split_dir": str(first_split_dir),
                "first_tfrecord": str(first_tfrecord),
                "tfrecord_example_index": int(idx),
                "ytid": ytid_value,
                "labels": labels_list,
            }

            save_metamer_results(
                metamer=metamer,
                original=original_waveform_model,
                metadata=metadata,
                output_dir=metamer_dir,
                sample_idx=idx,
                class_label=class_label,
                modality="audio",
            )
            print(f"  Saved to {metamer_dir}")

    print(f"\nDone! Results in {exp_root / 'metamers'}")


if __name__ == "__main__":
    main()
