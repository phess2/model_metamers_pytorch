"""
CLI script for generating model metamers.

Loads a trained model from a config + checkpoint, extracts target
representations from dataset samples, and optimises noise inputs to
match those representations at specified layers.

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

import pathlib
import sys
from argparse import ArgumentParser
from typing import cast

from src.analysis.metamer import MetamerGenerator, load_model_from_checkpoint
from src.analysis.saving import save_metamer_results
from src.data.datasets import ImageNetFolder, get_vision_dataset

CONFIG_ROOT = pathlib.Path("configs").resolve()


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


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(description="Generate model metamers")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config JSON/YAML"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to model checkpoint"
    )
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

    args = parser.parse_args()

    # Parse boolean
    fake_relu = args.fake_relu.lower() in ("true", "1", "yes")

    # ---- Load model -------------------------------------------------
    print(f"Loading model from {args.config} + {args.ckpt_path}")
    model, config = load_model_from_checkpoint(
        args.config, args.ckpt_path, device=args.device
    )
    print(f"Model: {model}")

    # ---- List layers mode -------------------------------------------
    available_layers = list(getattr(model, "metamer_layers"))
    if args.list_layers:
        print("\nAvailable metamer layers:")
        for layer in available_layers:
            print(f"  - {layer}")
        sys.exit(0)

    # ---- Determine layers -------------------------------------------
    layers = args.layers if args.layers else available_layers
    print(f"Generating metamers for layers: {layers}")

    # ---- Derive experiment path -------------------------------------
    config_path = pathlib.Path(args.config).resolve()
    exp_dir = pathlib.Path(args.exp_dir)
    try:
        rel = config_path.relative_to(CONFIG_ROOT)
        exp_root = exp_dir / rel.with_suffix("")
    except ValueError:
        exp_root = exp_dir / config_path.stem

    # ---- Load dataset -----------------------------------------------
    data_settings = config.get("data_settings", {})
    data_dir = args.data_dir or data_settings.get("data_dir")
    image_size = data_settings.get("image_size", 224)

    if data_dir is None:
        print(
            "ERROR: --data_dir not provided and not found in config.", file=sys.stderr
        )
        sys.exit(1)

    print(f"Loading {args.dataset} dataset from {data_dir} (split={args.split})")
    # raw=True: images in pixel space [0, 1] (no ImageNet normalisation).
    # MetamerGenerator handles normalisation internally before forward passes.
    if args.imagenet_subset != "none":
        dataset, selected_subset_indices = cast(
            tuple[ImageNetFolder, list[int]],
            get_vision_dataset(
                args.dataset,
                data_dir,
                image_size,
                stage="validate",
                raw=True,
                imagenet_subset=args.imagenet_subset,
                return_imagenet_subset_indices=True,
            ),
        )
    else:
        dataset = cast(
            ImageNetFolder,
            get_vision_dataset(
                args.dataset,
                data_dir,
                image_size,
                stage="validate",
                raw=True,
            ),
        )
        selected_subset_indices = []

    # ---- Optional ImageNet subset ----------------------------------
    selected_dataset_indices: list[int] = []
    use_subset_indices = args.imagenet_subset != "none"
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
            if len(selected_dataset_indices) != 400:
                print(
                    "ERROR: Expected exactly 400 images in imagenet_400_val subset, got "
                    f"{len(selected_dataset_indices)}.",
                    file=sys.stderr,
                )
                sys.exit(1)
            dataset_obj = getattr(dataset, "dataset", dataset)
            subset_samples = getattr(dataset_obj, "samples")
            preview = [
                pathlib.Path(subset_samples[i][0]).name
                for i in selected_dataset_indices[:5]
            ]
            print(f"Subset reproducibility preview (first 5 filenames): {preview}")

    # ---- Parse indices -----------------------------------------------
    sample_indices = parse_indices(args.indices)
    print(f"Sample indices (requested): {sample_indices}")

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

    # ---- Get class names mapping ------------------------------------
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # ---- Generate metamers ------------------------------------------
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
            lambda_tv=args.lambda_tv,
            lambda_range=args.lambda_range,
            range_norm_p=args.range_norm_p,
            noise_scale=args.noise_scale,
            noise_mean=args.noise_mean,
            device=args.device,
        )

        for idx in sample_indices:
            dataset_idx = idx
            if use_subset_indices:
                dataset_idx = selected_dataset_indices[idx]

            image, label = dataset[dataset_idx]
            class_label = idx_to_class.get(label, str(label))
            if use_subset_indices:
                print(
                    f"\n  Sample subset_idx={idx} -> dataset_idx={dataset_idx}: "
                    f"class={class_label} (label={label})"
                )
            else:
                print(f"\n  Sample {idx}: class={class_label} (label={label})")

            # Add batch dimension
            image_batch = image.unsqueeze(0)

            # Extract target representation
            target_rep = generator.extract_target(image_batch)

            # Generate metamer
            metamer, metadata = generator.generate(
                target_rep, shape=image_batch.shape, seed=args.seed + idx
            )

            # Add layer/model info to metadata
            metadata["layer_name"] = layer_name
            metadata["model_name"] = config.get("model_name", "unknown")
            metadata["config_path"] = str(args.config)
            metadata["ckpt_path"] = str(args.ckpt_path)

            # Save results
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


if __name__ == "__main__":
    main()
