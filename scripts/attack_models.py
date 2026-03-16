"""
CLI script for evaluating adversarial robustness on ImageNet-400 validation.

Loads a model from config + checkpoint, runs untargeted L2 adversarial attacks
on the curated ``imagenet_400_val`` subset, and writes per-sample CSV results.
"""

import sys
from argparse import ArgumentParser

import torch

from src.analysis.metamer import (
    L2AdversarialAttacker,
    derive_experiment_root,
    load_eval_dataset_with_subset,
    load_model_from_checkpoint,
    resolve_model_normalize_fn,
)
from src.analysis.saving import save_adversarial_results_csv
from src.training.metrics import topk_accuracy


def parse_indices(spec: str) -> list[int]:
    """Parse comma-separated indices/ranges into sorted unique integers."""
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def epsilon_to_filename_token(epsilon: float) -> str:
    """Convert epsilon to a filesystem-friendly token."""
    return format(epsilon, "g").replace(".", "p")


def main():
    parser = ArgumentParser(description="Evaluate adversarial robustness on ImageNet-400")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config JSON/YAML"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to model checkpoint"
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
        "--imagenet_subset",
        type=str,
        default="imagenet_400_val",
        choices=["imagenet_400_val"],
        help="ImageNet subset selection mode.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="0-399",
        help='Subset indices to attack, e.g. "0-9" or "0,4,10".',
    )
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument(
        "--epsilon", type=float, required=True, help="L2 epsilon budget for attack"
    )
    parser.add_argument(
        "--step_size", type=float, default=0.1, help="L2 step size per PGD iteration"
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of PGD iterations"
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading model from {args.config} + {args.ckpt_path}")
    model, config = load_model_from_checkpoint(
        args.config, args.ckpt_path, device=args.device
    )
    normalize_fn = resolve_model_normalize_fn(model, config)

    exp_root = derive_experiment_root(
        config_path=args.config, exp_dir=args.exp_dir, config_root="configs"
    )
    adv_dir = exp_root / "adversarial"

    data_settings = config.get("data_settings", {})
    data_dir = args.data_dir or data_settings.get("data_dir")
    if data_dir is None:
        print(
            "ERROR: --data_dir not provided and not found in config.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {args.dataset} dataset from {data_dir} (split={args.split})")
    try:
        dataset, selected_subset_indices, use_subset_indices = load_eval_dataset_with_subset(
            config=config,
            dataset_name=args.dataset,
            split=args.split,
            data_dir=args.data_dir,
            imagenet_subset=args.imagenet_subset,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not use_subset_indices:
        print(
            "ERROR: attack_models currently requires --imagenet_subset imagenet_400_val.",
            file=sys.stderr,
        )
        sys.exit(1)

    sample_indices = parse_indices(args.indices)
    subset_max_index = len(selected_subset_indices) - 1
    if any(i < 0 or i > subset_max_index for i in sample_indices):
        print(
            "ERROR: At least one requested --indices value is out of range for the "
            f"imagenet_400_val subset [0, {subset_max_index}].",
            file=sys.stderr,
        )
        sys.exit(1)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    attacker = L2AdversarialAttacker(
        model=model,
        normalize_fn=normalize_fn,
        epsilon=args.epsilon,
        step_size=args.step_size,
        num_steps=args.num_steps,
        device=args.device,
    )

    rows = []
    attacked_logits = []
    attacked_targets = []

    print(
        f"Running untargeted L2 attack on {len(sample_indices)} samples "
        f"(epsilon={args.epsilon}, steps={args.num_steps}, step_size={args.step_size})"
    )
    for subset_idx in sample_indices:
        dataset_idx = selected_subset_indices[subset_idx]
        image, label = dataset[dataset_idx]

        image_batch = image.unsqueeze(0)
        target = torch.tensor([label], dtype=torch.long, device=args.device)
        adversarial = attacker.attack(image_batch, target)

        with torch.no_grad():
            logits = model(normalize_fn(adversarial))
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(logits.argmax(dim=1).item())
            true_softmax = float(probs[0, int(label)].item())

        true_class_label = idx_to_class.get(int(label), str(int(label)))
        pred_class_label = idx_to_class.get(pred_idx, str(pred_idx))
        is_correct = pred_idx == int(label)

        rows.append(
            {
                "sample_idx": subset_idx,
                "dataset_idx": dataset_idx,
                "true_label_idx": int(label),
                "predicted_label_idx": pred_idx,
                "true_class_label": true_class_label,
                "predicted_class_label": pred_class_label,
                "is_correct": is_correct,
                "true_class_softmax": true_softmax,
                "epsilon_l2": args.epsilon,
            }
        )
        attacked_logits.append(logits.detach().cpu())
        attacked_targets.append(target.detach().cpu())

    if attacked_logits:
        metrics = topk_accuracy(
            output=torch.cat(attacked_logits, dim=0),
            target=torch.cat(attacked_targets, dim=0),
            topk=(1, 5),
        )
    else:
        metrics = {"top1_acc": 0.0, "top5_acc": 0.0}

    epsilon_token = epsilon_to_filename_token(args.epsilon)
    output_csv = adv_dir / f"adversarial_l2_eps_{epsilon_token}.csv"
    saved_path = save_adversarial_results_csv(rows=rows, output_path=output_csv)

    print("\nAdversarial evaluation complete.")
    print(f"Top-1 accuracy: {metrics['top1_acc']:.4f}")
    print(f"Top-5 accuracy: {metrics['top5_acc']:.4f}")
    print(f"Saved per-sample results to: {saved_path}")


if __name__ == "__main__":
    main()
