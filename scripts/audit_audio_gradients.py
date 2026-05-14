"""Audit waveform-to-representation gradients for audio model wrappers."""

from __future__ import annotations

import json
from argparse import ArgumentParser

import torch

from src.analysis.audio_adversarial import audit_waveform_gradient
from src.models.audio import get_audio_model


def main() -> None:
    parser = ArgumentParser(description="Audit audio representation gradients")
    parser.add_argument(
        "--audio_model_name",
        type=str,
        default="audiomae_as2m_ft_as20k",
        help="Audio model registry entry to load.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional model checkpoint override.",
    )
    parser.add_argument(
        "--tokenizer_checkpoint_path",
        type=str,
        default=None,
        help="Optional tokenizer checkpoint for BEATs-style models.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer name to audit. Defaults to first metamer layer.",
    )
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--clip_seconds", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = get_audio_model(
        args.audio_model_name,
        checkpoint_path=args.checkpoint_path,
        tokenizer_checkpoint_path=args.tokenizer_checkpoint_path,
        device=args.device,
        freeze=True,
    )
    available_layers = list(getattr(model, "metamer_layers", []))
    if len(available_layers) == 0:
        raise RuntimeError(
            f"Model {args.audio_model_name!r} does not expose metamer_layers."
        )
    layer = args.layer or available_layers[0]

    sample_count = int(round(args.sample_rate * args.clip_seconds))
    waveform = torch.randn(1, 1, sample_count, device=args.device)
    result = audit_waveform_gradient(
        model=model,
        waveform=waveform,
        sample_rate=args.sample_rate,
        layer_name=layer,
        device=args.device,
    )
    result["audio_model_name"] = args.audio_model_name
    result["chosen_layer"] = layer
    result["available_metamer_layers"] = available_layers
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
