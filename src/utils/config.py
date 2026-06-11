"""
Configuration loading and CLI override utilities.

Centralises config parsing so that all training / evaluation scripts share
the same logic.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Union

import yaml


def load_config(path: Union[str, Path]) -> dict:
    """
    Load a training config from a JSON or YAML file.

    Args:
        path: Path to the config file.

    Returns:
        Parsed config dictionary.
    """
    path = Path(path)
    with open(path, "r") as f:
        if path.suffix == ".json":
            return json.load(f)
        else:
            return yaml.load(f, Loader=yaml.FullLoader)


def apply_cli_overrides(config: dict, args: Namespace) -> dict:
    """
    Apply command-line overrides to a config dict (in-place).

    Handles:
      - ``--num_workers`` -> ``config["data_settings"]["num_workers"]``
      - ``--gpus``        -> ``config["ngpus"]``
      - ``--lr``          -> learning rate in optimizer settings

    Args:
        config: The loaded config dict.
        args: Parsed CLI arguments.

    Returns:
        The (mutated) config dict for convenience.
    """
    if hasattr(args, "num_workers") and args.num_workers is not None:
        config.setdefault("data_settings", {})["num_workers"] = args.num_workers

    if hasattr(args, "gpus") and args.gpus is not None:
        config["ngpus"] = args.gpus

    if hasattr(args, "lr") and args.lr is not None:
        _override_lr(config, args.lr)

    return config


def _override_lr(config: dict, lr: float) -> None:
    """Override the learning rate in all optimizer settings."""
    optim = config.get("optim_settings", {})
    if optim.get("mult_optimizers", False):
        if "linear_optimizer" in optim:
            optim["linear_optimizer"]["kwargs"]["lr"] = lr
        if "conv_optimizer" in optim:
            optim["conv_optimizer"]["kwargs"]["lr"] = lr
    else:
        if "optimizer" in optim:
            optim["optimizer"]["kwargs"]["lr"] = lr
