"""
Optimizer and LR-scheduler setup utilities.

The main entry point is ``configure_optimizers``, which builds one or two
optimizers (with optional LR schedulers) from ``optim_settings`` in the
training config.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from src.models.layers.LipsLayers import LipsConv2d, LipsLinear


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_optimizers(optim_settings: dict, model: nn.Module) -> list:
    """
    Build optimizer(s) and optional LR schedulers for *model* according to
    ``optim_settings``.

    For the **multi-optimizer** case (``mult_optimizers: true``), parameters
    are automatically split into *conv* (``LipsConv2d``) and *linear*
    (``LipsLinear`` / remaining) groups.

    Returns a list of dicts, each with an ``"optimizer"`` key and an
    optional ``"lr_scheduler"`` key -- the format expected by Lightning
    ``configure_optimizers`` in manual-optimization mode.
    """
    if optim_settings.get("mult_optimizers", False):
        conv_params, linear_params = _split_parameters(model)
        return _build_multi_optimizer(optim_settings, conv_params, linear_params)
    else:
        return _build_single_optimizer(optim_settings, model.parameters())


# ---------------------------------------------------------------------------
# Parameter splitting
# ---------------------------------------------------------------------------

def _split_parameters(model: nn.Module):
    """Split model parameters into conv-type and linear-type groups."""
    conv_param_ids = set()
    linear_param_ids = set()

    for _name, module in model.named_modules():
        if isinstance(module, LipsConv2d):
            for p in module.parameters():
                conv_param_ids.add(id(p))
        elif isinstance(module, LipsLinear):
            for p in module.parameters():
                linear_param_ids.add(id(p))

    # Parameters that are neither LipsConv2d nor LipsLinear (e.g. BatchNorm)
    # fall into the conv group by default.
    conv_params = []
    linear_params = []
    for p in model.parameters():
        pid = id(p)
        if pid in linear_param_ids:
            linear_params.append(p)
        else:
            conv_params.append(p)

    return conv_params, linear_params


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer, scheduler_config):
    """Build an LR scheduler from a config dict, or return None."""
    if scheduler_config is None:
        return None
    scheduler_class = scheduler_config.get("class")
    scheduler_kwargs = scheduler_config.get("kwargs", {})
    if scheduler_class is None:
        return None
    scheduler = scheduler_class(optimizer, **scheduler_kwargs)
    return {
        "scheduler": scheduler,
        "interval": scheduler_kwargs.get("interval", "step"),
        "frequency": scheduler_kwargs.get("frequency", 1),
    }


def _build_single_optimizer(optim_settings, parameters) -> list:
    """Build a single optimizer (+ optional scheduler)."""
    opt_cfg = optim_settings["optimizer"]
    opt_class = getattr(torch.optim, opt_cfg["name"])
    optimizer = opt_class([{"params": list(parameters)}], **opt_cfg["kwargs"].copy())

    result = [{"optimizer": optimizer}]
    sched = _build_scheduler(optimizer, opt_cfg.get("lr_scheduler"))
    if sched is not None:
        result[0]["lr_scheduler"] = sched
    return result


def _build_multi_optimizer(optim_settings, conv_params, linear_params) -> list:
    """Build separate conv and linear optimizers (+ optional schedulers)."""
    # Linear optimizer
    lin_cfg = optim_settings["linear_optimizer"]
    lin_class = getattr(torch.optim, lin_cfg["name"])
    lin_opt = lin_class([{"params": linear_params}], **lin_cfg["kwargs"].copy())

    # Conv optimizer
    conv_cfg = optim_settings["conv_optimizer"]
    conv_class = getattr(torch.optim, conv_cfg["name"])
    conv_opt = conv_class([{"params": conv_params}], **conv_cfg["kwargs"].copy())

    result = [{"optimizer": lin_opt}, {"optimizer": conv_opt}]

    lin_sched = _build_scheduler(lin_opt, lin_cfg.get("lr_scheduler"))
    if lin_sched is not None:
        result[0]["lr_scheduler"] = lin_sched

    conv_sched = _build_scheduler(conv_opt, conv_cfg.get("lr_scheduler"))
    if conv_sched is not None:
        result[1]["lr_scheduler"] = conv_sched

    return result
