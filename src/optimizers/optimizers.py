"""
Optimizer and LR-scheduler setup utilities.

The main entry point is ``configure_optimizers``, which builds one or two
optimizers (with optional LR schedulers) from ``optim_settings`` in the
training config.
"""

from __future__ import annotations


import torch
from torch import nn

from src.models.layers.LipsLayers import LipsConv2d, LipsLinear


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_optimizers(
    optim_settings: dict,
    model: nn.Module,
    steps_per_epoch: int = 1,
) -> list:
    """
    Build optimizer(s) and optional LR schedulers for *model* according to
    ``optim_settings``.

    For the **multi-optimizer** case (``mult_optimizers: true``), parameters
    are automatically split into *conv* (``LipsConv2d``) and *linear*
    (``LipsLinear`` / remaining) groups.

    Parameters
    ----------
    steps_per_epoch : int
        Number of training steps in one epoch.  Used to convert epoch-based
        scheduler parameters to step-based when
        ``convert_epochs_to_steps`` is set in the scheduler config.

    Returns a list of dicts, each with an ``"optimizer"`` key and an
    optional ``"lr_scheduler"`` key -- the format expected by Lightning
    ``configure_optimizers`` in manual-optimization mode.
    """
    if optim_settings.get("mult_optimizers", False):
        conv_params, linear_params = _split_parameters(model)
        return _build_multi_optimizer(
            optim_settings, conv_params, linear_params, steps_per_epoch
        )
    else:
        return _build_single_optimizer(
            optim_settings, model.parameters(), steps_per_epoch
        )


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


def _build_sub_scheduler(
    optimizer, sub_config, steps_per_epoch: int = 1, convert: bool = False
):
    """Build a single (non-composite) LR scheduler for use inside SequentialLR.

    When *convert* is True, any ``total_iters`` or ``T_max`` values in
    *kwargs* are multiplied by *steps_per_epoch* so that epoch-denominated
    config values become step-denominated at runtime.
    """
    name = sub_config["name"]
    kwargs = sub_config.get("kwargs", {}).copy()

    if convert and steps_per_epoch > 1:
        # Keys whose values represent durations in epochs that should be
        # converted to steps.
        _EPOCH_KEYS = ("total_iters", "T_max")
        for key in _EPOCH_KEYS:
            if key in kwargs:
                kwargs[key] = int(kwargs[key] * steps_per_epoch)

    sched_class = getattr(torch.optim.lr_scheduler, name)
    return sched_class(optimizer, **kwargs)


def _build_scheduler(optimizer, scheduler_config, steps_per_epoch: int = 1):
    """Build an LR scheduler from a config dict, or return None.

    Supports both simple schedulers (looked up by ``name`` from
    ``torch.optim.lr_scheduler``) and composite ``SequentialLR`` configs
    that specify a list of sub-schedulers and milestones.

    When the config contains ``"convert_epochs_to_steps": true``, all
    duration-related kwargs (``total_iters``, ``T_max``) and
    ``milestones`` are multiplied by *steps_per_epoch* so the scheduler
    operates per training step while the config stays in epoch units.

    The returned dict includes an ``"interval"`` key (``"step"`` or
    ``"epoch"``) that the Lightning module uses to decide *when* to call
    ``scheduler.step()``.
    """
    if scheduler_config is None:
        return None
    name = scheduler_config.get("name")
    if name is None:
        return None

    interval = scheduler_config.get("interval", "step")
    convert = scheduler_config.get("convert_epochs_to_steps", False)
    sched_class = getattr(torch.optim.lr_scheduler, name)

    if name == "SequentialLR":
        # Build each sub-scheduler, then compose them.
        sub_scheds = [
            _build_sub_scheduler(optimizer, s, steps_per_epoch, convert)
            for s in scheduler_config["schedulers"]
        ]
        milestones = list(scheduler_config["milestones"])
        if convert and steps_per_epoch > 1:
            milestones = [int(m * steps_per_epoch) for m in milestones]
        scheduler = sched_class(
            optimizer,
            schedulers=sub_scheds,
            milestones=milestones,
        )
    else:
        kwargs = scheduler_config.get("kwargs", {}).copy()
        if convert and steps_per_epoch > 1:
            for key in ("total_iters", "T_max"):
                if key in kwargs:
                    kwargs[key] = int(kwargs[key] * steps_per_epoch)
        scheduler = sched_class(optimizer, **kwargs)

    return {
        "scheduler": scheduler,
        "interval": interval,
        "frequency": 1,
    }


def _build_single_optimizer(
    optim_settings, parameters, steps_per_epoch: int = 1
) -> list:
    """Build a single optimizer (+ optional scheduler)."""
    opt_cfg = optim_settings["optimizer"]
    opt_class = getattr(torch.optim, opt_cfg["name"])
    optimizer = opt_class([{"params": list(parameters)}], **opt_cfg["kwargs"].copy())

    result = [{"optimizer": optimizer}]
    sched = _build_scheduler(optimizer, opt_cfg.get("lr_scheduler"), steps_per_epoch)
    if sched is not None:
        result[0]["lr_scheduler"] = sched
    return result


def _build_multi_optimizer(
    optim_settings, conv_params, linear_params, steps_per_epoch: int = 1
) -> list:
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

    lin_sched = _build_scheduler(lin_opt, lin_cfg.get("lr_scheduler"), steps_per_epoch)
    if lin_sched is not None:
        result[0]["lr_scheduler"] = lin_sched

    conv_sched = _build_scheduler(
        conv_opt, conv_cfg.get("lr_scheduler"), steps_per_epoch
    )
    if conv_sched is not None:
        result[1]["lr_scheduler"] = conv_sched

    return result
