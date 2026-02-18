"""
Core metamer generation module.

Provides ``MetamerGenerator`` for synthesising stimuli whose intermediate
neural representations match a reference at a target layer, and a helper
``load_model_from_checkpoint`` for loading trained models outside Lightning.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from src.analysis.regularizers import range_regularizer, total_variation_loss
from src.models.registry import get_model
from src.utils.config import load_config

# ImageNet normalisation constants (pixel-space [0,1] → model input space)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _normalize_imagenet(x: Tensor) -> Tensor:
    """Normalise a ``[0, 1]`` pixel-space tensor with ImageNet statistics."""
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    config_path: Union[str, Path],
    ckpt_path: Union[str, Path],
    device: str = "cuda",
) -> Tuple[torch.nn.Module, dict]:
    """
    Load a trained model from a config file and Lightning checkpoint.

    Handles the Lightning checkpoint format where ``state_dict`` keys are
    prefixed with ``"model."``.

    Args:
        config_path: Path to the model config (JSON / YAML).
        ckpt_path: Path to the Lightning ``.ckpt`` file.
        device: Target device.

    Returns:
        ``(model, config)`` with the model already on *device* and in
        ``eval()`` mode.
    """
    config = load_config(config_path)
    model = get_model(config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Strip the "model." prefix added by LipsLightningModule
    state_dict = {
        k.replace("model.", "", 1): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }

    # Remap keys missing "features." prefix (for older checkpoints)
    remapped_state_dict = {}
    for key, value in state_dict.items():
        if "." in key and not any(
            key.startswith(prefix)
            for prefix in ["features.", "classifier.", "fake_relu_dict", "avgpool"]
        ):
            parts = key.split(".", 1)
            if len(parts) == 2 and parts[0].isdigit():
                idx = int(parts[0])
                # LipsAlexNet features indices: 0, 3, 6, 8, 10
                if idx in [0, 3, 6, 8, 10]:
                    remapped_state_dict[f"features.{key}"] = value
                    continue
        remapped_state_dict[key] = value

    # Use strict=False to allow missing keys (e.g., classifier keys in old checkpoints)
    model.load_state_dict(remapped_state_dict, strict=False)
    model.to(device).eval()
    return model, config


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _normalized_l2_loss(current: Tensor, target: Tensor) -> Tensor:
    """``||current - target||_2 / ||target||_2``."""
    return torch.norm(current - target) / (torch.norm(target) + 1e-8)


def _l2_loss(current: Tensor, target: Tensor) -> Tensor:
    return torch.norm(current - target)


def _cosine_loss(current: Tensor, target: Tensor) -> Tensor:
    """1 - cosine_similarity (so that minimisation → matching)."""
    return 1.0 - torch.nn.functional.cosine_similarity(
        current.flatten(), target.flatten(), dim=0
    )


_LOSS_FNS = {
    "normalized_l2": _normalized_l2_loss,
    "l2": _l2_loss,
    "cosine": _cosine_loss,
}


# ---------------------------------------------------------------------------
# MetamerGenerator
# ---------------------------------------------------------------------------


class MetamerGenerator:
    """
    Synthesise a *metamer* — a noise image optimised so that its
    representation at a given model layer matches a target representation.

    Parameters
    ----------
    model : torch.nn.Module
        A ``LipsModel`` (or compatible) with ``forward_with_representations``.
    layer_name : str
        Key in the ``all_outputs`` dict returned by
        ``forward_with_representations`` to match against.
    lr : float
        Initial learning rate (decays by *lr_decay* each round).
    num_rounds : int
        Number of optimisation rounds.
    steps_per_round : int
        Gradient steps per round.
    lr_decay : float
        Multiplicative LR decay applied each round.
    fake_relu : bool
        If ``True``, use fake-ReLU for better gradient flow.
    loss_type : str
        One of ``"normalized_l2"``, ``"l2"``, ``"cosine"``.
    clamp_range : tuple[float, float]
        Range to clamp the metamer tensor to after each step.
    lambda_tv : float
        Weight for the total-variation (smoothness) regularizer.
        ``0`` disables it.  Typical values: 5e-6 (early layers),
        5e-5 (mid), 5e-4 (later layers).
    lambda_range : float
        Weight for the Lp range regularizer.  ``0`` disables it.
        Typical value: 0.005.
    range_norm_p : int
        Norm order for the range regularizer (default 6).
    normalize_fn : callable, optional
        Function that maps the raw optimisation variable to
        model-normalised space.  Regularizers are evaluated on the
        normalised signal.  Defaults to ImageNet normalisation;
        override for audio or other modalities.
    noise_scale : float
        Scale of the initial Gaussian noise (default 0.05).
    noise_mean : float
        Mean of the initial noise (default 0.5, i.e. centered on gray).
    device : str
        Device for optimisation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        lr: float = 1.0,
        num_rounds: int = 8,
        steps_per_round: int = 3000,
        lr_decay: float = 0.5,
        fake_relu: bool = True,
        loss_type: str = "normalized_l2",
        clamp_range: Tuple[float, float] = (0.0, 1.0),
        lambda_tv: float = 0.0,
        lambda_range: float = 0.0,
        range_norm_p: int = 6,
        normalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        noise_scale: float = 0.05,
        noise_mean: float = 0.5,
        device: str = "cuda",
    ):
        self.model = model
        self.layer_name = layer_name
        self.lr = lr
        self.num_rounds = num_rounds
        self.steps_per_round = steps_per_round
        self.lr_decay = lr_decay
        self.fake_relu = fake_relu
        self.loss_type = loss_type
        self.clamp_range = clamp_range
        self.lambda_tv = lambda_tv
        self.lambda_range = lambda_range
        self.range_norm_p = range_norm_p
        self.normalize_fn = (
            normalize_fn if normalize_fn is not None else _normalize_imagenet
        )
        self.noise_scale = noise_scale
        self.noise_mean = noise_mean
        self.device = device

        if loss_type not in _LOSS_FNS:
            raise ValueError(
                f"Unknown loss_type '{loss_type}'. Choose from {list(_LOSS_FNS)}"
            )
        self._loss_fn = _LOSS_FNS[loss_type]

    # -----------------------------------------------------------------
    # Target extraction
    # -----------------------------------------------------------------

    @torch.no_grad()
    def extract_target(self, x: Tensor) -> Tensor:
        """Forward *x* through the model and return the representation at
        ``self.layer_name``.

        *x* should be in **pixel space** ``[0, 1]``; normalisation
        is applied internally (via ``normalize_fn``) before the forward pass.
        """
        x = x.to(self.device)
        x_norm = self.normalize_fn(x)
        _logits, all_outputs = self.model.forward_with_representations(
            x_norm, fake_relu=self.fake_relu
        )
        if self.layer_name not in all_outputs:
            available = sorted(all_outputs.keys())
            raise KeyError(
                f"Layer '{self.layer_name}' not found. Available: {available}"
            )
        return all_outputs[self.layer_name].detach()

    # -----------------------------------------------------------------
    # Optimisation
    # -----------------------------------------------------------------

    def generate(
        self,
        target_rep: Tensor,
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Optimise a noise tensor in pixel space ``[0, 1]`` to match
        *target_rep* using normalised gradient descent.

        The metamer is normalised with ImageNet statistics before each
        forward pass, but the optimisation variable itself stays in
        pixel space.

        Parameters
        ----------
        target_rep : Tensor
            The target representation to match (from ``extract_target``).
        shape : tuple
            Shape of the metamer tensor to create (e.g. ``(1, 3, 224, 224)``).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        metamer : Tensor
            The optimised metamer image tensor in ``[0, 1]`` pixel space.
        metadata : dict
            Optimisation metadata including per-round losses and timing.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Initialise with small Gaussian perturbation centred on gray
        metamer = (
            torch.randn(shape, device=self.device) * self.noise_scale + self.noise_mean
        ).clamp(*self.clamp_range)
        metamer.requires_grad_(True)

        target_rep = target_rep.to(self.device).detach()

        loss_per_round = []
        t_start = time.time()

        for rnd in range(self.num_rounds):
            current_lr = self.lr * (self.lr_decay**rnd)

            round_losses = []
            for step in range(self.steps_per_round):
                # Normalise to model input space before forward pass
                metamer_norm = self.normalize_fn(metamer)
                _logits, all_outputs = self.model.forward_with_representations(
                    metamer_norm, fake_relu=self.fake_relu
                )
                current_rep = all_outputs[self.layer_name]
                loss = self._loss_fn(current_rep, target_rep)

                # Regularizers (computed on the normalised signal)
                if self.lambda_tv > 0:
                    loss = loss + self.lambda_tv * total_variation_loss(metamer_norm)
                if self.lambda_range > 0:
                    loss = loss + self.lambda_range * range_regularizer(
                        metamer_norm, p=self.range_norm_p
                    )

                # Normalised gradient step (L2Step from reference code)
                grad = torch.autograd.grad(loss, metamer)[0]
                with torch.no_grad():
                    g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(
                        -1, 1, 1, 1
                    )
                    # Descend: subtract normalised gradient * step size
                    metamer = metamer - (grad / (g_norm + 1e-10)) * current_lr
                    metamer.clamp_(*self.clamp_range)
                    metamer.requires_grad_(True)

                round_losses.append(loss.item())

            avg_loss = sum(round_losses) / len(round_losses)
            final_step_loss = round_losses[-1]
            loss_per_round.append(
                {
                    "round": rnd,
                    "lr": current_lr,
                    "avg_loss": avg_loss,
                    "final_step_loss": final_step_loss,
                }
            )
            print(
                f"  Round {rnd + 1}/{self.num_rounds}  "
                f"lr={current_lr:.6f}  "
                f"loss={final_step_loss:.6f}"
            )

        elapsed = time.time() - t_start

        metadata = {
            "loss_per_round": loss_per_round,
            "final_loss": loss_per_round[-1]["final_step_loss"],
            "optimization_params": {
                "optimizer": "normalized_gradient_descent",
                "lr": self.lr,
                "num_rounds": self.num_rounds,
                "steps_per_round": self.steps_per_round,
                "lr_decay": self.lr_decay,
                "fake_relu": self.fake_relu,
                "loss_type": self.loss_type,
                "clamp_range": list(self.clamp_range),
                "lambda_tv": self.lambda_tv,
                "lambda_range": self.lambda_range,
                "range_norm_p": self.range_norm_p,
                "noise_scale": self.noise_scale,
                "noise_mean": self.noise_mean,
            },
            "elapsed_seconds": elapsed,
        }

        return metamer.detach(), metadata
