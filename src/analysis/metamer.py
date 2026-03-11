"""
Core metamer generation module.

Provides ``MetamerGenerator`` for synthesising stimuli whose intermediate
neural representations match a reference at a target layer, and a helper
``load_model_from_checkpoint`` for loading trained models outside Lightning.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from src.analysis.regularizers import range_regularizer, total_variation_loss
from src.data.datasets import ImageNetFolder, get_vision_dataset
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


def _normalize_identity(x: Tensor) -> Tensor:
    """Return input unchanged."""
    return x


def resolve_model_normalize_fn(
    model: torch.nn.Module, config: Optional[dict] = None
) -> Callable[[Tensor], Tensor]:
    """
    Resolve the pixel->model normalization function for metamer generation.

    Priority:
      1) config["metamer_normalization"] if set
      2) model.metamer_normalize_mode if set
      3) default ImageNet normalization
    """
    config = config or {}
    mode = config.get("metamer_normalization") or getattr(
        model, "metamer_normalize_mode", "imagenet"
    )
    if mode == "identity":
        return _normalize_identity
    if mode == "imagenet":
        return _normalize_imagenet
    raise ValueError(
        f"Unknown metamer normalization mode '{mode}'. Expected one of "
        "['identity', 'imagenet']."
    )


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    config_path: Union[str, Path],
    ckpt_path: Union[str, Path],
    device: str = "cuda",
) -> Tuple[torch.nn.Module, dict]:
    """
    Load a trained model from a config file and checkpoint.

    Supports:
      - Lightning ``.ckpt`` files for Lipschitz models
      - robustness ``.pt`` files for legacy adversarially-trained models

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
    checkpoint_format = config.get("checkpoint_format", "lightning")

    if checkpoint_format == "robustness":
        if not hasattr(model, "load_checkpoint"):
            raise TypeError(
                "Config requests checkpoint_format='robustness', but model "
                f"{type(model).__name__} does not implement load_checkpoint()."
            )
        model.load_checkpoint(str(ckpt_path), device=device)
        model.to(device).eval()
        return model, config

    if checkpoint_format != "lightning":
        raise ValueError(
            f"Unknown checkpoint_format '{checkpoint_format}'. "
            "Expected one of ['lightning', 'robustness']."
        )

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


def derive_experiment_root(
    config_path: Union[str, Path],
    exp_dir: Union[str, Path] = "experiments",
    config_root: Union[str, Path] = "configs",
) -> Path:
    """Build experiment root path using config-relative structure when possible."""
    config_path_obj = Path(config_path).resolve()
    exp_dir_obj = Path(exp_dir)
    config_root_obj = Path(config_root).resolve()
    try:
        rel = config_path_obj.relative_to(config_root_obj)
        return exp_dir_obj / rel.with_suffix("")
    except ValueError:
        return exp_dir_obj / config_path_obj.stem


def load_eval_dataset_with_subset(
    *,
    config: dict,
    dataset_name: str,
    split: str,
    data_dir: Optional[str],
    imagenet_subset: str,
) -> Tuple[ImageNetFolder, list[int], bool]:
    """
    Load evaluation dataset in raw pixel space with optional ImageNet subset.

    Returns:
        dataset: full validation dataset wrapper
        selected_subset_indices: ordered dataset indices when subset is enabled
        use_subset_indices: whether caller should interpret sample indices in subset order
    """
    data_settings = config.get("data_settings", {})
    resolved_data_dir = data_dir or data_settings.get("data_dir")
    if resolved_data_dir is None:
        raise ValueError("data_dir is required via CLI or config['data_settings']['data_dir'].")
    image_size = data_settings.get("image_size", 224)

    if imagenet_subset != "none":
        dataset, selected_subset_indices = cast(
            tuple[ImageNetFolder, list[int]],
            get_vision_dataset(
                dataset_name,
                resolved_data_dir,
                image_size,
                stage="validate",
                raw=True,
                imagenet_subset=imagenet_subset,
                return_imagenet_subset_indices=True,
            ),
        )
    else:
        dataset = cast(
            ImageNetFolder,
            get_vision_dataset(
                dataset_name,
                resolved_data_dir,
                image_size,
                stage="validate",
                raw=True,
            ),
        )
        selected_subset_indices = []

    use_subset_indices = imagenet_subset != "none"
    if use_subset_indices:
        if dataset_name != "imagenet":
            raise ValueError("--imagenet_subset is only supported with --dataset imagenet.")
        if split != "val":
            raise ValueError("--imagenet_subset imagenet_400_val requires --split val.")
        if imagenet_subset == "imagenet_400_val" and len(selected_subset_indices) != 400:
            raise ValueError(
                "Expected exactly 400 images in imagenet_400_val subset, "
                f"got {len(selected_subset_indices)}."
            )
    return dataset, selected_subset_indices, use_subset_indices


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


class L2AdversarialAttacker:
    """Untargeted projected-gradient attacker with L2 constraints in pixel space."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalize_fn: Callable[[Tensor], Tensor],
        epsilon: float,
        step_size: float,
        num_steps: int,
        clamp_range: Tuple[float, float] = (0.0, 1.0),
        device: str = "cuda",
    ):
        self.model = model
        self.normalize_fn = normalize_fn
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.clamp_range = clamp_range
        self.device = device

    def _project_l2_ball(self, delta: Tensor) -> Tensor:
        flat = delta.view(delta.shape[0], -1)
        norms = flat.norm(p=2, dim=1, keepdim=True)
        scale = torch.clamp(self.epsilon / (norms + 1e-12), max=1.0)
        return (flat * scale).view_as(delta)

    def attack(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Create adversarial examples for (x, y) with untargeted L2 PGD.

        x is expected in pixel space [0, 1].
        """
        x = x.to(self.device)
        y = y.to(self.device)
        delta = torch.zeros_like(x, device=self.device)

        for _ in range(self.num_steps):
            adv = (x + delta).clamp(*self.clamp_range).detach().requires_grad_(True)
            adv_norm = self.normalize_fn(adv)
            logits = self.model(adv_norm)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, adv)[0]

            grad_flat = grad.view(grad.shape[0], -1)
            grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
            normalized_grad = grad / (grad_norm + 1e-12)

            with torch.no_grad():
                delta = delta + self.step_size * normalized_grad
                delta = self._project_l2_ball(delta)
                adv = (x + delta).clamp(*self.clamp_range)
                delta = (adv - x).detach()

        return (x + delta).clamp(*self.clamp_range).detach()


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
