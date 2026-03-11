from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import dill
import torch

from ..base import LipsModel
from .robust import build_internal_attacker_model

_RESNET50_REMAP_KEYS = {
    "normalizer.new_mean": "preproc.normalize.new_mean",
    "normalizer.new_std": "preproc.normalize.new_std",
    "attacker.normalize.new_mean": "attacker.preproc.normalize.new_mean",
    "attacker.normalize.new_std": "attacker.preproc.normalize.new_std",
}

_ALEXNET_METAMER_LAYERS = [
    "input_after_preproc",
    "relu0_fake_relu",
    "relu1_fake_relu",
    "relu2_fake_relu",
    "relu3_fake_relu",
    "relu4_fake_relu",
    "fc0_relu_fake_relu",
    "fc1_relu_fake_relu",
    "final",
]

_RESNET50_METAMER_LAYERS = [
    "input_after_preproc",
    "conv1_relu1_fake_relu",
    "layer1_fake_relu",
    "layer2_fake_relu",
    "layer3_fake_relu",
    "layer4_fake_relu",
    "avgpool",
    "final",
]


@dataclass(frozen=True)
class RobustVariantSpec:
    arch: str
    checkpoint_name: str
    metamer_layers: list[str]
    remap_checkpoint_keys: dict[str, str]


_ROBUST_ALEXNET_VARIANTS: dict[str, RobustVariantSpec] = {
    "alexnet_l2_3_robust": RobustVariantSpec(
        arch="alexnet",
        checkpoint_name="alexnet_l2_3_robust_training.pt",
        metamer_layers=_ALEXNET_METAMER_LAYERS,
        remap_checkpoint_keys={},
    ),
    "alexnet_linf_8_robust": RobustVariantSpec(
        arch="alexnet",
        checkpoint_name="alexnet_linf_8_robust_training.pt",
        metamer_layers=_ALEXNET_METAMER_LAYERS,
        remap_checkpoint_keys={},
    ),
    "alexnet_random_l2_3_perturb": RobustVariantSpec(
        arch="alexnet",
        checkpoint_name="alexnet_l2_3_random_perturb.pt",
        metamer_layers=_ALEXNET_METAMER_LAYERS,
        remap_checkpoint_keys={},
    ),
    "alexnet_random_linf8_perturb": RobustVariantSpec(
        arch="alexnet",
        checkpoint_name="alexnet_random_linf_8_perturb.pt",
        metamer_layers=_ALEXNET_METAMER_LAYERS,
        remap_checkpoint_keys={},
    ),
}

_ROBUST_RESNET50_VARIANTS: dict[str, RobustVariantSpec] = {
    "resnet50_l2_3_robust": RobustVariantSpec(
        arch="resnet50",
        checkpoint_name="resnet50_imagenet_l2_3_0_robustness.pt",
        metamer_layers=_RESNET50_METAMER_LAYERS,
        remap_checkpoint_keys=_RESNET50_REMAP_KEYS,
    ),
    "resnet50_linf_4_robust": RobustVariantSpec(
        arch="resnet50",
        checkpoint_name="resnet50_imagenet_linf_4_0_robustness.pt",
        metamer_layers=_RESNET50_METAMER_LAYERS,
        remap_checkpoint_keys=_RESNET50_REMAP_KEYS,
    ),
    "resnet50_linf_8_robust": RobustVariantSpec(
        arch="resnet50",
        checkpoint_name="resnet50_imagenet_linf_8_0_robustness.pt",
        metamer_layers=_RESNET50_METAMER_LAYERS,
        remap_checkpoint_keys=_RESNET50_REMAP_KEYS,
    ),
    "resnet50_random_l2_perturb": RobustVariantSpec(
        arch="resnet50",
        checkpoint_name="resnet50_random_l2_perturb.pt",
        metamer_layers=_RESNET50_METAMER_LAYERS,
        remap_checkpoint_keys=_RESNET50_REMAP_KEYS,
    ),
    "resnet50_random_linf8_perturb": RobustVariantSpec(
        arch="resnet50",
        checkpoint_name="resnet50_random_linf8_perturb.pt",
        metamer_layers=_RESNET50_METAMER_LAYERS,
        remap_checkpoint_keys=_RESNET50_REMAP_KEYS,
    ),
}


class _BaseRobustVisionModel(LipsModel):
    """
    Adapter around legacy robust checkpoints.

    This model expects raw image tensors in pixel space `[0, 1]`.
    Normalization is handled internally by the wrapped robust model preproc.
    """

    available_variants: dict[str, RobustVariantSpec] = {}
    metamer_normalize_mode = "identity"

    def __init__(
        self,
        variant: str,
        checkpoint_root: str = "model_analysis_folders/visual_networks/pytorch_checkpoints",
        imagenet_path: str = "",
        strict_checkpoint_load: bool = True,
    ) -> None:
        super().__init__()
        if variant not in self.available_variants:
            raise ValueError(
                f"Unknown robust variant '{variant}'. "
                f"Available: {sorted(self.available_variants.keys())}"
            )
        self.variant = variant
        self._spec = self.available_variants[variant]
        self.checkpoint_root = checkpoint_root
        self.imagenet_path = imagenet_path
        self.strict_checkpoint_load = strict_checkpoint_load
        self.metamer_layers = list(self._spec.metamer_layers)
        self._wrapped_model: Optional[torch.nn.Module] = None

    @property
    def wrapped_model(self) -> Optional[torch.nn.Module]:
        return self._wrapped_model

    def _resolve_checkpoint_path(self, ckpt_path: Optional[str] = None) -> Path:
        if ckpt_path is not None:
            return Path(ckpt_path)
        return Path(self.checkpoint_root) / self._spec.checkpoint_name

    def _build_unloaded_model(self) -> torch.nn.Module:
        return build_internal_attacker_model(self._spec.arch)

    def load_checkpoint(self, ckpt_path: Optional[str], device: str = "cuda") -> None:
        checkpoint_path = self._resolve_checkpoint_path(ckpt_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = self._build_unloaded_model()
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=device,
            pickle_module=dill,
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        for old_key, new_key in self._spec.remap_checkpoint_keys.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

        model.load_state_dict(state_dict, strict=self.strict_checkpoint_load)
        self._wrapped_model = model.to(device).eval()

    def forward_with_representations(
        self, x: torch.Tensor, fake_relu: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._wrapped_model is None:
            raise RuntimeError(
                "Robust checkpoint is not loaded. Call load_checkpoint() first."
            )
        output, _inp = self._wrapped_model(
            x,
            with_latent=True,
            fake_relu=fake_relu,
            with_image=True,
        )
        logits = output[0]
        all_outputs = output[-1]
        return logits, all_outputs

    def forward(self, x: torch.Tensor, fake_relu: bool = False) -> torch.Tensor:
        logits, _ = self.forward_with_representations(x, fake_relu=fake_relu)
        return logits

    @torch.no_grad()
    def project_weights(self) -> dict[str, float]:
        # Legacy robust models are unconstrained; no projection to perform.
        return {}

    def get_lips_bound(self) -> float:
        # This family is unconstrained; exact/global bounds are not defined here.
        return float("nan")

    def extra_repr(self) -> str:
        return f"variant={self.variant}, checkpoint_root='{self.checkpoint_root}'"


class RobustAlexNet(_BaseRobustVisionModel):
    available_variants = _ROBUST_ALEXNET_VARIANTS


class RobustResNet50(_BaseRobustVisionModel):
    available_variants = _ROBUST_RESNET50_VARIANTS


def get_supported_robust_variants() -> dict[str, list[str]]:
    return {
        "robustalexnet": sorted(_ROBUST_ALEXNET_VARIANTS.keys()),
        "robustresnet50": sorted(_ROBUST_RESNET50_VARIANTS.keys()),
    }


def is_robust_vision_model(model: Any) -> bool:
    return isinstance(model, _BaseRobustVisionModel)
