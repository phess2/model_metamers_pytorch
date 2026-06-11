from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .alexnet import alexnet_classifier
from .resnet import resnet50_classifier

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class _DatasetSpec:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    min_value: float
    max_value: float


_IMAGENET_DATASET_SPEC = _DatasetSpec(
    mean=_IMAGENET_MEAN,
    std=_IMAGENET_STD,
    min_value=0.0,
    max_value=1.0,
)


class InputNormalize(nn.Module):
    """Clamp + channel-wise normalization with checkpoint-compatible buffers."""

    def __init__(
        self,
        new_mean: tuple[float, ...],
        new_std: tuple[float, ...],
        min_value: float,
        max_value: float,
    ) -> None:
        super().__init__()
        mean_tensor = torch.tensor(new_mean, dtype=torch.float32)[..., None, None]
        std_tensor = torch.tensor(new_std, dtype=torch.float32)[..., None, None]
        self.register_buffer("new_mean", mean_tensor)
        self.register_buffer("new_std", std_tensor)
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.min_value, self.max_value)
        return (x - self.new_mean) / self.new_std


class GraphPreprocessing(nn.Module):
    """Visual-only subset of legacy preprocessing graph."""

    def __init__(self, dataset_spec: _DatasetSpec) -> None:
        super().__init__()
        self.normalize = InputNormalize(
            dataset_spec.mean,
            dataset_spec.std,
            dataset_spec.min_value,
            dataset_spec.max_value,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class _InternalAttacker(nn.Module):
    """Minimal compatibility shim to preserve old checkpoint key structure."""

    def __init__(self, model: nn.Module, dataset_spec: _DatasetSpec) -> None:
        super().__init__()
        self.preproc = GraphPreprocessing(dataset_spec)
        self.model = model


class InternalAttackerModel(nn.Module):
    """Inference-only replacement for robustness.AttackerModel."""

    def __init__(self, model: nn.Module, dataset_spec: _DatasetSpec) -> None:
        super().__init__()
        self.preproc = GraphPreprocessing(dataset_spec)
        self.model = model
        self.attacker = _InternalAttacker(model, dataset_spec)

    def forward(
        self,
        inp: torch.Tensor,
        target=None,
        make_adv: bool = False,
        with_latent: bool = False,
        fake_relu: bool = False,
        no_relu: bool = False,
        with_image: bool = True,
        **_: object,
    ):
        del target
        if make_adv:
            raise NotImplementedError(
                "Internal robust wrapper is inference-only and does not support attacks."
            )
        if with_image:
            preproc_inp = self.preproc(inp)
            output = self.model(
                preproc_inp,
                with_latent=with_latent,
                fake_relu=fake_relu,
                no_relu=no_relu,
            )
        else:
            output = None
        return output, inp


_CLASSIFIERS: dict[str, Callable[[], nn.Module]] = {
    "alexnet": alexnet_classifier,
    "resnet50": resnet50_classifier,
}


def build_internal_attacker_model(arch: str) -> InternalAttackerModel:
    if arch not in _CLASSIFIERS:
        raise ValueError(f"Unsupported robust architecture '{arch}'.")
    classifier = _CLASSIFIERS[arch]()
    return InternalAttackerModel(classifier, _IMAGENET_DATASET_SPEC)
