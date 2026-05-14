from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseAudioModelWrapper(nn.Module, ABC):
    """Shared interface for differentiable audio model wrappers.

    Wrappers should keep preprocessing differentiable whenever possible so
    metamer optimisation can backpropagate from representation space to the
    raw waveform.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def forward_waveform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Run the wrapped model from raw waveform input."""
        raise NotImplementedError

    @abstractmethod
    def forward_features(self, waveform: torch.Tensor, sr: int, layer=None) -> torch.Tensor:
        """Return intermediate features, optionally from a specific layer."""
        raise NotImplementedError

    @property
    @abstractmethod
    def available_layers(self) -> list[str]:
        """Names that can be requested via ``forward_features(..., layer=...)``."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Map raw waveform input to the representation consumed by ``self.model``."""
        raise NotImplementedError

    def freeze(self) -> None:
        """Freeze wrapper/model weights while preserving input gradients."""
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def get_classifier_logits(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Return classifier logits when the wrapped model exposes them."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose classifier logits."
        )
