"""
Hook-based activation extraction utility.

``ActivationExtractor`` captures intermediate activations from arbitrary
named layers of a model without modifying the model's ``forward`` method.
This is the foundation for future metamer generation and adversarial
analysis pipelines.

Example usage::

    model = LipsAlexNet(...)
    with ActivationExtractor(model, ["features.0", "classifier.1"]) as extractor:
        output = model(x)
        activations = extractor.activations  # {"features.0": ..., "classifier.1": ...}
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn


class ActivationExtractor:
    """
    Context manager that registers forward hooks on named layers and
    collects their output tensors.

    Parameters
    ----------
    model : nn.Module
        The model to instrument.
    layer_names : sequence of str
        Names of layers (as returned by ``model.named_modules()``) whose
        outputs should be captured.
    detach : bool
        If ``True`` (default), captured activations are detached from the
        computation graph.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Sequence[str],
        detach: bool = True,
    ):
        self.model = model
        self.layer_names = list(layer_names)
        self.detach = detach
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivationExtractor":
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        name_to_module = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in name_to_module:
                raise ValueError(
                    f"Layer '{name}' not found in model. "
                    f"Available: {sorted(name_to_module.keys())}"
                )
            hook = name_to_module[name].register_forward_hook(
                self._make_hook(name)
            )
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if self.detach:
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
                else:
                    self.activations[name] = output
            else:
                self.activations[name] = output
        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self) -> None:
        """Clear captured activations (hooks remain registered)."""
        self.activations.clear()
