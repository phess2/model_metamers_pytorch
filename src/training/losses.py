"""
Loss function utilities.

Provides a factory ``get_loss_fn`` that returns a loss callable based on
the training config.  Defaults to ``nn.CrossEntropyLoss``.

Extend this module when adding multi-task losses (e.g. joint word + speaker
classification for audio).
"""

from __future__ import annotations

from torch import nn


def get_loss_fn(config: dict) -> nn.Module:
    """
    Build a loss function from the training config.

    Currently supports:
      - ``"cross_entropy"`` (default) -- standard ``nn.CrossEntropyLoss``

    The config may contain a ``loss`` section; if absent, cross-entropy is
    used.

    Example config snippet::

        {
            "loss": {
                "name": "cross_entropy",
                "kwargs": {"label_smoothing": 0.1}
            }
        }
    """
    loss_config = config.get("loss", {})
    name = loss_config.get("name", "cross_entropy")
    kwargs = loss_config.get("kwargs", {})

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: '{name}'")
