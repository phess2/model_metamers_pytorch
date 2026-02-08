"""
Model registry -- maps config ``model_name`` to a concrete ``LipsModel``
subclass and instantiates it from config ``hparams``.

To register a new model (e.g. an audio architecture), add it to
``_MODEL_REGISTRY`` below or call ``register_model`` at import time.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

from .base import LipsModel

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Type[LipsModel]] = {}


def register_model(name: str, cls: Type[LipsModel]) -> None:
    """Register a ``LipsModel`` subclass under *name*."""
    _MODEL_REGISTRY[name] = cls


def get_model(config: dict) -> LipsModel:
    """
    Instantiate a model from a training config dict.

    The config must contain a top-level ``model_name`` key that maps to a
    registered model class, and a ``hparams`` dict whose keys are forwarded
    to the model constructor.
    """
    model_name = config["model_name"]
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    cls = _MODEL_REGISTRY[model_name]
    hparams = config.get("hparams", {})

    # Forward relevant hparams to the model constructor
    model_kwargs = {
        "num_classes": hparams.get("num_classes", 1000),
        "w_max": hparams.get("w_max", 1.0),
        "projection": hparams.get("projection", None),
    }

    # ResNet-specific hparams
    if "layer_sizes" in hparams:
        model_kwargs["layer_sizes"] = hparams["layer_sizes"]
    if "block_type" in hparams:
        model_kwargs["block_type"] = hparams["block_type"]
    if "zero_init_residual" in hparams:
        model_kwargs["zero_init_residual"] = hparams["zero_init_residual"]

    return cls(**model_kwargs)


# ---------------------------------------------------------------------------
# Auto-register built-in models
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from .vision.lipsalexnet import LipsAlexNet
    from .vision.lipsresnet import LipsResNet

    register_model("lipsalexnet", LipsAlexNet)
    register_model("lipsresnet", LipsResNet)


_register_builtins()
