"""
Model registry -- maps config ``model_name`` to a concrete ``LipsModel``
subclass and instantiates it from config ``hparams``.

To register a new model (e.g. an audio architecture), add it to
``_MODEL_REGISTRY`` below or call ``register_model`` at import time.
"""

from __future__ import annotations

import inspect
from typing import Dict, Type

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
            f"Unknown model '{model_name}'. Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    cls = _MODEL_REGISTRY[model_name]
    hparams = config.get("hparams", {})

    sig = inspect.signature(cls.__init__)
    accepted_kwargs = {
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    model_kwargs = {k: v for k, v in hparams.items() if k in accepted_kwargs}
    if "num_classes" in accepted_kwargs and "num_classes" not in model_kwargs:
        model_kwargs["num_classes"] = 1000

    return cls(**model_kwargs)


# ---------------------------------------------------------------------------
# Auto-register built-in models
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    from .vision.lipsalexnet import LipsAlexNet
    from .vision.lipsresnet import LipsResNet
    from .vision.robustvision import RobustAlexNet, RobustResNet50

    register_model("lipsalexnet", LipsAlexNet)
    register_model("lipsresnet", LipsResNet)
    register_model("robustalexnet", RobustAlexNet)
    register_model("robustresnet50", RobustResNet50)


_register_builtins()
