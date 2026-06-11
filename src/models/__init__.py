"""Model package with lazy exports for optional audio dependencies."""

from __future__ import annotations

import importlib
from typing import Any

from .base import LipsModel
from .registry import get_model, register_model

__all__ = [
    "LipsModel",
    "get_model",
    "register_model",
    "BaseAudioModelWrapper",
    "get_audio_model",
    "list_audio_models",
    "register_audio_model",
]

_AUDIO_EXPORTS = {
    "BaseAudioModelWrapper",
    "get_audio_model",
    "list_audio_models",
    "register_audio_model",
}


def __getattr__(name: str) -> Any:
    if name in _AUDIO_EXPORTS:
        audio = importlib.import_module(".audio", __name__)
        return getattr(audio, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
