"""Audio model registry for pretrained wrapper lookup."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .audiomae import (
    load_audiomae_as2m_audio_model,
    load_audiomae_as2m_ft_as20k_audio_model,
    load_audiomae_audio_model,
)
from .base import BaseAudioModelWrapper
from .beats import (
    load_beats_audio_model,
    load_beats_iter3_audio_model,
    load_beats_iter3_plus_as2m_audio_model,
)
from .clap import load_clap_audio_model
from .panns_cnn14 import load_panns_cnn14_audio_model

AudioModelFactory = Callable[..., BaseAudioModelWrapper]

_AUDIO_MODEL_REGISTRY: Dict[str, AudioModelFactory] = {}


def register_audio_model(name: str, factory: AudioModelFactory) -> None:
    """Register an audio model wrapper factory under ``name``."""
    _AUDIO_MODEL_REGISTRY[name.lower()] = factory


def list_audio_models() -> list[str]:
    """Return registered audio model names in sorted order."""
    return sorted(_AUDIO_MODEL_REGISTRY.keys())


def get_audio_model(
    name: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> BaseAudioModelWrapper:
    """Instantiate a registered audio model wrapper by name."""
    model_name = name.lower()
    if model_name not in _AUDIO_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown audio model '{name}'. Available: {list_audio_models()}"
        )
    factory = _AUDIO_MODEL_REGISTRY[model_name]
    return factory(checkpoint_path=checkpoint_path, device=device, **kwargs)


def _placeholder_factory(model_name: str) -> AudioModelFactory:
    def _factory(
        checkpoint_path: Optional[str] = None, device: str = "cuda", **kwargs
    ) -> BaseAudioModelWrapper:
        _ = checkpoint_path, device, kwargs
        raise NotImplementedError(
            f"Audio model '{model_name}' is registered but not implemented yet. "
            "Add a concrete wrapper and register its loader factory."
        )

    return _factory


def _register_builtins() -> None:
    for name in ("ssast", "maskspec"):
        register_audio_model(name, _placeholder_factory(name))
    register_audio_model("beats", load_beats_audio_model)
    register_audio_model("beats_iter3", load_beats_iter3_audio_model)
    register_audio_model("beats_iter3_plus_as2m", load_beats_iter3_plus_as2m_audio_model)
    register_audio_model("audiomae", load_audiomae_audio_model)
    register_audio_model("audiomae_as2m", load_audiomae_as2m_audio_model)
    register_audio_model(
        "audiomae_as2m_ft_as20k", load_audiomae_as2m_ft_as20k_audio_model
    )
    register_audio_model("clap", load_clap_audio_model)
    register_audio_model("panns_cnn14", load_panns_cnn14_audio_model)


_register_builtins()
