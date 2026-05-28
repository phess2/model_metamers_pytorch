from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path
from typing import Optional, Protocol, Sequence, cast

import torch
import torchaudio
from torch import nn

from .base import BaseAudioModelWrapper

_DEFAULT_BEATS_CODE_DIR = Path(
    "/orcd/data/jhm/001/om2/rphess/pretrained_models/beats/code/beats"
)
_DEFAULT_BEATS_CHECKPOINT_DIR = Path(
    "/orcd/data/jhm/001/om2/rphess/pretrained_models/beats/checkpoints"
)

_DEFAULT_MODEL_CHECKPOINTS = {
    "beats_iter3": _DEFAULT_BEATS_CHECKPOINT_DIR / "BEATs_iter3.pt",
    "beats_iter3_plus_as2m": _DEFAULT_BEATS_CHECKPOINT_DIR
    / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt",
}
_DEFAULT_TOKENIZER_CHECKPOINTS = {
    "beats_iter3": _DEFAULT_BEATS_CHECKPOINT_DIR / "Tokenizer_iter3.pt",
    "beats_iter3_plus_as2m": _DEFAULT_BEATS_CHECKPOINT_DIR
    / "Tokenizer_iter3_plus_AS2M.pt",
}
_ALIAS_TO_VARIANT = {
    "beats": "beats_iter3",
    "beats_iter3": "beats_iter3",
    "beats_iter3_plus_as2m": "beats_iter3_plus_as2m",
}


@contextlib.contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    original_sys_path = list(sys.path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        sys.path = original_sys_path


def _load_beats_modules(code_dir: Path):
    with _prepend_sys_path(code_dir):
        beats_module = importlib.import_module("BEATs")
        tokenizers_module = importlib.import_module("Tokenizers")
    required_beats_symbols = ("BEATs", "BEATsConfig")
    required_tokenizer_symbols = ("Tokenizers", "TokenizersConfig")
    missing_beats = [
        name for name in required_beats_symbols if not hasattr(beats_module, name)
    ]
    if missing_beats:
        raise RuntimeError(
            "BEATs code module is missing required symbols: "
            + ", ".join(sorted(missing_beats))
        )
    missing_tokenizer = [
        name
        for name in required_tokenizer_symbols
        if not hasattr(tokenizers_module, name)
    ]
    if missing_tokenizer:
        raise RuntimeError(
            "Tokenizers code module is missing required symbols: "
            + ", ".join(sorted(missing_tokenizer))
        )
    return (
        beats_module.BEATs,
        beats_module.BEATsConfig,
        tokenizers_module.Tokenizers,
        tokenizers_module.TokenizersConfig,
    )


def _resolve_variant_name(variant: str) -> str:
    key = variant.lower()
    if key not in _ALIAS_TO_VARIANT:
        raise ValueError(
            f"Unknown BEATs variant '{variant}'. Available: {sorted(_ALIAS_TO_VARIANT.keys())}"
        )
    return _ALIAS_TO_VARIANT[key]


def _resolve_checkpoint_paths(
    variant: str,
    checkpoint_path: Optional[str],
    tokenizer_checkpoint_path: Optional[str],
) -> tuple[str, str]:
    resolved_checkpoint = checkpoint_path or str(_DEFAULT_MODEL_CHECKPOINTS[variant])
    resolved_tokenizer = tokenizer_checkpoint_path or str(
        _DEFAULT_TOKENIZER_CHECKPOINTS[variant]
    )
    return resolved_checkpoint, resolved_tokenizer


class _BeatsEncoderLike(Protocol):
    layers: Sequence[nn.Module]

    def __call__(
        self,
        x: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor],
        layer: Optional[int],
    ) -> tuple[torch.Tensor, object]: ...


class _BeatsModelLike(Protocol):
    encoder: _BeatsEncoderLike
    patch_embedding: nn.Module
    layer_norm: nn.Module
    post_extract_proj: Optional[nn.Module]
    dropout_input: nn.Module
    predictor_dropout: nn.Module
    predictor: Optional[nn.Module]

    def preprocess(
        self, waveform: torch.Tensor, *, fbank_mean: float, fbank_std: float
    ) -> torch.Tensor: ...

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor: ...


class BeatsAudioModelWrapper(BaseAudioModelWrapper):
    """Differentiable wrapper around local UniLM BEATs checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: nn.Module,
        classifier_label_mids: Optional[Sequence[str]] = None,
        target_sample_rate: int = 16_000,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> None:
        super().__init__(model=model)
        self._beats_model = cast(_BeatsModelLike, model)
        self.tokenizer = tokenizer
        self.classifier_label_mids = (
            [str(mid) for mid in classifier_label_mids]
            if classifier_label_mids is not None
            else None
        )
        self.target_sample_rate = target_sample_rate
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self._transformer_layer_names = self._build_transformer_layer_names()
        self._available_layers = [
            "input_features",
            "patch_features",
            "projected_features",
            *self._transformer_layer_names,
            "final_pooled_embedding",
            "final",
        ]
        if self._has_predictor_head:
            self._available_layers.extend(["logits", "probs"])
        self.metamer_layers = [
            "projected_features",
            *self._transformer_layer_names,
            "final_pooled_embedding",
        ]

    @property
    def _has_predictor_head(self) -> bool:
        return self._beats_model.predictor is not None

    def _build_transformer_layer_names(self) -> list[str]:
        layer_count = len(self._beats_model.encoder.layers)
        return [f"transformer_layer_{idx:02d}" for idx in range(layer_count)]

    @property
    def available_layers(self) -> list[str]:
        return list(self._available_layers)

    def _ensure_batch_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        if waveform.dim() == 2:
            return waveform
        if waveform.dim() == 3:
            return waveform.mean(dim=1)
        raise ValueError(
            "Expected waveform with shape [T], [B, T], or [B, C, T]; "
            f"got {tuple(waveform.shape)}."
        )

    def _resample_if_needed(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sample_rate:
            return waveform
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sample_rate
            )
        return self._resamplers[sr].to(waveform.device)(waveform)

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        waveform = self._ensure_batch_waveform(waveform)
        waveform = self._resample_if_needed(waveform, sr)
        return self._beats_model.preprocess(
            waveform, fbank_mean=self.fbank_mean, fbank_std=self.fbank_std
        )

    def _collect_encoder_outputs(
        self, encoder_input: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        layer_outputs: dict[str, torch.Tensor] = {}
        hook_handles = []
        for idx, layer in enumerate(self._beats_model.encoder.layers):
            layer_name = f"transformer_layer_{idx:02d}"

            def _capture_layer_output(_module, _args, output, name=layer_name):
                layer_outputs[name] = output[0].transpose(0, 1)

            hook_handles.append(layer.register_forward_hook(_capture_layer_output))

        try:
            final_features, _ = self._beats_model.encoder(
                encoder_input, padding_mask=padding_mask, layer=None
            )
        finally:
            for handle in hook_handles:
                handle.remove()

        return final_features, layer_outputs

    def _compute_logits(
        self, final_features: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        predictor = self._beats_model.predictor
        if predictor is None:
            raise RuntimeError("BEATs predictor head is not available on this checkpoint.")
        logits = self._beats_model.predictor_dropout(final_features)
        logits = predictor(logits)
        if padding_mask is not None and padding_mask.any():
            logits = logits.clone()
            logits[padding_mask] = 0
            token_counts = (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            logits = logits.sum(dim=1) / token_counts
        else:
            logits = logits.mean(dim=1)
        return logits

    def forward_waveform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        return self.forward_features(waveform=waveform, sr=sr, layer="final")

    def forward_features(
        self, waveform: torch.Tensor, sr: int, layer: Optional[str] = None
    ) -> torch.Tensor:
        _, representations = self.forward_with_representations(waveform, sr=sr)
        selected_layer = layer or "final"
        if selected_layer not in representations:
            raise ValueError(
                f"Unknown layer '{selected_layer}'. Available: {self.available_layers}"
            )
        return representations[selected_layer]

    def get_classifier_logits(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        _, representations = self.forward_with_representations(waveform=waveform, sr=sr)
        if "logits" not in representations:
            raise NotImplementedError(
                f"{type(self).__name__} does not expose classifier logits."
            )
        return representations["logits"]

    def forward_with_representations(
        self, waveform: torch.Tensor, sr: int = 16_000, fake_relu: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del fake_relu
        input_features = self.preprocess(waveform=waveform, sr=sr)
        padding_mask = None

        fbank = input_features.unsqueeze(1)
        patch_features = self._beats_model.patch_embedding(fbank)
        patch_features = patch_features.reshape(
            patch_features.shape[0], patch_features.shape[1], -1
        ).transpose(1, 2)
        patch_features = self._beats_model.layer_norm(patch_features)
        if padding_mask is not None:
            padding_mask = self._beats_model.forward_padding_mask(patch_features, padding_mask)

        projected_features = (
            self._beats_model.post_extract_proj(patch_features)
            if self._beats_model.post_extract_proj is not None
            else patch_features
        )
        encoder_input = self._beats_model.dropout_input(projected_features)
        final_features, layer_outputs = self._collect_encoder_outputs(
            encoder_input=encoder_input, padding_mask=padding_mask
        )
        final_pooled_embedding = final_features.mean(dim=1)

        all_outputs: dict[str, torch.Tensor] = {
            "input_features": input_features,
            "patch_features": patch_features,
            "projected_features": projected_features,
            "final_pooled_embedding": final_pooled_embedding,
            "final": final_features,
        }
        for layer_name in self._transformer_layer_names:
            if layer_name in layer_outputs:
                all_outputs[layer_name] = layer_outputs[layer_name]

        if self._has_predictor_head:
            logits = self._compute_logits(
                final_features=final_features, padding_mask=padding_mask
            )
            all_outputs["logits"] = logits
            all_outputs["probs"] = torch.sigmoid(logits)

        return final_features, all_outputs


def load_beats_audio_model(
    checkpoint_path: Optional[str] = None,
    tokenizer_checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    freeze: bool = True,
    variant: str = "beats",
    code_dir: Optional[str] = None,
    **torch_load_kwargs,
) -> BeatsAudioModelWrapper:
    """Load BEATs + tokenizer checkpoints from local UniLM implementation."""
    resolved_variant = _resolve_variant_name(variant)
    resolved_checkpoint_path, resolved_tokenizer_checkpoint_path = (
        _resolve_checkpoint_paths(
            variant=resolved_variant,
            checkpoint_path=checkpoint_path,
            tokenizer_checkpoint_path=tokenizer_checkpoint_path,
        )
    )
    beats_code_dir = Path(code_dir) if code_dir else _DEFAULT_BEATS_CODE_DIR

    try:
        beats_cls, beats_config_cls, tokenizer_cls, tokenizer_config_cls = (
            _load_beats_modules(beats_code_dir)
        )
    except ImportError as exc:
        raise ImportError(
            "Loading audio model 'beats' requires the local UniLM BEATs code to be importable "
            f"from '{beats_code_dir}'."
        ) from exc

    checkpoint = torch.load(
        resolved_checkpoint_path, map_location=device, **torch_load_kwargs
    )
    beats_model = beats_cls(beats_config_cls(checkpoint["cfg"]))
    beats_model.load_state_dict(checkpoint["model"])
    label_dict = checkpoint.get("label_dict")
    classifier_label_mids = None
    if isinstance(label_dict, dict):
        classifier_label_mids = [
            str(mid) for _idx, mid in sorted(label_dict.items(), key=lambda item: int(item[0]))
        ]

    tokenizer_checkpoint = torch.load(
        resolved_tokenizer_checkpoint_path, map_location=device, **torch_load_kwargs
    )
    tokenizer_model = tokenizer_cls(tokenizer_config_cls(tokenizer_checkpoint["cfg"]))
    tokenizer_model.load_state_dict(tokenizer_checkpoint["model"])
    tokenizer_model.eval()

    wrapper = BeatsAudioModelWrapper(
        model=beats_model,
        tokenizer=tokenizer_model,
        classifier_label_mids=classifier_label_mids,
    ).to(device)
    if freeze:
        wrapper.freeze()
    else:
        wrapper.eval()
    return wrapper


def load_beats_iter3_audio_model(
    checkpoint_path: Optional[str] = None,
    tokenizer_checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    freeze: bool = True,
    code_dir: Optional[str] = None,
    **torch_load_kwargs,
) -> BeatsAudioModelWrapper:
    return load_beats_audio_model(
        checkpoint_path=checkpoint_path,
        tokenizer_checkpoint_path=tokenizer_checkpoint_path,
        device=device,
        freeze=freeze,
        variant="beats_iter3",
        code_dir=code_dir,
        **torch_load_kwargs,
    )


def load_beats_iter3_plus_as2m_audio_model(
    checkpoint_path: Optional[str] = None,
    tokenizer_checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    freeze: bool = True,
    code_dir: Optional[str] = None,
    **torch_load_kwargs,
) -> BeatsAudioModelWrapper:
    return load_beats_audio_model(
        checkpoint_path=checkpoint_path,
        tokenizer_checkpoint_path=tokenizer_checkpoint_path,
        device=device,
        freeze=freeze,
        variant="beats_iter3_plus_as2m",
        code_dir=code_dir,
        **torch_load_kwargs,
    )
