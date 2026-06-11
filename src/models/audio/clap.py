from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torchaudio
from torch import nn
from transformers import AutoProcessor, ClapAudioModelWithProjection

from .base import BaseAudioModelWrapper


class ClapAudioModelWrapper(BaseAudioModelWrapper):
    """Differentiable wrapper around Hugging Face CLAP audio embeddings."""

    def __init__(
        self,
        model: nn.Module,
        processor: Optional[Any] = None,
        target_sample_rate: int = 48_000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 480,
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: float = 14_000.0,
        log_eps: float = 1e-6,
        max_length_seconds: float = 10.0,
    ) -> None:
        super().__init__(model=model)
        self.processor = processor
        self.target_sample_rate = target_sample_rate
        self.log_eps = log_eps
        self.max_length_seconds = max_length_seconds
        self.max_samples = int(round(target_sample_rate * max_length_seconds))
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        config = getattr(self.model, "config", None)
        audio_config = getattr(config, "audio_config", config)
        self._enable_fusion = bool(getattr(audio_config, "enable_fusion", False))
        self._hidden_state_layer_names = self._build_hidden_state_layer_names()
        self._transformer_block_layer_names = self._build_transformer_block_layer_names()
        self._available_layers = [
            "input_features",
            *self._transformer_block_layer_names,
            *self._hidden_state_layer_names,
            "audio_hidden_states",
            "last_hidden_state",
            "pooler_output",
            "audio_embeds",
            "final_audio_embedding",
            "final",
        ]
        self.metamer_layers = [
            *self._transformer_block_layer_names,
            "pooler_output",
            "final_audio_embedding",
        ]
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
            normalized=False,
        )

    @property
    def available_layers(self) -> list[str]:
        return list(self._available_layers)

    def _build_hidden_state_layer_names(self) -> list[str]:
        layer_count = 0
        config = getattr(self.model, "config", None)
        audio_config = getattr(config, "audio_config", config)
        if audio_config is not None:
            layer_count = int(getattr(audio_config, "num_hidden_layers", 0))
        # Hidden state tuples include embedding + encoder blocks.
        return [f"hidden_state_{idx:02d}" for idx in range(layer_count + 1)]

    def _get_audio_encoder(self) -> Optional[nn.Module]:
        if hasattr(self.model, "audio_model"):
            return cast(Optional[nn.Module], getattr(self.model.audio_model, "audio_encoder", None))
        return cast(Optional[nn.Module], getattr(self.model, "audio_encoder", None))

    def _build_transformer_block_layer_names(self) -> list[str]:
        encoder = self._get_audio_encoder()
        block_count = 0
        if encoder is not None and hasattr(encoder, "layers"):
            encoder_layers = cast(list[Any], encoder.layers)
            for layer in encoder_layers:
                block_count += len(getattr(layer, "blocks", []))
        if block_count == 0:
            config = getattr(self.model, "config", None)
            audio_config = getattr(config, "audio_config", config)
            depths = getattr(audio_config, "depths", [])
            block_count = int(sum(depths)) if depths else 0
        return [f"transformer_block_{idx:02d}" for idx in range(block_count)]

    def _register_transformer_block_hooks(
        self,
    ) -> tuple[dict[str, torch.Tensor], list[Any]]:
        encoder = self._get_audio_encoder()
        if encoder is None or not hasattr(encoder, "layers"):
            return {}, []

        block_outputs: dict[str, torch.Tensor] = {}
        hook_handles = []
        block_idx = 0
        encoder_layers = cast(list[Any], encoder.layers)
        for layer in encoder_layers:
            for block in getattr(layer, "blocks", []):
                block_name = f"transformer_block_{block_idx:02d}"
                block_idx += 1

                def _capture_block_output(_module, _args, output, name=block_name):
                    if isinstance(output, tuple):
                        output = output[0]
                    if isinstance(output, torch.Tensor):
                        block_outputs[name] = output

                hook_handles.append(block.register_forward_hook(_capture_block_output))
        return block_outputs, hook_handles

    def _ensure_batch_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        if waveform.dim() == 2:
            return waveform
        if waveform.dim() == 3:
            # Accept [B, C, T] and collapse channels for CLAP mono input.
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
        resampler = self._resamplers[sr].to(waveform.device)
        return resampler(waveform)

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        waveform = self._ensure_batch_waveform(waveform)
        waveform = self._resample_if_needed(waveform, sr)
        waveform = self._pad_or_crop_waveform(waveform)
        melspec = self.melspec.to(waveform.device)(waveform)
        log_melspec = torch.log10(melspec + self.log_eps)
        return log_melspec

    def _pad_or_crop_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        sample_count = waveform.shape[-1]
        if sample_count > self.max_samples:
            return waveform[..., : self.max_samples]
        if sample_count == self.max_samples:
            return waveform
        # Deterministic repeat-pad to match CLAP feature extractor defaults.
        repeat_count = max(self.max_samples // sample_count, 1)
        repeated = waveform.repeat(1, repeat_count)
        if repeated.shape[-1] > self.max_samples:
            return repeated[..., : self.max_samples]
        if repeated.shape[-1] == self.max_samples:
            return repeated
        pad_amount = self.max_samples - repeated.shape[-1]
        return torch.nn.functional.pad(repeated, (0, pad_amount))

    def _prepare_model_inputs(self, waveform: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
        input_features = self.preprocess(waveform, sr=sr)
        stacked_features = input_features.transpose(1, 2).unsqueeze(1)
        if self._enable_fusion:
            stacked_features = stacked_features.repeat(1, 4, 1, 1)
        model_inputs = {
            "input_features": stacked_features
        }
        is_longer_flag = bool(self._enable_fusion)
        model_inputs["is_longer"] = torch.full(
            (model_inputs["input_features"].shape[0], 1),
            fill_value=is_longer_flag,
            dtype=torch.bool,
            device=model_inputs["input_features"].device,
        )
        return model_inputs

    def _extract_audio_representations(
        self, model_inputs: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        audio_embeds: Optional[torch.Tensor] = None
        audio_hidden_states: Optional[torch.Tensor] = None
        last_hidden_state: Optional[torch.Tensor] = None
        pooler_output: Optional[torch.Tensor] = None
        hidden_state_outputs: dict[str, torch.Tensor] = {}
        transformer_block_outputs: dict[str, torch.Tensor] = {}
        input_features = model_inputs["input_features"]
        block_hook_handles: list[Any] = []

        has_custom_forward = type(self.model).forward is not nn.Module.forward
        if has_custom_forward:
            transformer_block_outputs, block_hook_handles = (
                self._register_transformer_block_hooks()
            )
            try:
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_hidden_states=True,
                )
            except TypeError:
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                )
            finally:
                for handle in block_hook_handles:
                    handle.remove()

            if hasattr(outputs, "audio_embeds"):
                audio_embeds = outputs.audio_embeds
            elif hasattr(outputs, "pooler_output"):
                audio_embeds = outputs.pooler_output
                pooler_output = outputs.pooler_output
            elif isinstance(outputs, dict) and "audio_embeds" in outputs:
                audio_embeds = outputs["audio_embeds"]
            elif isinstance(outputs, dict) and "pooler_output" in outputs:
                audio_embeds = outputs["pooler_output"]
                pooler_output = outputs["pooler_output"]

            audio_model_output = getattr(outputs, "audio_model_output", None)
            if audio_model_output is not None:
                hidden_states = getattr(audio_model_output, "hidden_states", None)
                if hidden_states:
                    audio_hidden_states = hidden_states[-1]
                    for idx, hidden_state in enumerate(hidden_states):
                        hidden_state_outputs[f"hidden_state_{idx:02d}"] = hidden_state
                elif hasattr(audio_model_output, "last_hidden_state"):
                    audio_hidden_states = audio_model_output.last_hidden_state
                    last_hidden_state = audio_model_output.last_hidden_state

            if audio_hidden_states is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states:
                    audio_hidden_states = hidden_states[-1]
                    for idx, hidden_state in enumerate(hidden_states):
                        hidden_state_outputs[f"hidden_state_{idx:02d}"] = hidden_state
                elif isinstance(outputs, dict) and "hidden_states" in outputs:
                    raw_states = outputs["hidden_states"]
                    if raw_states:
                        audio_hidden_states = raw_states[-1]
                        for idx, hidden_state in enumerate(raw_states):
                            hidden_state_outputs[f"hidden_state_{idx:02d}"] = hidden_state
                elif not isinstance(outputs, dict) and hasattr(outputs, "last_hidden_state"):
                    audio_hidden_states = outputs.last_hidden_state
                    last_hidden_state = outputs.last_hidden_state
                elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
                    audio_hidden_states = outputs["last_hidden_state"]
                    last_hidden_state = outputs["last_hidden_state"]

            if (
                last_hidden_state is None
                and not isinstance(outputs, dict)
                and hasattr(outputs, "last_hidden_state")
            ):
                last_hidden_state = outputs.last_hidden_state
            if (
                pooler_output is None
                and not isinstance(outputs, dict)
                and hasattr(outputs, "pooler_output")
            ):
                pooler_output = outputs.pooler_output

        if audio_embeds is None and hasattr(self.model, "get_audio_features"):
            audio_embeds = cast(Any, self.model).get_audio_features(
                input_features=input_features
            )

        if audio_embeds is None:
            raise RuntimeError(
                f"{type(self.model).__name__} did not return `audio_embeds` for CLAP audio."
            )

        if audio_hidden_states is None:
            audio_hidden_states = audio_embeds.unsqueeze(1)

        if last_hidden_state is None:
            last_hidden_state = audio_hidden_states
        if pooler_output is None:
            pooler_output = audio_embeds

        layer_outputs = {}
        layer_outputs.update(transformer_block_outputs)
        layer_outputs.update(hidden_state_outputs)
        return (
            audio_embeds,
            audio_hidden_states,
            last_hidden_state,
            pooler_output,
            layer_outputs,
        )

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

    def forward_with_representations(
        self, waveform: torch.Tensor, sr: int = 48_000, fake_relu: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del fake_relu
        model_inputs = self._prepare_model_inputs(waveform, sr=sr)
        input_features = model_inputs["input_features"]
        (
            audio_embeds,
            audio_hidden_states,
            last_hidden_state,
            pooler_output,
            hidden_state_outputs,
        ) = self._extract_audio_representations(model_inputs=model_inputs)
        all_outputs = {
            "input_features": input_features,
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooler_output,
            "audio_hidden_states": audio_hidden_states,
            "audio_embeds": audio_embeds,
            "final_audio_embedding": audio_embeds,
            "final": audio_embeds,
        }
        all_outputs.update(hidden_state_outputs)
        return audio_embeds, all_outputs


def load_clap_audio_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    model_id: str = "laion/clap-htsat-fused",
    sample_rate: int = 48_000,
    freeze: bool = True,
    **from_pretrained_kwargs,
) -> ClapAudioModelWrapper:
    """Load the CLAP audio encoder wrapper from Hugging Face."""
    source = checkpoint_path or model_id

    clap_model = ClapAudioModelWithProjection.from_pretrained(
        source, **from_pretrained_kwargs
    )
    processor = AutoProcessor.from_pretrained(source)
    wrapper = ClapAudioModelWrapper(
        model=clap_model,
        processor=processor,
        target_sample_rate=sample_rate,
    )
    wrapper = wrapper.to(device)
    if freeze:
        wrapper.freeze()
    else:
        wrapper.eval()
    return wrapper
