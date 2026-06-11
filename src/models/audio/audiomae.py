from __future__ import annotations

from typing import Any, Optional, cast

import timm
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torchaudio.compliance import kaldi

from .base import BaseAudioModelWrapper

_AUDIOMAE_AS2M_MODEL_ID = "gaunernst/vit_base_patch16_1024_128.audiomae_as2m"
_AUDIOMAE_AS2M_FT_AS20K_MODEL_ID = (
    "gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k"
)


class AudioMaeAudioModelWrapper(BaseAudioModelWrapper):
    """Differentiable wrapper around Hugging Face AudioMAE checkpoints.

    Supports both AudioSet-2M self-supervised pretraining and the AudioSet-20k
    fine-tuned checkpoint:
    - `gaunernst/vit_base_patch16_1024_128.audiomae_as2m`
    - `gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k`

    The fine-tuned checkpoint typically exposes classifier logits, while the
    pretraining checkpoint is commonly used as an embedding model.
    """

    def __init__(
        self,
        model: nn.Module,
        target_sample_rate: int = 16_000,
        num_mel_bins: int = 128,
        target_frames: int = 1024,
        mean: float = -4.2677393,
        std: float = 4.5689974,
        patch_time_stride: int = 16,
        patch_mel_stride: int = 16,
    ) -> None:
        super().__init__(model=model)
        self.target_sample_rate = target_sample_rate
        self.num_mel_bins = num_mel_bins
        self.target_frames = target_frames
        self.mean = mean
        self.std = std
        self.patch_time_stride = patch_time_stride
        self.patch_mel_stride = patch_mel_stride
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self._transformer_layer_names = self._build_transformer_layer_names()
        self._has_classifier_head = self._detect_classifier_head()
        self._available_layers = [
            "input_features",
            "tokens",
            "cls_token",
            "patch_tokens",
            "vit_patch_embeddings",
            "frame_features",
            *self._transformer_layer_names,
            "final_pooled_embedding",
            "final",
        ]
        if self._has_classifier_head:
            self._available_layers.append("logits")
        self.metamer_layers = [
            "vit_patch_embeddings",
            *self._transformer_layer_names,
            "final_pooled_embedding",
        ]

    @property
    def available_layers(self) -> list[str]:
        return list(self._available_layers)

    def _detect_classifier_head(self) -> bool:
        if not hasattr(self.model, "get_classifier"):
            return False
        classifier = cast(Any, self.model).get_classifier()
        if classifier is None:
            return False
        if isinstance(classifier, nn.Identity):
            return False
        return True

    def _build_transformer_layer_names(self) -> list[str]:
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            return []
        return [f"transformer_block_{idx:02d}" for idx in range(len(blocks))]

    def _ensure_batch_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        if waveform.dim() == 2:
            return waveform
        if waveform.dim() == 3:
            # AudioMAE consumes mono waveforms in [B, T] shape.
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

    def _compute_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        fbank_features = []
        for sample in waveform:
            sample_features = kaldi.fbank(
                sample.unsqueeze(0),
                sample_frequency=self.target_sample_rate,
                htk_compat=True,
                window_type="hanning",
                num_mel_bins=self.num_mel_bins,
            )
            frame_count = sample_features.shape[0]
            if frame_count < self.target_frames:
                sample_features = F.pad(
                    sample_features, (0, 0, 0, self.target_frames - frame_count)
                )
            else:
                sample_features = sample_features[: self.target_frames]
            fbank_features.append(sample_features)
        return torch.stack(fbank_features, dim=0)

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        waveform = self._ensure_batch_waveform(waveform)
        waveform = self._resample_if_needed(waveform, sr)
        fbank_features = self._compute_fbank(waveform)
        normalized = (fbank_features - self.mean) / (self.std * 2.0)
        return normalized.unsqueeze(1)

    def _extract_tokens_and_transformer_outputs(
        self, input_features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], Optional[torch.Tensor]]:
        layer_outputs: dict[str, torch.Tensor] = {}
        patch_embedding_tokens: Optional[torch.Tensor] = None

        blocks = getattr(self.model, "blocks", None)
        if blocks is None or len(blocks) == 0:
            return (
                cast(Any, self.model).forward_features(input_features),
                layer_outputs,
                patch_embedding_tokens,
            )

        hook_handles = []

        def _capture_first_block_input(_module, args):
            nonlocal patch_embedding_tokens
            if args and isinstance(args[0], torch.Tensor):
                patch_embedding_tokens = args[0]

        hook_handles.append(
            blocks[0].register_forward_pre_hook(_capture_first_block_input)
        )

        for idx, block in enumerate(blocks):
            layer_name = f"transformer_block_{idx:02d}"

            def _capture_block_output(_module, _args, output, name=layer_name):
                if isinstance(output, tuple):
                    output = output[0]
                if isinstance(output, torch.Tensor):
                    layer_outputs[name] = output

            hook_handles.append(block.register_forward_hook(_capture_block_output))

        try:
            tokens = cast(Any, self.model).forward_features(input_features)
        finally:
            for handle in hook_handles:
                handle.remove()

        return tokens, layer_outputs, patch_embedding_tokens

    def _extract_frame_features(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        expected_time_patches = self.target_frames // self.patch_time_stride
        expected_mel_patches = self.num_mel_bins // self.patch_mel_stride
        expected_patches = expected_time_patches * expected_mel_patches
        if patch_tokens.dim() != 3 or patch_tokens.shape[1] != expected_patches:
            return patch_tokens
        frame_grid = patch_tokens.unflatten(
            1, (expected_time_patches, expected_mel_patches)
        )
        return frame_grid.mean(dim=2)

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
        _, representations = self.forward_with_representations(waveform, sr=sr)
        if "logits" not in representations:
            raise NotImplementedError(
                f"{type(self).__name__} does not expose classifier logits."
            )
        return representations["logits"]

    def forward_with_representations(
        self, waveform: torch.Tensor, sr: int = 16_000, fake_relu: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del fake_relu
        input_features = self.preprocess(waveform, sr=sr)
        tokens, layer_outputs, patch_embedding_tokens = (
            self._extract_tokens_and_transformer_outputs(input_features)
        )
        if tokens.dim() != 3 or tokens.shape[1] <= 1:
            raise RuntimeError(
                f"Expected ViT token tensor [B, N, D], got {tuple(tokens.shape)}."
            )

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        if (
            patch_embedding_tokens is not None
            and patch_embedding_tokens.dim() == 3
            and patch_embedding_tokens.shape[1] > 1
        ):
            vit_patch_embeddings = patch_embedding_tokens[:, 1:]
        else:
            vit_patch_embeddings = patch_tokens
        frame_features = self._extract_frame_features(patch_tokens)
        final_pooled_embedding = cls_token
        final = self.model(input_features)
        all_outputs = {
            "input_features": input_features,
            "tokens": tokens,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "vit_patch_embeddings": vit_patch_embeddings,
            "frame_features": frame_features,
            "final_pooled_embedding": final_pooled_embedding,
            "final": final,
        }
        for layer_name in self._transformer_layer_names:
            if layer_name in layer_outputs:
                layer_features = layer_outputs[layer_name]
                if layer_features.dim() == 3 and layer_features.shape[1] > 1:
                    all_outputs[layer_name] = layer_features[:, 1:]
                else:
                    all_outputs[layer_name] = layer_features
        if self._has_classifier_head:
            all_outputs["logits"] = final
        return final, all_outputs


def load_audiomae_audio_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    model_id: str = _AUDIOMAE_AS2M_FT_AS20K_MODEL_ID,
    freeze: bool = True,
    pretrained: bool = True,
    **create_model_kwargs,
) -> AudioMaeAudioModelWrapper:
    """Load AudioMAE wrapper from Hugging Face via `timm`."""
    source = checkpoint_path or model_id
    if not source.startswith("hf_hub:"):
        source = f"hf_hub:{source}"

    audiomae_model = timm.create_model(
        source, pretrained=pretrained, **create_model_kwargs
    )
    wrapper = AudioMaeAudioModelWrapper(model=audiomae_model)
    wrapper = wrapper.to(device)
    if freeze:
        wrapper.freeze()
    else:
        wrapper.eval()
    return wrapper


def load_audiomae_as2m_audio_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    freeze: bool = True,
    pretrained: bool = True,
    **create_model_kwargs,
) -> AudioMaeAudioModelWrapper:
    """Load AudioMAE AudioSet-2M checkpoint (no classifier head by default)."""
    return load_audiomae_audio_model(
        checkpoint_path=checkpoint_path,
        device=device,
        model_id=_AUDIOMAE_AS2M_MODEL_ID,
        freeze=freeze,
        pretrained=pretrained,
        **create_model_kwargs,
    )


def load_audiomae_as2m_ft_as20k_audio_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    freeze: bool = True,
    pretrained: bool = True,
    **create_model_kwargs,
) -> AudioMaeAudioModelWrapper:
    """Load AudioMAE AudioSet-20k fine-tuned checkpoint."""
    return load_audiomae_audio_model(
        checkpoint_path=checkpoint_path,
        device=device,
        model_id=_AUDIOMAE_AS2M_FT_AS20K_MODEL_ID,
        freeze=freeze,
        pretrained=pretrained,
        **create_model_kwargs,
    )
