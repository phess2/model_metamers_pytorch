from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

from .base import BaseAudioModelWrapper

_DEFAULT_PANNS_CODE_ROOT = Path(
    "/orcd/data/jhm/001/om2/rphess/pretrained_models/panns/code/audioset_tagging_cnn"
)
_DEFAULT_HF_REPO = "nicofarr/panns_Cnn14"
_DEFAULT_HF_FILENAME = "model.safetensors"


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


def _load_panns_cnn14_class(code_root: Path):
    pytorch_dir = code_root / "pytorch"
    model_file = pytorch_dir / "models.py"
    if not model_file.exists():
        raise FileNotFoundError(
            f"PANNs models.py not found at '{model_file}'. "
            "Clone qiuqiangkong/audioset_tagging_cnn and point code_dir there."
        )

    module_name = "panns_audioset_models"
    spec = importlib.util.spec_from_file_location(module_name, str(model_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for '{model_file}'.")

    module = importlib.util.module_from_spec(spec)
    with _prepend_sys_path(pytorch_dir):
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == "torchlibrosa":
                raise ModuleNotFoundError(
                    "Loading PANNs Cnn14 requires 'torchlibrosa'. "
                    "Install torchlibrosa in the project's uv environment."
                ) from exc
            raise

    if not hasattr(module, "Cnn14"):
        raise RuntimeError(
            "PANNs models.py is missing symbol 'Cnn14'. "
            f"Loaded from '{model_file}'."
        )
    return module.Cnn14


def _load_panns_state_dict(
    checkpoint_path: Optional[str],
    model_id: str,
    model_filename: str,
) -> dict[str, torch.Tensor]:
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"PANNs checkpoint not found: '{checkpoint_file}'.")
    else:
        checkpoint_file = Path(hf_hub_download(repo_id=model_id, filename=model_filename))

    suffix = checkpoint_file.suffix.lower()
    if suffix == ".safetensors":
        loaded = load_file(str(checkpoint_file))
        return {str(key): value for key, value in loaded.items()}

    loaded_checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
    if isinstance(loaded_checkpoint, dict):
        if "state_dict" in loaded_checkpoint and isinstance(
            loaded_checkpoint["state_dict"], dict
        ):
            return {
                str(key): value for key, value in loaded_checkpoint["state_dict"].items()
            }
        if "model" in loaded_checkpoint and isinstance(loaded_checkpoint["model"], dict):
            return {str(key): value for key, value in loaded_checkpoint["model"].items()}
        if all(isinstance(value, torch.Tensor) for value in loaded_checkpoint.values()):
            return {str(key): value for key, value in loaded_checkpoint.items()}

    raise RuntimeError(
        "Unsupported PANNs checkpoint format. Expected a safetensors state dict, "
        "or a torch checkpoint with one of: ['state_dict', 'model']."
    )


def _strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("backbone.", "model.")
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
        stripped[normalized] = value
    return stripped


class PannsCnn14AudioModelWrapper(BaseAudioModelWrapper):
    """Differentiable wrapper around supervised PANNs Cnn14 AudioSet classifier."""

    def __init__(self, model: nn.Module, target_sample_rate: int = 32_000) -> None:
        super().__init__(model=model)
        self.target_sample_rate = int(target_sample_rate)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self._available_layers = ["embedding", "logits", "probs", "final"]
        self.metamer_layers: list[str] = []
        # Preserve canonical AudioSet indexing behavior (no remap CSV required).
        self.classifier_label_mids = None

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
        return self._resample_if_needed(waveform, sr)

    def input_log_mel_spectrogram(
        self, waveform: torch.Tensor, sr: int
    ) -> torch.Tensor:
        """Log-mel input to the CNN backbone as ``[mel_bins, time_frames]`` (for plots)."""
        processed = self.preprocess(waveform, sr=sr)
        spec = self.model.spectrogram_extractor(processed)
        logmel = self.model.logmel_extractor(spec)
        # PANNs returns [B, 1, time, mel]; transpose to [mel, time] for cochleagram-style plots.
        logmel = logmel.squeeze(0).squeeze(0)
        return logmel.transpose(0, 1)

    def _forward_backbone(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors qiuqiangkong/audioset_tagging_cnn Cnn14.forward, but exposes pre-sigmoid logits.
        x = self.model.spectrogram_extractor(waveform)
        x = self.model.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.model.bn0(x)
        x = x.transpose(1, 3)

        x = self.model.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.model.training)
        x = self.model.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.model.training)
        x = self.model.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.model.training)
        x = self.model.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.model.training)
        x = self.model.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.model.training)
        x = self.model.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.model.training)
        x = torch.mean(x, dim=3)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.model.training)
        x = F.relu_(self.model.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.model.training)
        logits = self.model.fc_audioset(x)
        return embedding, logits

    def forward_waveform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        return self.forward_features(waveform=waveform, sr=sr, layer="final")

    def forward_features(
        self, waveform: torch.Tensor, sr: int, layer: Optional[str] = None
    ) -> torch.Tensor:
        _, representations = self.forward_with_representations(waveform=waveform, sr=sr)
        selected_layer = layer or "final"
        if selected_layer not in representations:
            raise ValueError(
                f"Unknown layer '{selected_layer}'. Available: {self.available_layers}"
            )
        return representations[selected_layer]

    def forward_with_representations(
        self, waveform: torch.Tensor, sr: int = 32_000, fake_relu: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del fake_relu
        processed_waveform = self.preprocess(waveform, sr=sr)
        embedding, logits = self._forward_backbone(processed_waveform)
        probs = torch.sigmoid(logits)
        outputs = {
            "embedding": embedding,
            "logits": logits,
            "probs": probs,
            "final": logits,
        }
        return logits, outputs

    def get_classifier_logits(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        return self.forward_features(waveform=waveform, sr=sr, layer="logits")


def load_panns_cnn14_audio_model(
    checkpoint_path: Optional[str] = None,
    tokenizer_checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    freeze: bool = True,
    code_dir: Optional[str] = None,
    model_id: str = _DEFAULT_HF_REPO,
    model_filename: str = _DEFAULT_HF_FILENAME,
    sample_rate: int = 32_000,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14_000,
    classes_num: int = 527,
    **_unused_kwargs,
) -> PannsCnn14AudioModelWrapper:
    """Load supervised PANNs Cnn14 with AudioSet classifier head."""
    del tokenizer_checkpoint_path
    code_root = Path(code_dir) if code_dir else _DEFAULT_PANNS_CODE_ROOT
    cnn14_cls = _load_panns_cnn14_class(code_root=code_root)
    model = cnn14_cls(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        fmin=fmin,
        fmax=fmax,
        classes_num=classes_num,
    )

    raw_state_dict = _load_panns_state_dict(
        checkpoint_path=checkpoint_path,
        model_id=model_id,
        model_filename=model_filename,
    )
    normalized_state_dict = _strip_known_prefixes(raw_state_dict)
    model.load_state_dict(normalized_state_dict, strict=True)

    wrapper = PannsCnn14AudioModelWrapper(
        model=model,
        target_sample_rate=sample_rate,
    ).to(device)
    if freeze:
        wrapper.freeze()
    else:
        wrapper.eval()
    return wrapper
