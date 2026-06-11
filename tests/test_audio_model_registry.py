import unittest
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
_AUDIO_BASE = importlib.import_module("models.audio.base")
_AUDIO_REGISTRY = importlib.import_module("models.audio.registry")
_AUDIO_AUDIOMAE = importlib.import_module("models.audio.audiomae")
_AUDIO_CLAP = importlib.import_module("models.audio.clap")
_AUDIO_BEATS = importlib.import_module("models.audio.beats")
_AUDIO_PANNS = importlib.import_module("models.audio.panns_cnn14")

BaseAudioModelWrapper = _AUDIO_BASE.BaseAudioModelWrapper
AudioMaeAudioModelWrapper = _AUDIO_AUDIOMAE.AudioMaeAudioModelWrapper
load_audiomae_audio_model = _AUDIO_AUDIOMAE.load_audiomae_audio_model
ClapAudioModelWrapper = _AUDIO_CLAP.ClapAudioModelWrapper
load_clap_audio_model = _AUDIO_CLAP.load_clap_audio_model
BeatsAudioModelWrapper = _AUDIO_BEATS.BeatsAudioModelWrapper
PannsCnn14AudioModelWrapper = _AUDIO_PANNS.PannsCnn14AudioModelWrapper
get_audio_model = _AUDIO_REGISTRY.get_audio_model
list_audio_models = _AUDIO_REGISTRY.list_audio_models

_RUN_AUDIO_MODEL_GPU_TESTS = os.environ.get("RUN_AUDIO_MODEL_GPU_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class _DummyAudioWrapper(BaseAudioModelWrapper):
    @property
    def available_layers(self) -> list[str]:
        return ["features"]

    def preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        del sr
        # Keep this differentiable to mirror metamer constraints.
        return waveform * 0.5

    def forward_waveform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        return self.model(self.preprocess(waveform, sr))

    def forward_features(
        self, waveform: torch.Tensor, sr: int, layer=None
    ) -> torch.Tensor:
        del layer
        return self.forward_waveform(waveform, sr)


class _FakeClapModel(nn.Module):
    def get_audio_features(self, input_features: torch.Tensor) -> torch.Tensor:
        pooled = input_features.mean(dim=-1)
        return pooled[:, :4]


class _FakeClapAudioModel:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return _FakeClapModel()


class _FakeAutoProcessor:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        class _Processor:
            def __call__(
                self,
                audio,
                sampling_rate: int,
                return_tensors: str,
                padding: bool = True,
            ):
                del sampling_rate, return_tensors, padding
                features = torch.zeros((len(audio), 4, 1001, 64), dtype=torch.float32)
                is_longer = torch.ones((len(audio), 1), dtype=torch.bool)
                return {"input_features": features, "is_longer": is_longer}

        return _Processor()


class _FakeAudioMaeBackbone(nn.Module):
    def __init__(self, with_classifier: bool):
        super().__init__()
        self.proj = nn.Conv2d(1, 8, kernel_size=(16, 16), stride=(16, 16), bias=False)
        self.head = nn.Linear(8, 4, bias=False) if with_classifier else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        patch_grid = self.proj(x)  # [B, 8, 64, 8]
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)  # [B, 512, 8]
        cls_token = patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, 8]
        return torch.cat([cls_token, patch_tokens], dim=1)  # [B, 513, 8]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = self.forward_features(x)[:, 0]
        return self.head(cls_token)

    def get_classifier(self):
        return self.head


class _FakeTimm:
    @staticmethod
    def create_model(model_name: str, pretrained: bool = True, **_kwargs):
        del pretrained
        if "audiomae_as2m_ft_as20k" in model_name:
            return _FakeAudioMaeBackbone(with_classifier=True)
        return _FakeAudioMaeBackbone(with_classifier=False)


class _FakeBeatsConfig:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.encoder_layers = cfg.get("encoder_layers", 3)
        self.finetuned_model = cfg.get("finetuned_model", False)


class _FakeTokenizerConfig:
    def __init__(self, cfg=None):
        del cfg


class _FakeBeatsEncoderLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, self_attn_padding_mask=None, need_weights=False, pos_bias=None):
        del self_attn_padding_mask, need_weights
        return self.proj(x), None, pos_bias


class _FakeBeatsEncoder(nn.Module):
    def __init__(self, layer_count: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeBeatsEncoderLayer(dim=dim) for _ in range(layer_count)]
        )
        self.layer_norm_first = False
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask=None, layer=None):
        del padding_mask
        x = x.transpose(0, 1)
        layer_results = []
        for idx, layer_module in enumerate(self.layers):
            x, _, _ = layer_module(x, self_attn_padding_mask=None, need_weights=False)
            if layer is not None and idx >= layer:
                break
        x = x.transpose(0, 1)
        return x, layer_results


class _FakeBeatsModel(nn.Module):
    def __init__(self, cfg: _FakeBeatsConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = nn.Conv2d(1, 8, kernel_size=4, stride=4, bias=False)
        self.layer_norm = nn.LayerNorm(8)
        self.post_extract_proj = nn.Linear(8, 8, bias=False)
        self.dropout_input = nn.Identity()
        self.encoder = _FakeBeatsEncoder(layer_count=cfg.encoder_layers, dim=8)
        if cfg.finetuned_model:
            self.predictor_dropout = nn.Identity()
            self.predictor = nn.Linear(8, 4, bias=False)
        else:
            self.predictor = None

    def preprocess(
        self,
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        features = source.unsqueeze(-1).repeat(1, 1, 128)
        target_frames = 32
        if features.shape[1] >= target_frames:
            features = features[:, :target_frames, :]
        else:
            features = F.pad(features, (0, 0, 0, target_frames - features.shape[1]))
        return (features - fbank_mean) / (2 * fbank_std)


class _FakeTokenizerModel(nn.Module):
    def __init__(self, cfg: _FakeTokenizerConfig):
        super().__init__()
        del cfg
        self.proj = nn.Linear(8, 8, bias=False)


class _FakeBeatsModule:
    BEATs = _FakeBeatsModel
    BEATsConfig = _FakeBeatsConfig


class _FakeTokenizersModule:
    Tokenizers = _FakeTokenizerModel
    TokenizersConfig = _FakeTokenizerConfig


def _fake_beats_checkpoint(finetuned_model: bool = False) -> dict:
    cfg = {"encoder_layers": 3, "finetuned_model": finetuned_model}
    model = _FakeBeatsModel(_FakeBeatsConfig(cfg))
    return {"cfg": cfg, "model": model.state_dict()}


def _fake_tokenizer_checkpoint() -> dict:
    tokenizer = _FakeTokenizerModel(_FakeTokenizerConfig())
    return {"cfg": {}, "model": tokenizer.state_dict()}


def _fake_beats_import_module(module_name: str):
    if module_name == "BEATs":
        return _FakeBeatsModule()
    if module_name == "Tokenizers":
        return _FakeTokenizersModule()
    raise ImportError(module_name)


class _FakePannsSpectrogram(nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform [B, T], got {tuple(waveform.shape)}")
        features = waveform.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        features = features.repeat(1, 1, 64, 513)
        return features


class _FakePannsLogMel(nn.Module):
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return spectrogram[..., :64]


class _FakePannsConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu(self.conv(x))
        if pool_type != "avg":
            raise ValueError("Only avg pooling is supported in this fake block.")
        return F.avg_pool2d(x, kernel_size=pool_size)


class _FakePannsCnn14(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        fmin: int,
        fmax: int,
        classes_num: int,
    ):
        super().__init__()
        del sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        self.spectrogram_extractor = _FakePannsSpectrogram()
        self.logmel_extractor = _FakePannsLogMel()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = _FakePannsConvBlock(channels=1)
        self.conv_block2 = _FakePannsConvBlock(channels=1)
        self.conv_block3 = _FakePannsConvBlock(channels=1)
        self.conv_block4 = _FakePannsConvBlock(channels=1)
        self.conv_block5 = _FakePannsConvBlock(channels=1)
        self.conv_block6 = _FakePannsConvBlock(channels=1)
        self.fc1 = nn.Linear(1, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)


class AudioModelRegistryTests(unittest.TestCase):
    def test_freeze_disables_model_parameter_gradients(self):
        wrapper = _DummyAudioWrapper(nn.Conv1d(1, 2, kernel_size=3))
        self.assertTrue(all(param.requires_grad for param in wrapper.parameters()))
        self.assertTrue(wrapper.training)

        wrapper.freeze()

        self.assertFalse(wrapper.training)
        self.assertTrue(all(not param.requires_grad for param in wrapper.parameters()))

    def test_freeze_keeps_waveform_path_differentiable(self):
        wrapper = _DummyAudioWrapper(nn.Conv1d(1, 2, kernel_size=3))
        wrapper.freeze()
        waveform = torch.randn(1, 1, 16, requires_grad=True)
        sr = 16000

        output = wrapper.forward_waveform(waveform, sr)
        output.sum().backward()

        if waveform.grad is None:
            self.fail("Expected gradient on input waveform, got None.")
        self.assertEqual(waveform.grad.shape, waveform.shape)

    def test_registry_lists_expected_placeholder_model_names(self):
        names = list_audio_models()
        self.assertEqual(names, sorted(names))
        for model_name in (
            "ssast",
            "audiomae",
            "audiomae_as2m",
            "audiomae_as2m_ft_as20k",
            "beats",
            "beats_iter3",
            "beats_iter3_plus_as2m",
            "clap",
            "panns_cnn14",
            "maskspec",
        ):
            self.assertIn(model_name, names)

    def test_registry_rejects_unknown_audio_model(self):
        with self.assertRaisesRegex(ValueError, "Unknown audio model 'unknown_model'"):
            get_audio_model("unknown_model", device="cpu")

    def test_placeholder_models_raise_helpful_errors(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            "Audio model 'ssast' is registered but not implemented yet.",
        ):
            get_audio_model("ssast", checkpoint_path="/tmp/ssast.ckpt", device="cpu")

    def test_clap_registry_entry_loads_concrete_wrapper(self):
        with patch.object(_AUDIO_CLAP, "ClapAudioModel", _FakeClapAudioModel), patch.object(
            _AUDIO_CLAP, "AutoProcessor", _FakeAutoProcessor
        ):
            wrapper = get_audio_model(
                "clap",
                device="cpu",
                sample_rate=16000,
                freeze=False,
            )
        self.assertIsInstance(wrapper, ClapAudioModelWrapper)

    def test_audiomae_registry_entry_loads_concrete_wrapper(self):
        with patch.object(
            _AUDIO_AUDIOMAE.importlib,
            "import_module",
            return_value=_FakeTimm(),
        ):
            wrapper = get_audio_model(
                "audiomae",
                device="cpu",
                freeze=False,
            )
        self.assertIsInstance(wrapper, AudioMaeAudioModelWrapper)

    def test_audiomae_pretrained_and_finetuned_aliases_behave_differently(self):
        with patch.object(
            _AUDIO_AUDIOMAE.importlib,
            "import_module",
            return_value=_FakeTimm(),
        ):
            pretrained_wrapper = get_audio_model("audiomae_as2m", device="cpu", freeze=False)
            finetuned_wrapper = get_audio_model(
                "audiomae_as2m_ft_as20k", device="cpu", freeze=False
            )
        self.assertNotIn("logits", pretrained_wrapper.available_layers)
        self.assertIn("logits", finetuned_wrapper.available_layers)

    def test_beats_registry_entry_loads_concrete_wrapper(self):
        fake_checkpoint = _fake_beats_checkpoint(finetuned_model=False)
        fake_tokenizer_checkpoint = _fake_tokenizer_checkpoint()
        with patch.object(
            _AUDIO_BEATS.importlib,
            "import_module",
            side_effect=_fake_beats_import_module,
        ), patch.object(
            _AUDIO_BEATS.torch,
            "load",
            side_effect=[fake_checkpoint, fake_tokenizer_checkpoint],
        ):
            wrapper = get_audio_model(
                "beats",
                checkpoint_path="/tmp/beats.pt",
                tokenizer_checkpoint_path="/tmp/tokenizer.pt",
                device="cpu",
                freeze=False,
            )
        self.assertIsInstance(wrapper, BeatsAudioModelWrapper)

    def test_panns_registry_entry_loads_concrete_wrapper(self):
        fake_model = _FakePannsCnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        prefixed_state = {
            f"backbone.{name}": tensor.clone()
            for name, tensor in fake_model.state_dict().items()
        }
        with patch.object(
            _AUDIO_PANNS,
            "_load_panns_cnn14_class",
            return_value=_FakePannsCnn14,
        ), patch.object(
            _AUDIO_PANNS,
            "_load_panns_state_dict",
            return_value=prefixed_state,
        ):
            wrapper = get_audio_model("panns_cnn14", device="cpu", freeze=False)

        self.assertIsInstance(wrapper, PannsCnn14AudioModelWrapper)
        waveform = torch.randn(1, 1, 64)
        logits = wrapper.get_classifier_logits(waveform, sr=16_000)
        self.assertEqual(tuple(logits.shape), (1, 527))
        self.assertTrue(torch.isfinite(logits).all())

    def test_beats_aliases_resolve_expected_default_checkpoint_pairs(self):
        calls: list[str] = []
        fake_checkpoint = _fake_beats_checkpoint(finetuned_model=False)
        fake_tokenizer_checkpoint = _fake_tokenizer_checkpoint()

        def _fake_torch_load(path, *args, **kwargs):
            del args, kwargs
            calls.append(str(path))
            if "Tokenizer" in str(path):
                return fake_tokenizer_checkpoint
            return fake_checkpoint

        with patch.object(
            _AUDIO_BEATS.importlib,
            "import_module",
            side_effect=_fake_beats_import_module,
        ), patch.object(
            _AUDIO_BEATS.torch,
            "load",
            side_effect=_fake_torch_load,
        ):
            get_audio_model("beats_iter3", device="cpu", freeze=False)
            get_audio_model("beats_iter3_plus_as2m", device="cpu", freeze=False)

        self.assertTrue(any("BEATs_iter3.pt" in call for call in calls))
        self.assertTrue(any("Tokenizer_iter3.pt" in call for call in calls))
        self.assertTrue(
            any("BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt" in call for call in calls)
        )
        self.assertTrue(any("Tokenizer_iter3_plus_AS2M.pt" in call for call in calls))

    def test_beats_wrapper_forward_with_representations_keeps_gradient(self):
        model = _FakeBeatsModel(_FakeBeatsConfig({"encoder_layers": 3}))
        tokenizer = _FakeTokenizerModel(_FakeTokenizerConfig())
        wrapper = BeatsAudioModelWrapper(model=model, tokenizer=tokenizer, target_sample_rate=16000)

        waveform = torch.randn(2, 1, 128, requires_grad=True)
        output, reps = wrapper.forward_with_representations(waveform, sr=16000)

        self.assertIn("input_features", reps)
        self.assertIn("patch_features", reps)
        self.assertIn("projected_features", reps)
        self.assertIn("transformer_layer_00", reps)
        self.assertIn("transformer_layer_01", reps)
        self.assertIn("transformer_layer_02", reps)
        self.assertIn("final_pooled_embedding", reps)
        self.assertIn("final", reps)
        self.assertEqual(
            wrapper.metamer_layers,
            [
                "projected_features",
                "transformer_layer_00",
                "transformer_layer_01",
                "transformer_layer_02",
                "final_pooled_embedding",
            ],
        )
        self.assertTrue(torch.equal(output, reps["final"]))

        loss = reps["projected_features"].sum() + reps["transformer_layer_01"].sum()
        loss.backward()
        if waveform.grad is None:
            self.fail("Expected gradient on BEATs waveform input, got None.")
        self.assertEqual(waveform.grad.shape, waveform.shape)

    def test_audiomae_wrapper_forward_with_representations(self):
        wrapper = AudioMaeAudioModelWrapper(_FakeAudioMaeBackbone(with_classifier=False))
        waveform = torch.randn(2, 1, 16000)

        with patch.object(
            wrapper,
            "_compute_fbank",
            side_effect=lambda x: (
                x.unsqueeze(-1).repeat(1, 1, wrapper.num_mel_bins)[:, : wrapper.target_frames, :]
                if x.shape[1] >= wrapper.target_frames
                else F.pad(
                    x.unsqueeze(-1).repeat(1, 1, wrapper.num_mel_bins),
                    (0, 0, 0, wrapper.target_frames - x.shape[1]),
                )
            ),
        ):
            output, reps = wrapper.forward_with_representations(waveform, sr=16000)

        self.assertIn("input_features", reps)
        self.assertIn("tokens", reps)
        self.assertIn("cls_token", reps)
        self.assertIn("patch_tokens", reps)
        self.assertIn("vit_patch_embeddings", reps)
        self.assertIn("final_pooled_embedding", reps)
        self.assertIn("frame_features", reps)
        self.assertIn("final", reps)
        self.assertEqual(reps["tokens"].shape[1], 513)
        self.assertEqual(reps["patch_tokens"].shape[1], 512)
        self.assertEqual(reps["vit_patch_embeddings"].shape[1], 512)
        self.assertEqual(reps["frame_features"].shape[1], 64)
        self.assertTrue(torch.equal(output, reps["final"]))
        self.assertEqual(
            wrapper.metamer_layers,
            ["vit_patch_embeddings", "final_pooled_embedding"],
        )

    def test_audiomae_preprocess_normalizes_supported_waveform_shapes(self):
        wrapper = AudioMaeAudioModelWrapper(_FakeAudioMaeBackbone(with_classifier=False))
        observed_shapes: list[tuple[int, ...]] = []

        def _fake_compute_fbank(x: torch.Tensor) -> torch.Tensor:
            observed_shapes.append(tuple(x.shape))
            features = x.unsqueeze(-1).repeat(1, 1, wrapper.num_mel_bins)
            if features.shape[1] >= wrapper.target_frames:
                return features[:, : wrapper.target_frames, :]
            return F.pad(features, (0, 0, 0, wrapper.target_frames - features.shape[1]))

        with patch.object(wrapper, "_compute_fbank", side_effect=_fake_compute_fbank):
            waveform_1d = torch.randn(16000)
            waveform_2d = torch.randn(2, 16000)
            waveform_3d = torch.randn(2, 1, 16000)

            features_1d = wrapper.preprocess(waveform_1d, sr=16000)
            features_2d = wrapper.preprocess(waveform_2d, sr=16000)
            features_3d = wrapper.preprocess(waveform_3d, sr=16000)

        self.assertEqual(observed_shapes[0], (1, 16000))
        self.assertEqual(observed_shapes[1], (2, 16000))
        self.assertEqual(observed_shapes[2], (2, 16000))
        self.assertEqual(
            features_1d.shape, (1, 1, wrapper.target_frames, wrapper.num_mel_bins)
        )
        self.assertEqual(
            features_2d.shape, (2, 1, wrapper.target_frames, wrapper.num_mel_bins)
        )
        self.assertEqual(
            features_3d.shape, (2, 1, wrapper.target_frames, wrapper.num_mel_bins)
        )

    def test_audiomae_wrapper_exposes_classifier_logits_when_head_exists(self):
        wrapper = AudioMaeAudioModelWrapper(_FakeAudioMaeBackbone(with_classifier=True))
        waveform = torch.randn(1, 1, 16000)
        with patch.object(
            wrapper,
            "_compute_fbank",
            side_effect=lambda x: (
                x.unsqueeze(-1).repeat(1, 1, wrapper.num_mel_bins)[:, : wrapper.target_frames, :]
                if x.shape[1] >= wrapper.target_frames
                else F.pad(
                    x.unsqueeze(-1).repeat(1, 1, wrapper.num_mel_bins),
                    (0, 0, 0, wrapper.target_frames - x.shape[1]),
                )
            ),
        ):
            logits = wrapper.get_classifier_logits(waveform, sr=16000)
        self.assertEqual(logits.shape, (1, 4))

    def test_audiomae_wrapper_pretrained_raises_for_classifier_logits(self):
        wrapper = AudioMaeAudioModelWrapper(_FakeAudioMaeBackbone(with_classifier=False))
        waveform = torch.randn(1, 1, 16000)
        with patch.object(
            wrapper,
            "_compute_fbank",
            side_effect=lambda x: (
                x.unsqueeze(-1).repeat(1, 1, wrapper.num_mel_bins)[:, : wrapper.target_frames, :]
                if x.shape[1] >= wrapper.target_frames
                else F.pad(
                    x.unsqueeze(-1).repeat(1, 1, wrapper.num_mel_bins),
                    (0, 0, 0, wrapper.target_frames - x.shape[1]),
                )
            ),
        ):
            with self.assertRaisesRegex(NotImplementedError, "does not expose classifier logits"):
                wrapper.get_classifier_logits(waveform, sr=16000)

    def test_audiomae_wrapper_keeps_gradient_through_preprocess_path(self):
        wrapper = AudioMaeAudioModelWrapper(_FakeAudioMaeBackbone(with_classifier=False))
        waveform = torch.randn(2, 1, 16000, requires_grad=True)
        with patch.object(
            wrapper,
            "_compute_fbank",
            side_effect=lambda x: (
                x.unsqueeze(-1).repeat(1, 1, wrapper.num_mel_bins)[:, : wrapper.target_frames, :]
                if x.shape[1] >= wrapper.target_frames
                else F.pad(
                    x.unsqueeze(-1).repeat(1, 1, wrapper.num_mel_bins),
                    (0, 0, 0, wrapper.target_frames - x.shape[1]),
                )
            ),
        ):
            output = wrapper.forward_waveform(waveform, sr=16000)
        output.sum().backward()
        if waveform.grad is None:
            self.fail("Expected gradient on AudioMAE waveform input, got None.")
        self.assertEqual(waveform.grad.shape, waveform.shape)

    def test_clap_wrapper_forward_with_representations_keeps_gradient(self):
        wrapper = ClapAudioModelWrapper(_FakeClapModel(), target_sample_rate=16000)
        waveform = torch.randn(2, 1, 3200, requires_grad=True)
        output, reps = wrapper.forward_with_representations(waveform, sr=16000)

        self.assertIn("input_features", reps)
        self.assertIn("audio_hidden_states", reps)
        self.assertIn("audio_embeds", reps)
        self.assertIn("final_audio_embedding", reps)
        self.assertIn("final", reps)
        self.assertTrue(torch.equal(output, reps["final"]))
        self.assertIn("audio_hidden_states", wrapper.available_layers)
        self.assertIn("final_audio_embedding", wrapper.available_layers)
        self.assertEqual(
            wrapper.metamer_layers, ["audio_hidden_states", "final_audio_embedding"]
        )

        loss = output.sum()
        loss.backward()
        if waveform.grad is None:
            self.fail("Expected gradient on CLAP waveform input, got None.")
        self.assertEqual(waveform.grad.shape, waveform.shape)

    def test_clap_loader_surfaces_transformers_loading_errors(self):
        with patch.object(
            _AUDIO_CLAP.ClapAudioModel,
            "from_pretrained",
            side_effect=ImportError("transformers not installed"),
        ):
            with self.assertRaisesRegex(ImportError, "transformers not installed"):
                load_clap_audio_model(device="cpu")

    def test_audiomae_loader_reports_missing_timm_dependency(self):
        with patch.object(
            _AUDIO_AUDIOMAE.importlib,
            "import_module",
            side_effect=ImportError("timm not installed"),
        ):
            with self.assertRaisesRegex(ImportError, "requires the `timm` package"):
                load_audiomae_audio_model(device="cpu")


@unittest.skipUnless(
    _RUN_AUDIO_MODEL_GPU_TESTS and torch.cuda.is_available(),
    "Set RUN_AUDIO_MODEL_GPU_TESTS=1 and run on a CUDA host.",
)
class AudioModelRegistryGpuTests(unittest.TestCase):
    def test_real_audio_models_forward_full_clip_and_expose_metamer_features(self):
        model_names = [
            "clap",
            "audiomae",
            "audiomae_as2m",
            "audiomae_as2m_ft_as20k",
            "beats",
            "beats_iter3",
            "beats_iter3_plus_as2m",
        ]
        torch.manual_seed(0)
        waveform = torch.randn(1, 1, 10 * 16_000, device="cuda")

        for model_name in model_names:
            with self.subTest(model_name=model_name):
                wrapper = get_audio_model(model_name, device="cuda", freeze=True)
                with torch.no_grad():
                    output, reps = wrapper.forward_with_representations(waveform, sr=16_000)

                self.assertIn("final", reps)
                self.assertTrue(torch.equal(output, reps["final"]))
                self.assertEqual(output.shape[0], waveform.shape[0])
                self.assertTrue(torch.isfinite(output).all())

                metamer_layers = list(getattr(wrapper, "metamer_layers"))
                self.assertGreater(
                    len(metamer_layers),
                    0,
                    msg=f"{model_name} did not define any metamer layers.",
                )
                for layer_name in metamer_layers:
                    self.assertIn(
                        layer_name,
                        reps,
                        msg=f"{model_name} missing metamer layer '{layer_name}' in outputs.",
                    )
                    layer_features = reps[layer_name]
                    self.assertIsInstance(layer_features, torch.Tensor)
                    self.assertEqual(layer_features.shape[0], waveform.shape[0])
                    self.assertTrue(
                        torch.isfinite(layer_features).all(),
                        msg=f"{model_name}:{layer_name} produced non-finite values.",
                    )


if __name__ == "__main__":
    unittest.main()
