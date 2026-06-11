import json
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.analysis.metamer import load_model_from_checkpoint, resolve_model_normalize_fn
from src.models.registry import get_model
from src.models.vision.robustvision import (
    RobustAlexNet,
    RobustResNet50,
    get_supported_robust_variants,
)


class _DummyAttackerModel(torch.nn.Module):
    def forward(
        self,
        x,
        with_latent=False,
        fake_relu=False,
        with_image=True,
    ):
        del with_latent, fake_relu, with_image
        logits = torch.zeros((x.shape[0], 1000), dtype=x.dtype, device=x.device)
        all_outputs = {"input_after_preproc": x, "final": logits}
        return (logits, None, all_outputs), x


class _DummyRobustModel(torch.nn.Module):
    metamer_normalize_mode = "identity"

    def __init__(self):
        super().__init__()
        self.loaded = None
        self.to_device = None
        self.is_eval = False

    def load_checkpoint(self, ckpt_path, device="cuda"):
        self.loaded = (ckpt_path, device)

    def to(self, *args, **kwargs):
        self.to_device = args[0] if args else kwargs.get("device")
        return self

    def eval(self):
        self.is_eval = True
        return self


class RobustVisionLoadingTests(unittest.TestCase):
    def test_registry_instantiates_robust_models(self):
        alex_cfg = {
            "model_name": "robustalexnet",
            "hparams": {"variant": "alexnet_l2_3_robust"},
        }
        resnet_cfg = {
            "model_name": "robustresnet50",
            "hparams": {"variant": "resnet50_l2_3_robust"},
        }

        alex_model = get_model(alex_cfg)
        resnet_model = get_model(resnet_cfg)

        self.assertIsInstance(alex_model, RobustAlexNet)
        self.assertIsInstance(resnet_model, RobustResNet50)
        self.assertIn("fc1_relu_fake_relu", alex_model.metamer_layers)
        self.assertIn("layer4_fake_relu", resnet_model.metamer_layers)

        alex_random_cfg = {
            "model_name": "robustalexnet",
            "hparams": {"variant": "alexnet_random_l2_3_perturb"},
        }
        resnet_random_cfg = {
            "model_name": "robustresnet50",
            "hparams": {"variant": "resnet50_random_l2_perturb"},
        }
        self.assertIsInstance(get_model(alex_random_cfg), RobustAlexNet)
        self.assertIsInstance(get_model(resnet_random_cfg), RobustResNet50)

    def test_forward_with_representations_from_adapter(self):
        model = RobustAlexNet(variant="alexnet_l2_3_robust")
        model._wrapped_model = _DummyAttackerModel()
        x = torch.rand(2, 3, 224, 224)

        logits, all_outputs = model.forward_with_representations(x, fake_relu=True)
        self.assertEqual(logits.shape[0], 2)
        self.assertIn("input_after_preproc", all_outputs)
        self.assertIn("final", all_outputs)

    def test_normalize_fn_identity_for_robust_models(self):
        model = RobustResNet50(variant="resnet50_linf_8_robust")
        fn = resolve_model_normalize_fn(model, {"metamer_normalization": "identity"})
        x = torch.rand(1, 3, 8, 8)
        self.assertTrue(torch.equal(fn(x), x))

    def test_supported_variants_include_random_perturb_models(self):
        variants = get_supported_robust_variants()
        self.assertIn("alexnet_random_l2_3_perturb", variants["robustalexnet"])
        self.assertIn("alexnet_random_linf8_perturb", variants["robustalexnet"])
        self.assertIn("resnet50_random_l2_perturb", variants["robustresnet50"])
        self.assertIn("resnet50_random_linf8_perturb", variants["robustresnet50"])

    def test_robust_builder_no_external_robustness_import(self):
        source = inspect.getsource(RobustAlexNet._build_unloaded_model)
        self.assertNotIn("robustness", source)

    def test_load_model_from_checkpoint_robust_branch(self):
        config = {
            "model_name": "robustalexnet",
            "checkpoint_format": "robustness",
            "hparams": {"variant": "alexnet_l2_3_robust"},
        }

        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "robust_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)

            dummy = _DummyRobustModel()
            ckpt_path = Path(tempdir) / "checkpoint.pt"
            with patch("src.analysis.metamer.get_model", return_value=dummy):
                model, loaded_cfg = load_model_from_checkpoint(
                    config_path=config_path,
                    ckpt_path=ckpt_path,
                    device="cpu",
                )

        self.assertIs(model, dummy)
        self.assertEqual(loaded_cfg["checkpoint_format"], "robustness")
        self.assertEqual(dummy.loaded, (str(ckpt_path), "cpu"))
        self.assertEqual(dummy.to_device, "cpu")
        self.assertTrue(dummy.is_eval)


if __name__ == "__main__":
    unittest.main()
