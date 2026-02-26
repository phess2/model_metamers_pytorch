#!/usr/bin/env python
"""
Sanity check script for the new metrics implementation.
Tests that top-1/top-5 accuracy and per-layer norm-change ratios work correctly.

Run with:
    python scripts/sanity_check_metrics.py
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.layers.LipsLayers import LipsLinear, LipsConv2d


def test_project_returns_ratio():
    """Test that project_() returns a valid norm-change ratio."""
    print("=" * 60)
    print("Testing project_() returns norm-change ratio")
    print("=" * 60)

    # Test LipsLinear
    print("\n1. Testing LipsLinear with spectral_normalize projection:")
    linear = LipsLinear(128, 64, w_max=6.0, projection="spectral_normalize")

    # Perturb weights to simulate optimizer update
    with torch.no_grad():
        linear.weight.data += torch.randn_like(linear.weight.data) * 0.1

    ratio = linear.project_()
    print(f"   Norm-change ratio: {ratio:.6f}")
    assert isinstance(ratio, float), "project_() should return a float"
    assert ratio >= 0, "Norm-change ratio should be non-negative"
    print("   PASSED: LipsLinear returns valid ratio")

    # Test LipsConv2d
    print("\n2. Testing LipsConv2d with orthogonalize projection:")
    conv = LipsConv2d(64, 128, kernel_size=3, w_max=6.0, projection="orthogonalize")

    # Perturb weights to simulate optimizer update
    with torch.no_grad():
        conv.weight.data += torch.randn_like(conv.weight.data) * 0.1

    ratio = conv.project_()
    print(f"   Norm-change ratio: {ratio:.6f}")
    assert isinstance(ratio, float), "project_() should return a float"
    assert ratio >= 0, "Norm-change ratio should be non-negative"
    print("   PASSED: LipsConv2d returns valid ratio")

    # Test with projection=None
    print("\n3. Testing LipsLinear with projection=None:")
    linear_no_proj = LipsLinear(128, 64, w_max=6.0, projection=None)
    ratio_none = linear_no_proj.project_()
    print(f"   Norm-change ratio: {ratio_none:.6f}")
    assert ratio_none == 0.0, "project_() should return 0.0 when projection is None"
    print("   PASSED: Returns 0.0 when projection is None")


def test_w_max_zero_no_projection():
    """With w_max=0 or projection=None, no projection is applied; init and get_lips_bound stay valid."""
    print("\n" + "=" * 60)
    print("Testing w_max=0 / no-projection mode")
    print("=" * 60)

    # w_max=0, projection=None: scale = sqrt(out/in), lips_weight_scale = scale (RMS->RMS)
    linear = LipsLinear(128, 64, w_max=0.0, projection=None)
    expected_scale = (64 / 128) ** 0.5
    assert abs(linear.scale.item() - expected_scale) < 1e-6, (
        "LipsLinear scale should be sqrt(out/in) when no projection"
    )
    assert linear.lips_weight_scale.item() == linear.scale.item(), (
        "LipsLinear lips_weight_scale should equal scale when no projection"
    )
    assert not torch.allclose(linear.weight, torch.zeros_like(linear.weight)), (
        "LipsLinear weights must not be zero (orthogonal init with scale)"
    )
    assert not torch.isnan(linear.weight).any(), "LipsLinear weights must not be NaN"
    ratio = linear.project_()
    assert ratio == 0.0, "project_() should return 0.0 when w_max=0"
    bound = linear.get_lips_bound()
    assert not torch.isnan(bound).any() and not torch.isinf(bound).any(), (
        "get_lips_bound() must be finite when no projection"
    )
    print(
        "   PASSED: LipsLinear with w_max=0 has valid init, project_ returns 0, bound finite"
    )

    conv = LipsConv2d(4, 16, kernel_size=3, w_max=0.0, projection=None)
    expected_conv_scale = (16 / 4) ** 0.5 / (3 * 3)
    assert abs(conv.scale.item() - expected_conv_scale) < 1e-6, (
        "LipsConv2d scale should be sqrt(oc/ic)/(kh*kw) when no projection"
    )
    assert conv.lips_weight_scale.item() == conv.scale.item(), (
        "LipsConv2d lips_weight_scale should equal scale when no projection"
    )
    assert not torch.isnan(conv.weight).any(), "LipsConv2d weights must not be NaN"
    ratio = conv.project_()
    assert ratio == 0.0, "project_() should return 0.0 when w_max=0"
    bound = conv.get_lips_bound()
    assert not torch.isnan(bound).any() and not torch.isinf(bound).any(), (
        "get_lips_bound() must be finite when no projection"
    )
    print(
        "   PASSED: LipsConv2d with w_max=0 has valid init, project_ returns 0, bound finite"
    )

    # projection=None with w_max>0 also uses no-projection mode (scale = sqrt(out/in), lips_weight_scale = scale)
    linear2 = LipsLinear(32, 16, w_max=2.0, projection=None)
    expected_scale2 = (16 / 32) ** 0.5
    assert (
        abs(linear2.scale.item() - expected_scale2) < 1e-6
        and abs(linear2.lips_weight_scale.item() - linear2.scale.item()) < 1e-6
    ), (
        "LipsLinear with projection=None should use scale=sqrt(out/in), lips_weight_scale=scale"
    )
    assert linear2.project_() == 0.0
    print("   PASSED: projection=None with w_max>0 uses no-projection init")


def test_topk_accuracy():
    """Test that top-k accuracy calculation works correctly."""
    print("\n" + "=" * 60)
    print("Testing top-1 and top-5 accuracy computation")
    print("=" * 60)

    # Create deterministic logits for controlled test
    batch_size = 16
    num_classes = 1000

    # Start with zeros
    output = torch.zeros(batch_size, num_classes)
    target = torch.arange(batch_size, dtype=torch.long)  # target[i] = i

    # Set up controlled logits:
    # Samples 0-7: target class has highest logit (top-1 correct)
    # Samples 8-11: target class has 3rd highest logit (top-5 correct, top-1 wrong)
    # Samples 12-15: target class has very low logit (neither top-1 nor top-5)

    for i in range(8):  # Make top-1 correct for first 8 samples
        output[i, target[i].item()] = 100.0  # Highest
        output[i, (target[i].item() + 1) % num_classes] = 50.0  # Second

    for i in range(8, 12):  # Make top-5 (but not top-1) correct for next 4 samples
        # Put 4 classes higher than target
        for k in range(4):
            output[i, (target[i].item() + k + 1) % num_classes] = 100.0 - k
        output[i, target[i].item()] = 90.0  # 5th highest

    for i in range(12, 16):  # Make neither top-1 nor top-5 correct
        # Put 10 classes higher than target
        for k in range(10):
            output[i, (target[i].item() + k + 1) % num_classes] = 100.0 - k
        output[i, target[i].item()] = 0.0  # Very low

    # Compute top-k accuracy using the same logic as in validation_step
    _, pred = output.topk(5, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    top1_correct = correct[0].sum().item()
    top5_correct = correct.any(dim=0).sum().item()

    top1_acc = top1_correct / batch_size
    top5_acc = top5_correct / batch_size

    print(f"\n   Batch size: {batch_size}")
    print(f"   Top-1 correct: {top1_correct}, Top-1 accuracy: {top1_acc:.4f}")
    print(f"   Top-5 correct: {top5_correct}, Top-5 accuracy: {top5_acc:.4f}")

    # Verify expected results
    expected_top1 = 8
    expected_top5 = 12  # 8 (top-1) + 4 (top-5 only)

    assert top1_correct == expected_top1, (
        f"Expected {expected_top1} top-1 correct, got {top1_correct}"
    )
    assert top5_correct == expected_top5, (
        f"Expected {expected_top5} top-5 correct, got {top5_correct}"
    )
    assert top5_acc >= top1_acc, "Top-5 accuracy should be >= top-1 accuracy"
    print("   PASSED: Top-k accuracy computation is correct")


def test_module_integration():
    """Test that current model + Lightning module wiring is valid."""
    print("\n" + "=" * 60)
    print("Testing LipsLightningModule integration")
    print("=" * 60)

    # Create a minimal config
    config = {
        "model_name": "lipsalexnet",
        "hparams": {
            "num_classes": 100,  # Smaller for faster test
            "w_max": 6.0,
            "projection": "spectral_normalize",
            "max_epochs": 1,
        },
        "data_settings": {
            "batch_size": 4,
            "data_dir": "/tmp/fake_data",  # Won't actually use
        },
        "optim_settings": {
            "mult_optimizers": False,
            "optimizer": {
                "name": "AdamW",
                "kwargs": {"lr": 0.001},
            },
        },
    }

    from src.models.registry import get_model
    from src.training.losses import get_loss_fn
    from src.training.module import LipsLightningModule

    print("\n   Creating model + LipsLightningModule...")
    model = get_model(config)
    loss_fn = get_loss_fn(config)
    module = LipsLightningModule(model=model, config=config, loss_fn=loss_fn)

    # Check norm ratio tracking containers exist
    assert hasattr(module, "_norm_ratio_sums"), "Module should track norm ratio sums"
    assert hasattr(module, "_norm_ratio_counts"), (
        "Module should track norm ratio counts"
    )

    # Run a fake forward pass
    print("\n   Running forward pass with random data...")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = module(x)
    print(f"   Output shape: {output.shape}")
    assert output.shape[0] == 2, "Forward pass should preserve batch dimension"
    print("   PASSED: LipsLightningModule integration works")


def main():
    print("\n" + "=" * 60)
    print("SANITY CHECK: New Metrics Implementation")
    print("=" * 60)

    test_project_returns_ratio()
    test_w_max_zero_no_projection()
    test_topk_accuracy()
    test_module_integration()

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED!")
    print("=" * 60)
    print(
        "\nThe implementation is ready. New metrics will be logged to Weights & Biases:"
    )
    print("  - val/top1_acc: Top-1 validation accuracy")
    print("  - val/top5_acc: Top-5 validation accuracy")
    print("  - train/norm_ratio/<layer_name>: Per-layer norm-change ratio")
    print("\nRun a full training job to verify metrics appear in Wandb.")


if __name__ == "__main__":
    main()
