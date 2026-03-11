import math
import unittest
from typing import Any, cast

import torch
from torch import nn

from src.models.layers.rms_bounds import (
    batchnorm2d_lips_bound,
    conv2d_rms_lips_bound,
    input_normalize_lips_bound,
    linear_rms_lips_bound,
    product_bound,
)
from src.models.vision.robust import build_internal_attacker_model
from src.models.vision.robust.alexnet import RobustAlexNetClassifier
from src.models.vision.robust.resnet import _Bottleneck, resnet50_classifier
from src.models.vision.robustvision import RobustAlexNet


class RobustLipsBoundsTests(unittest.TestCase):
    def test_linear_conv_and_batchnorm_helpers(self):
        linear = nn.Linear(2, 2, bias=False)
        linear.weight.data = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(linear_rms_lips_bound(linear), 2.0, places=4)

        conv = nn.Conv2d(2, 2, kernel_size=1, bias=False)
        conv.weight.data = torch.tensor(
            [[[[2.0]], [[0.0]]], [[[0.0]], [[1.0]]]], dtype=torch.float32
        )
        self.assertAlmostEqual(conv2d_rms_lips_bound(conv), 2.0, places=4)

        bn = nn.BatchNorm2d(2, affine=True)
        bn.eval()
        weight = cast(torch.Tensor, bn.weight)
        weight.data = torch.tensor([2.0, 0.5], dtype=torch.float32)
        running_var = cast(torch.Tensor, bn.running_var)
        running_var.data = torch.tensor([3.0, 1.0], dtype=torch.float32)
        expected_bn = max(
            2.0 / math.sqrt(3.0 + bn.eps),
            0.5 / math.sqrt(1.0 + bn.eps),
        )
        self.assertAlmostEqual(batchnorm2d_lips_bound(bn), expected_bn, places=4)

    def test_bottleneck_block_identity_skip_formula(self):
        block = _Bottleneck(inplanes=256, planes=64)
        torch.manual_seed(11)
        main_bound = product_bound(
            (
                conv2d_rms_lips_bound(block.conv1),
                batchnorm2d_lips_bound(block.bn1),
                conv2d_rms_lips_bound(block.conv2),
                batchnorm2d_lips_bound(block.bn2),
                conv2d_rms_lips_bound(block.conv3),
                batchnorm2d_lips_bound(block.bn3),
            )
        )
        torch.manual_seed(11)
        self.assertAlmostEqual(block.get_lips_bound(), 1.0 + main_bound, places=4)

    def test_classifier_bounds_are_finite(self):
        alexnet = RobustAlexNetClassifier()
        alex_bound = alexnet.get_lips_bound()
        self.assertTrue(math.isfinite(alex_bound))
        self.assertGreater(alex_bound, 0.0)

        resnet = resnet50_classifier()
        resnet_bound = resnet.get_lips_bound()
        self.assertTrue(math.isfinite(resnet_bound))
        self.assertGreater(resnet_bound, 0.0)

    def test_wrapper_bound_multiplies_preproc_and_classifier(self):
        wrapped = build_internal_attacker_model("alexnet")
        model = RobustAlexNet(variant="alexnet_l2_3_robust")
        model._wrapped_model = wrapped
        preproc_bound = input_normalize_lips_bound(
            cast(torch.Tensor, wrapped.preproc.normalize.new_std)
        )
        torch.manual_seed(7)
        classifier_bound = cast(Any, wrapped.model).get_lips_bound()
        torch.manual_seed(7)
        self.assertAlmostEqual(
            model.get_lips_bound(),
            preproc_bound * classifier_bound,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
