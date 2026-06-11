import unittest

import torch

from src.models.layers.bound_methods import (
    modular_linf_conv_bound,
    modular_linf_linear_bound,
    rms_spectral_linear_bound,
)
from src.models.layers.LipsLayers import LipsConv2d, LipsLinear


class BoundMethodTests(unittest.TestCase):
    def test_modular_linf_linear_matches_row_sum_norm(self):
        weight = torch.tensor([[1.0, -2.0], [0.5, 0.5]])
        scale = torch.tensor(1.0)
        expected = float(weight.abs().sum(dim=-1).max())
        self.assertAlmostEqual(
            modular_linf_linear_bound(weight, scale),
            expected,
            places=5,
        )

    def test_modular_linf_conv_uses_max_slice_norm(self):
        conv = LipsConv2d(2, 2, kernel_size=1, w_max=2.0, projection="modular_linf_cap")
        bound = modular_linf_conv_bound(conv.weight.data, conv.lips_weight_scale)
        self.assertGreater(bound, 0.0)

    def test_lips_linear_bound_method_switch(self):
        weight = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        scale = torch.tensor(1.0)
        spectral = rms_spectral_linear_bound(weight, scale)
        linf = modular_linf_linear_bound(weight, scale)
        self.assertAlmostEqual(spectral, 2.0, places=4)
        self.assertAlmostEqual(linf, 2.0, places=4)

        linf_layer = LipsLinear(
            2, 2, w_max=2.0, projection="modular_linf_cap", bound_method="modular_linf"
        )
        self.assertEqual(linf_layer.bound_method_name, "modular_linf")
        self.assertGreater(float(linf_layer.get_lips_bound()), 0.0)

    def test_modular_linf_projection_reduces_norm(self):
        layer = LipsLinear(
            4,
            4,
            w_max=0.5,
            projection="modular_linf_cap",
            bound_method="modular_linf",
        )
        with torch.no_grad():
            layer.weight.mul_(4.0)
        ratio = layer.project_()
        self.assertGreater(ratio, 0.0)
        self.assertLessEqual(float(layer.get_lips_bound()), 0.5 + 1e-4)


if __name__ == "__main__":
    unittest.main()
