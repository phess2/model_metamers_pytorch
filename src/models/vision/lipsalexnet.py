from typing import Optional

import torch
from torch import nn

from ..base import LipsModel
from ..layers.custom_modules import FakeReLUM
from ..layers.LipsLayers import LipsConv2d, LipsLinear


class LipsAlexNet(LipsModel):
    """
    Lipschitz-constrained AlexNet architecture.

    Pure ``nn.Module`` -- contains no Lightning, data-loading, or training
    logic.  Training orchestration is handled by ``LipsLightningModule``.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        w_max: float = 1.0,
        projection: Optional[str] = None,
        bound_method: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.w_max = w_max
        self.projection = projection
        self.bound_method = bound_method
        layer_kwargs = {
            "w_max": w_max,
            "projection": projection,
            "bound_method": bound_method,
        }

        self.features = nn.Sequential(
            LipsConv2d(
                3,
                64,
                kernel_size=11,
                stride=4,
                padding=2,
                **layer_kwargs,
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LipsConv2d(64, 192, kernel_size=5, padding=2, **layer_kwargs),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LipsConv2d(192, 384, kernel_size=3, padding=1, **layer_kwargs),
            nn.ReLU(inplace=False),
            LipsConv2d(384, 256, kernel_size=3, padding=1, **layer_kwargs),
            nn.ReLU(inplace=False),
            LipsConv2d(256, 256, kernel_size=3, padding=1, **layer_kwargs),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.feature_names = [
            "conv0",
            "relu0",
            "maxpool0",
            "conv1",
            "relu1",
            "maxpool1",
            "conv2",
            "relu2",
            "conv3",
            "relu3",
            "conv4",
            "relu4",
            "maxpool2",
        ]

        # FakeReLU modules for straight-through gradient estimation
        # (used during metamer / adversarial optimisation)
        self.fake_relu_dict = nn.ModuleDict()
        for name in self.feature_names:
            if "relu" in name:
                self.fake_relu_dict[name] = FakeReLUM()

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            LipsLinear(256 * 6 * 6, 4096, **layer_kwargs),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            LipsLinear(4096, 4096, **layer_kwargs),
            nn.ReLU(inplace=False),
            LipsLinear(4096, num_classes, **layer_kwargs),
        )
        self.classifier_names = [
            "dropout0",
            "fc0",
            "fc0_relu",
            "dropout1",
            "fc1",
            "fc1_relu",
            "fctop",
        ]
        self.fake_relu_dict["fc0_relu"] = FakeReLUM()
        self.fake_relu_dict["fc1_relu"] = FakeReLUM()

        self.metamer_layers = [
            "relu0",
            "relu1",
            "relu2",
            "relu3",
            "relu4",
            "fc0_relu",
            "fc1_relu",
            "final",
            # fake_relu variants
            "relu0_fake_relu",
            "relu1_fake_relu",
            "relu2_fake_relu",
            "relu3_fake_relu",
            "relu4_fake_relu",
            "fc0_relu_fake_relu",
            "fc1_relu_fake_relu",
        ]

    def __str__(self):
        return (
            f"LipsAlexNet(num_classes={self.num_classes}, "
            f"w_max={self.w_max}, projection={self.projection}, "
            f"bound_method={self.bound_method})"
        )

    def forward_with_representations(self, x, fake_relu=False):
        """Return ``(logits, all_outputs)`` with all intermediate activations."""
        logits, all_outputs = self.forward(
            x, fake_relu=fake_relu, return_all_outputs=True
        )
        return logits, all_outputs

    def forward(
        self,
        x: torch.Tensor,
        fake_relu: bool = False,
        return_all_outputs: bool = False,
    ) -> torch.Tensor:
        all_outputs = {}
        all_outputs["input_after_preproc"] = x

        for layer, name in zip(self.features, self.feature_names):
            if ("relu" in name) and fake_relu:
                all_outputs[name + "_fake_relu"] = self.fake_relu_dict[name](x)
            x = layer(x)
            if return_all_outputs:
                all_outputs[name] = x

        x = self.avgpool(x)
        if return_all_outputs:
            all_outputs["avgpool"] = x

        x = x.view(x.size(0), 256 * 6 * 6)
        if return_all_outputs:
            all_outputs["xview"] = x

        for layer, name in zip(self.classifier, self.classifier_names):
            if ("relu" in name) and fake_relu:
                all_outputs[name + "_fake_relu"] = self.fake_relu_dict[name](x)
            x = layer(x)
            if return_all_outputs:
                all_outputs[name] = x

        if return_all_outputs:
            all_outputs["final"] = all_outputs["fctop"]
            return x, all_outputs  # type: ignore
        return x
