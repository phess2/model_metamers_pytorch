from __future__ import annotations

import torch.nn as nn

from ...layers.custom_modules import FakeReLUM


class RobustAlexNetClassifier(nn.Module):
    """Legacy-compatible AlexNet classifier used by robust checkpoints."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
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

        self.fake_relu_dict = nn.ModuleDict()
        for layer_name in self.feature_names:
            if "relu" in layer_name:
                self.fake_relu_dict[layer_name] = FakeReLUM()

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
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

    def forward(
        self,
        x,
        with_latent: bool = False,
        fake_relu: bool = False,
        no_relu: bool = False,
    ):
        all_outputs = {"input_after_preproc": x}

        for layer, name in zip(self.features, self.feature_names):
            if ("relu" in name) and fake_relu:
                all_outputs[f"{name}_fake_relu"] = self.fake_relu_dict[name](x)
            x = layer(x)
            all_outputs[name] = x

        x = self.avgpool(x)
        all_outputs["avgpool"] = x

        x = x.view(x.size(0), 256 * 6 * 6)
        all_outputs["xview"] = x

        for layer, name in zip(self.classifier, self.classifier_names):
            if ("relu" in name) and fake_relu:
                all_outputs[f"{name}_fake_relu"] = self.fake_relu_dict[name](x)
            x = layer(x)
            all_outputs[name] = x

        all_outputs["final"] = all_outputs["fctop"]

        if with_latent and no_relu:
            raise ValueError("no_relu is deprecated")
        if with_latent:
            return x, None, all_outputs
        return x


def alexnet_classifier() -> RobustAlexNetClassifier:
    return RobustAlexNetClassifier()
