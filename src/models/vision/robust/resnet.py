from __future__ import annotations

import torch.nn as nn

from ...layers.custom_modules import FakeReLU, SequentialWithArgs
from ...layers.rms_bounds import (
    batchnorm2d_lips_bound,
    conv2d_rms_lips_bound,
    linear_rms_lips_bound,
    product_bound,
)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _bound_for_module(module: nn.Module) -> float:
    if isinstance(module, nn.Conv2d):
        return conv2d_rms_lips_bound(module)
    if isinstance(module, nn.Linear):
        return linear_rms_lips_bound(module)
    if isinstance(module, nn.BatchNorm2d):
        return batchnorm2d_lips_bound(module)
    return 1.0


def _module_bound_product(module: nn.Module | None) -> float:
    if module is None:
        return 1.0
    bounds = [_bound_for_module(submodule) for submodule in module.modules()]
    return product_bound(bounds)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        last_block: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.last_block = last_block

    def get_lips_bound(self) -> float:
        main_bound = product_bound(
            (
                conv2d_rms_lips_bound(self.conv1),
                batchnorm2d_lips_bound(self.bn1),
                conv2d_rms_lips_bound(self.conv2),
                batchnorm2d_lips_bound(self.bn2),
            )
        )
        skip_bound = _module_bound_product(self.downsample)
        return skip_bound + main_bound

    def forward(self, x, fake_relu: bool = False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if fake_relu and self.last_block:
            return FakeReLU.apply(out)
        return self.relu(out)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        last_block: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.last_block = last_block

    def get_lips_bound(self) -> float:
        main_bound = product_bound(
            (
                conv2d_rms_lips_bound(self.conv1),
                batchnorm2d_lips_bound(self.bn1),
                conv2d_rms_lips_bound(self.conv2),
                batchnorm2d_lips_bound(self.bn2),
                conv2d_rms_lips_bound(self.conv3),
                batchnorm2d_lips_bound(self.bn3),
            )
        )
        skip_bound = _module_bound_product(self.downsample)
        return skip_bound + main_bound

    def forward(self, x, fake_relu: bool = False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if fake_relu and self.last_block:
            return FakeReLU.apply(out)
        return self.relu(out)


class RobustResNetClassifier(nn.Module):
    """Legacy-compatible ResNet classifier used by robust checkpoints."""

    def __init__(
        self,
        block: type[_BasicBlock] | type[_Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, _Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, _BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def get_lips_bound(self) -> float:
        stem_bound = product_bound(
            (
                conv2d_rms_lips_bound(self.conv1),
                batchnorm2d_lips_bound(self.bn1),
            )
        )
        stage_bounds: list[float] = []
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            stage_bounds.extend(
                block.get_lips_bound()  # type: ignore[attr-defined]
                for block in stage._modules.values()
                if isinstance(block, (_BasicBlock, _Bottleneck))
            )
        head_bound = linear_rms_lips_bound(self.fc)
        return product_bound((stem_bound, *stage_bounds, head_bound))

    def _make_layer(
        self,
        block: type[_BasicBlock] | type[_Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> SequentialWithArgs:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: list[nn.Module] = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for block_idx in range(1, blocks):
            is_last = block_idx == (blocks - 1)
            layers.append(block(self.inplanes, planes, last_block=is_last))

        return SequentialWithArgs(*layers)

    def forward(
        self,
        x,
        with_latent: bool = False,
        fake_relu: bool = False,
        no_relu: bool = False,
    ):
        del no_relu
        all_outputs = {"input_after_preproc": x}

        x = self.conv1(x)
        all_outputs["conv1"] = x
        x = self.bn1(x)
        all_outputs["bn1"] = x
        if fake_relu and with_latent:
            all_outputs["conv1_relu1_fake_relu"] = FakeReLU.apply(x)
        x = self.relu(x)
        all_outputs["conv1_relu1"] = x
        x = self.maxpool(x)
        all_outputs["maxpool1"] = x

        if fake_relu and with_latent:
            all_outputs["layer1_fake_relu"] = self.layer1(x, fake_relu=fake_relu)
        x = self.layer1(x)
        all_outputs["layer1"] = x

        if fake_relu and with_latent:
            all_outputs["layer2_fake_relu"] = self.layer2(x, fake_relu=fake_relu)
        x = self.layer2(x)
        all_outputs["layer2"] = x

        if fake_relu and with_latent:
            all_outputs["layer3_fake_relu"] = self.layer3(x, fake_relu=fake_relu)
        x = self.layer3(x)
        all_outputs["layer3"] = x

        if fake_relu and with_latent:
            all_outputs["layer4_fake_relu"] = self.layer4(x, fake_relu=fake_relu)
        x = self.layer4(x)
        all_outputs["layer4"] = x

        x = self.avgpool(x)
        all_outputs["avgpool"] = x

        pre_out = x.view(x.size(0), -1)
        final = self.fc(pre_out)
        all_outputs["final"] = final

        if with_latent:
            return final, pre_out, all_outputs
        return final


def resnet50_classifier() -> RobustResNetClassifier:
    return RobustResNetClassifier(_Bottleneck, [3, 4, 6, 3])
