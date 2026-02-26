from typing import Optional

from torch import nn

from ..base import LipsModel
from ..layers.custom_modules import FakeReLU, SequentialWithArgs
from ..layers.LipsLayers import LipsConv2d, LipsLinear


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------


def conv3x3(in_planes, out_planes, stride=1, w_max=1.0, projection=None):
    """3x3 Lipschitz convolution with padding."""
    return LipsConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        w_max=w_max,
        projection=projection,
    )


def conv1x1(in_planes, out_planes, stride=1, w_max=1.0, projection=None):
    """1x1 Lipschitz convolution."""
    return LipsConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        w_max=w_max,
        projection=projection,
    )


def _lips_product(module):
    """Product of Lipschitz bounds for Lips layers inside ``module``."""
    if module is None:
        return 1.0

    bound = 1.0
    for submodule in module.modules():
        if isinstance(submodule, (LipsConv2d, LipsLinear)):
            bound *= float(submodule.get_lips_bound())
    return bound


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------


class LipsBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        last_block=False,
        w_max=1.0,
        projection=None,
        total_residual_connections=1,
    ):
        super().__init__()
        if total_residual_connections <= 0:
            raise ValueError("total_residual_connections must be a positive integer.")
        self.conv1 = conv3x3(
            inplanes, planes, stride=stride, w_max=w_max, projection=projection
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, w_max=w_max, projection=projection)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last_block = last_block
        self.skip_scale = (total_residual_connections - 1) / total_residual_connections
        self.residual_scale = 1 / total_residual_connections

    def get_lips_bound(self):
        """
        Upper bound for a scaled residual block:
        ``L(a * skip + b * main) <= a * L(skip) + b * L(main)``.
        """
        main_bound = _lips_product(self.conv1) * _lips_product(self.conv2)
        skip_bound = 1.0 if self.downsample is None else _lips_product(self.downsample)
        return self.skip_scale * skip_bound + self.residual_scale * main_bound

    def forward(self, x, fake_relu=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_scale * identity + self.residual_scale * out

        if fake_relu and self.last_block:
            return FakeReLU.apply(out)
        return self.relu(out)


class LipsBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        last_block=False,
        w_max=1.0,
        projection=None,
        total_residual_connections=1,
    ):
        super().__init__()
        if total_residual_connections <= 0:
            raise ValueError("total_residual_connections must be a positive integer.")
        self.conv1 = conv1x1(inplanes, planes, w_max=w_max, projection=projection)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(
            planes, planes, stride=stride, w_max=w_max, projection=projection
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(
            planes, planes * self.expansion, w_max=w_max, projection=projection
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.last_block = last_block
        self.skip_scale = (total_residual_connections - 1) / total_residual_connections
        self.residual_scale = 1 / total_residual_connections

    def get_lips_bound(self):
        """
        Upper bound for a scaled residual block:
        ``L(a * skip + b * main) <= a * L(skip) + b * L(main)``.
        """
        main_bound = (
            _lips_product(self.conv1)
            * _lips_product(self.conv2)
            * _lips_product(self.conv3)
        )
        skip_bound = 1.0 if self.downsample is None else _lips_product(self.downsample)
        return self.skip_scale * skip_bound + self.residual_scale * main_bound

    def forward(self, x, fake_relu=False):
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

        out = self.skip_scale * identity + self.residual_scale * out

        if fake_relu and self.last_block:
            return FakeReLU.apply(out)
        return self.relu(out)


# ---------------------------------------------------------------------------
# Full ResNet model
# ---------------------------------------------------------------------------


class LipsResNet(LipsModel):
    """
    Lipschitz-constrained ResNet architecture.

    Pure ``nn.Module`` -- contains no Lightning, data-loading, or training
    logic.  Training orchestration is handled by ``LipsLightningModule``.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        w_max: float = 1.0,
        projection: Optional[str] = None,
        layer_sizes: Optional[list[int]] = None,
        block_type: str = "basic",
        zero_init_residual: bool = False,
    ):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [2, 2, 2, 2]
        if len(layer_sizes) != 4:
            raise ValueError(
                f"layer_sizes must define 4 stages; got {len(layer_sizes)} entries."
            )
        if any(stage_size <= 0 for stage_size in layer_sizes):
            raise ValueError("layer_sizes entries must all be positive integers.")

        self.num_classes = num_classes
        self.w_max = w_max
        self.projection = projection
        self.layer_sizes = layer_sizes
        self.block_type = block_type
        self.total_residual_connections = sum(self.layer_sizes)

        if block_type in ("basic", "resnet_block"):
            block_class = LipsBasicBlock
        elif block_type == "bottleneck":
            block_class = LipsBottleneck
        else:
            raise ValueError(f"Invalid block type: {block_type}")

        self.inplanes = 64
        self.conv1 = LipsConv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            w_max=w_max,
            projection=projection,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block_class,
            64,
            layer_sizes[0],
            total_residual_connections=self.total_residual_connections,
        )
        self.layer2 = self._make_layer(
            block_class,
            128,
            layer_sizes[1],
            stride=2,
            total_residual_connections=self.total_residual_connections,
        )
        self.layer3 = self._make_layer(
            block_class,
            256,
            layer_sizes[2],
            stride=2,
            total_residual_connections=self.total_residual_connections,
        )
        self.layer4 = self._make_layer(
            block_class,
            512,
            layer_sizes[3],
            stride=2,
            total_residual_connections=self.total_residual_connections,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LipsLinear(
            512 * block_class.expansion,
            num_classes,
            w_max=w_max,
            projection=projection,
        )

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LipsBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, LipsBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.metamer_layers = [
            "conv1_relu1",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "final",
            # fake_relu variants
            "conv1_relu1_fake_relu",
            "layer1_fake_relu",
            "layer2_fake_relu",
            "layer3_fake_relu",
            "layer4_fake_relu",
        ]

    def __str__(self):
        return (
            f"LipsResNet(num_classes={self.num_classes}, "
            f"w_max={self.w_max}, projection={self.projection})"
        )

    def get_lips_bound(self):
        """
        Residual-aware model bound.

        Uses the same convention as the base class for non-Lips modules
        (BN/ReLU/pooling omitted), but composes residual blocks as
        ``L(skip) + L(main)`` instead of flattening all Lips layers into one
        sequential product.
        """
        bound = _lips_product(self.conv1)

        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for stage_block in stage._modules.values():
                if isinstance(stage_block, (LipsBasicBlock, LipsBottleneck)):
                    bound *= float(stage_block.get_lips_bound())

        bound *= _lips_product(self.fc)
        return bound

    def forward_with_representations(self, x, fake_relu=False):
        """Return ``(logits, all_outputs)`` with all intermediate activations."""
        final, _pre_out, all_outputs = self.forward(
            x, with_latent=True, fake_relu=fake_relu
        )
        return final, all_outputs

    def _make_layer(
        self, block, planes, blocks, stride=1, total_residual_connections=1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride=stride,
                    w_max=self.w_max,
                    projection=self.projection,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                w_max=self.w_max,
                projection=self.projection,
                total_residual_connections=total_residual_connections,
            )
        )
        self.inplanes = planes * block.expansion
        for b in range(1, blocks):
            is_last = b == (blocks - 1)
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    last_block=is_last,
                    w_max=self.w_max,
                    projection=self.projection,
                    total_residual_connections=total_residual_connections,
                )
            )
        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        all_outputs = {}
        all_outputs["input_after_preproc"] = x

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
