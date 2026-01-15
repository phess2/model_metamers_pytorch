import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from typing import Optional

from src.data.dataloaders import ImageNetDataModule

from .layers.custom_modules import FakeReLUM, FakeReLU, SequentialWithArgs
from .layers.LipsLayers import LipsConv2d, LipsLinear


def conv3x3(in_planes, out_planes, stride=1, w_max=1.0, projection=None):
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
    return LipsConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        w_max=w_max,
        projection=projection,
    )


class LipsResNetBlock(nn.Module):
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
    ):
        super(LipsResNetBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, w_max, projection)
        self.bn1 = nn.BatchNorm2d(
            planes
        )  # TODO: Look into whether batchnorm is lipschitz bounded...
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, w_max, projection)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last_block = last_block

    def forward(self, x, fake_relu=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # TODO: Make residual connection lipschitz bounded...
        out += identity

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
    ):
        super(LipsBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride, w_max, projection)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, w_max, projection)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, w_max, projection)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.last_block = last_block

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

        out += identity

        if fake_relu and self.last_block:
            return FakeReLU.apply(out)

        return self.relu(out)


class LipsResNetModule(LightningModule):
    def __init__(self, config: dict):
        super(LipsResNetModule, self).__init__()
        self.automatic_optimization = False
        self.hparam_config = config["hparams"]
        self.optim_settings = config["optim_settings"]
        self.num_classes = self.hparam_config["num_classes"]
        self.w_max = self.hparam_config["w_max"]
        self.projection = self.hparam_config["projection"]
        self.save_hyperparameters()

        data_settings = config.get("data_settings", {})
        allowed_data_keys = {
            "data_dir",
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "image_size",
        }
        datamodule_kwargs = {
            key: value
            for key, value in data_settings.items()
            if key in allowed_data_keys
        }
        self.data_module = ImageNetDataModule(**datamodule_kwargs)

        self.inplanes = 64
        self.conv1 = LipsConv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            w_max=self.w_max,
            projection=self.projection,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        #TODO: Look into whether maxpool is lipschitz bounded...
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_sizes = config["hparams"]["layer_sizes"]
        self.block_type = config["hparams"]["block_type"]
        if self.block_type == "resnet_block":
            block_class = LipsResNetBlock
        elif self.block_type == "bottleneck":
            block_class = LipsBottleneck
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")
        self.layer1 = self._make_layer(block_class, 64, self.layer_sizes[0])
        self.layer2 = self._make_layer(block_class, 128, self.layer_sizes[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, self.layer_sizes[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, self.layer_sizes[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_class.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if config["hparams"]["zero_init_residual"]:
            for m in self.modules():
                if isinstance(m, LipsBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, LipsResNetBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, w_max=self.w_max, projection=self.projection),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, w_max=self.w_max, projection=self.projection))
        self.inplanes = planes * block.expansion
        for b in range(1, blocks):
            if b==(blocks-1):
                layers.append(block(self.inplanes, planes, last_block=True, w_max=self.w_max, projection=self.projection))
            else:
                layers.append(block(self.inplanes, planes, w_max=self.w_max, projection=self.projection))

        return SequentialWithArgs(*layers)