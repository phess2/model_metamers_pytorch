from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from src.data.dataloaders import ImageNetDataModule
from src.optimizers.optimizers import set_up_conv_optimizer

from .layers.custom_modules import FakeReLU, SequentialWithArgs
from .layers.LipsLayers import LipsConv2d, LipsLinear


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
    ):
        super().__init__()
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

    def forward(self, x, fake_relu=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual addition is not Lipschitz-bounded in this implementation.
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
        super().__init__()
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

        # Residual addition is not Lipschitz-bounded in this implementation.
        out += identity

        if fake_relu and self.last_block:
            return FakeReLU.apply(out)

        return self.relu(out)


class LipsResNetModule(LightningModule):
    def __init__(self, config: dict):
        super().__init__()
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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_sizes = self.hparam_config["layer_sizes"]
        self.block_type = self.hparam_config["block_type"]
        if self.block_type in ("basic", "resnet_block"):
            block_class = LipsBasicBlock
        elif self.block_type == "bottleneck":
            block_class = LipsBottleneck
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")

        self.layer1 = self._make_layer(block_class, 64, self.layer_sizes[0])
        self.layer2 = self._make_layer(block_class, 128, self.layer_sizes[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, self.layer_sizes[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, self.layer_sizes[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LipsLinear(
            512 * block_class.expansion,
            self.num_classes,
            w_max=self.w_max,
            projection=self.projection,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.hparam_config.get("zero_init_residual", False):
            for m in self.modules():
                if isinstance(m, LipsBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, LipsBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.test_step_outputs = []

        # Initialize accumulators for per-layer norm-change ratio metrics
        # These track how much the projection step modifies each layer's weights
        # Use defaultdict for dynamic layer names from named_modules()
        self.norm_ratio_sums = defaultdict(float)
        self.norm_ratio_counts = defaultdict(int)

    def __str__(self):
        return (
            f"LipsResNetModule(num_classes={self.num_classes}, "
            f"w_max={self.w_max}, projection={self.projection})"
        )

    def _make_layer(self, block, planes, blocks, stride=1):
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
                )
            )

        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        del no_relu
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

    def configure_optimizers(self):
        linear_parameters = list(self.fc.parameters())
        conv_parameters = [
            param
            for name, param in self.named_parameters()
            if not name.startswith("fc.")
        ]
        return set_up_conv_optimizer(
            self.optim_settings, conv_parameters, linear_parameters
        )

    def _setup_datamodule(self, stage: Optional[str] = None):
        if self.data_module is not None:
            self.data_module.setup(stage=stage)

    def train_dataloader(self):
        self._setup_datamodule(stage="fit")
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        self._setup_datamodule(stage="validate")
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        self._setup_datamodule(stage="test")
        return self.data_module.test_dataloader()

    def project_step(self):
        """
        Project weights to enforce Lipschitz constraints and accumulate
        per-layer norm-change ratio metrics.
        """
        if self.projection is None:
            return
        for name, module in self.named_modules():
            if isinstance(module, (LipsConv2d, LipsLinear)):
                ratio = module.project_()
                self.norm_ratio_sums[name] += float(ratio)
                self.norm_ratio_counts[name] += 1

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)
        if self.optim_settings["mult_optimizers"]:
            opt_1, opt_2 = self.optimizers()
            opt_1.zero_grad()
            opt_2.zero_grad()
            self.manual_backward(loss)
            opt_1.step()
            opt_2.step()
        else:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        self.project_step()

        def _step_scheduler(scheduler_obj):
            if scheduler_obj is None:
                return
            if isinstance(scheduler_obj, dict):
                scheduler_obj = scheduler_obj.get("scheduler")
            if isinstance(scheduler_obj, (list, tuple)):
                for sch in scheduler_obj:
                    _step_scheduler(sch)
                return
            scheduler_obj.step()

        schedulers = self.lr_schedulers()
        _step_scheduler(schedulers)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"train_loss": loss}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)

        # Compute top-1 and top-5 accuracy
        batch_size = target.size(0)
        _, pred = output.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()  # Shape: (5, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Top-1: check only the first prediction
        top1_correct = correct[0].sum().item()
        # Top-5: check if any of the top 5 predictions match
        top5_correct = correct.any(dim=0).sum().item()

        top1_acc = top1_correct / batch_size
        top5_acc = top5_correct / batch_size

        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val/accuracy",
            top1_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/top1_acc",
            top1_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/top5_acc",
            top5_acc,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val/lips_bound",
            self.get_lips_bound(),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return {"val_loss": loss, "val_accuracy": top1_acc, "val_top5_acc": top5_acc}

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        loss = F.cross_entropy(output, target)

        self.test_step_outputs.append(
            {
                "test_loss": loss,
                "correct": correct,
                "num_samples": target.shape[0],
            }
        )

        self.log("test/loss", loss, on_step=False, on_epoch=True, logger=True)
        return {
            "test_loss": loss,
            "correct": correct,
            "num_samples": target.shape[0],
        }

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            avg_test_loss = torch.stack(
                [x["test_loss"] for x in self.test_step_outputs]
            ).mean()
            total_correct = sum([x["correct"] for x in self.test_step_outputs])
            total_samples = sum([x["num_samples"] for x in self.test_step_outputs])
            test_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

            self.log("test/loss_epoch", avg_test_loss, logger=True)
            self.log("test/accuracy", test_accuracy, logger=True, prog_bar=True)
            self.test_step_outputs.clear()

    def on_train_epoch_end(self):
        """
        Log per-layer norm-change ratio metrics at end of each training epoch.
        These metrics show how much the Lipschitz projection step modified each layer.
        """
        for name in list(self.norm_ratio_sums.keys()):
            count = self.norm_ratio_counts[name]
            if count > 0:
                avg_ratio = self.norm_ratio_sums[name] / count
                self.log(
                    f"train/norm_ratio/{name}",
                    avg_ratio,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )
            # Reset accumulators for next epoch
            self.norm_ratio_sums[name] = 0.0
            self.norm_ratio_counts[name] = 0

    def get_lips_bound(self):
        """
        Gets the Lipschitz bound of the model.
        """
        lips_bound = 1.0
        for module in self.modules():
            if module is self:
                continue
            bound_fn = getattr(module, "get_lips_bound", None)
            if callable(bound_fn):
                lips_bound *= bound_fn()
        return lips_bound
