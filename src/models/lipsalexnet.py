import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from typing import Optional

from src.data.dataloaders import ImageNetDataModule

from .layers.custom_modules import FakeReLUM
from .layers.LipsLayers import LipsConv2d, LipsLinear

from src.optimizers.optimizers import set_up_conv_optimizer

class LipsAlexNetModule(LightningModule):
    def __init__(self, config: dict):
        super(LipsAlexNetModule, self).__init__()
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
        self.model = nn.Sequential(
            LipsConv2d(
                3,
                64,
                kernel_size=11,
                w_max=self.w_max,
                stride=4,
                padding=2,
                projection=self.projection,
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LipsConv2d(
                64,
                192,
                kernel_size=5,
                w_max=self.w_max,
                padding=2,
                projection=self.projection,
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LipsConv2d(
                192,
                384,
                kernel_size=3,
                w_max=self.w_max,
                padding=1,
                projection=self.projection,
            ),
            nn.ReLU(inplace=False),
            LipsConv2d(
                384,
                256,
                kernel_size=3,
                w_max=self.w_max,
                padding=1,
                projection=self.projection,
            ),
            nn.ReLU(inplace=False),
            LipsConv2d(
                256,
                256,
                kernel_size=3,
                w_max=self.w_max,
                padding=1,
                projection=self.projection,
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.featurenames = [
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

        # TODO: Figure out what the code below is doing for Jenelle's code
        self.fake_relu_dict = nn.ModuleDict()
        for layer_name in self.featurenames:
            if "relu" in layer_name:
                self.fake_relu_dict[layer_name] = FakeReLUM()

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            LipsLinear(256 * 6 * 6, 4096, w_max=self.w_max, projection=self.projection),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            LipsLinear(4096, 4096, w_max=self.w_max, projection=self.projection),
            nn.ReLU(inplace=False),
            LipsLinear(
                4096, self.num_classes, w_max=self.w_max, projection=self.projection
            ),
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
        # self.project_dict = config["project_dict"]

        self.test_step_outputs = []

    def __str__(self):
        return f"LipsAlexNetModule(num_classes={self.num_classes}, w_max={self.w_max}, projection={self.projection})"

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers for Lightning manual optimization.
        """
        optim_settings = self.optim_settings
        linear_parameters = self.classifier.parameters()
        conv_parameters = self.model.parameters()
        result = set_up_conv_optimizer(optim_settings, conv_parameters, linear_parameters)
        return result

    def forward(
        self, x: torch.Tensor, fake_relu: bool = False, return_all_outputs: bool = False
    ) -> torch.Tensor:
        all_outputs = {}
        all_outputs["input_after_preproc"] = x

        for layer, name in list(zip(self.model, self.featurenames)):
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

        for layer, name in list(zip(self.classifier, self.classifier_names)):
            if ("relu" in name) and fake_relu:
                all_outputs[name + "_fake_relu"] = self.fake_relu_dict[name](x)
            x = layer(x)
            if return_all_outputs:
                all_outputs[name] = x

        if return_all_outputs:
            all_outputs["final"] = all_outputs["fctop"]

        if return_all_outputs:
            return x, all_outputs
        return x

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
        if self.projection is not None:
            for layer, name in list(zip(self.model, self.featurenames)):
                if "conv" in name:
                    layer.project_()
            for layer, name in list(zip(self.classifier, self.classifier_names)):
                if name.startswith("fc") and "relu" not in name:
                    layer.project_()

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
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / target.size(0)
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val/accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/lips_bound",
            self.get_lips_bound(),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return {"val_loss": loss, "val_accuracy": accuracy}

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

    def get_lips_bound(self):
        """
        Gets the Lipschitz bound of the model
        """
        lips_bound = 1.0
        for layer, name in list(zip(self.model, self.featurenames)):
            if "conv" in name:
                lips_bound *= layer.get_lips_bound()
        for layer, name in list(zip(self.classifier, self.classifier_names)):
            if name.startswith("fc") and "relu" not in name:
                lips_bound *= layer.get_lips_bound()
        return lips_bound
