"""
Generic Lightning training module for Lipschitz-constrained models.

``LipsLightningModule`` wraps any ``LipsModel`` and handles:
  - manual optimisation (single or multi-optimizer)
  - Lipschitz weight projection after each optimiser step
  - per-layer norm-change ratio tracking & logging
  - validation / test metrics (top-k accuracy, Lipschitz bound)
  - LR scheduler stepping

New model architectures (vision, audio, ...) only need to subclass
``LipsModel`` -- all training orchestration lives here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from src.models.base import LipsModel
from src.optimizers.optimizers import configure_optimizers as _configure_optimizers
from src.training.metrics import topk_accuracy


class LipsLightningModule(LightningModule):
    """
    Wraps a :class:`LipsModel` with Lightning training logic.

    Parameters
    ----------
    model : LipsModel
        The architecture to train.
    config : dict
        Full training config (must contain ``optim_settings`` and ``hparams``).
    loss_fn : nn.Module, optional
        Loss function.  Defaults to ``nn.CrossEntropyLoss()``.
    """

    def __init__(
        self,
        model: LipsModel,
        config: dict,
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.model = model
        self.config = config
        self.optim_settings = config["optim_settings"]
        self.hparam_config = config.get("hparams", {})
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        # Save config (but not the model -- it would duplicate parameters)
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        # Per-layer norm-change ratio accumulators
        self._norm_ratio_sums: Dict[str, float] = defaultdict(float)
        self._norm_ratio_counts: Dict[str, int] = defaultdict(int)

        # Accumulator for test step outputs
        self._test_step_outputs: list = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, **kwargs)

    # ------------------------------------------------------------------
    # Loss (override for multi-task or custom losses)
    # ------------------------------------------------------------------

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the training/validation loss.  Override for custom losses."""
        return self.loss_fn(output, target)

    # ------------------------------------------------------------------
    # Optimiser configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Delegate to ``src.optimizers.optimizers.configure_optimizers``."""
        return _configure_optimizers(self.optim_settings, self.model)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.compute_loss(output, target)

        # Manual optimiser step (handles single & multi-optimizer)
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for opt in optimizers:
            opt.zero_grad()
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()

        # Lipschitz projection
        ratios = self.model.project_weights()
        for name, ratio in ratios.items():
            self._norm_ratio_sums[name] += ratio
            self._norm_ratio_counts[name] += 1

        # LR scheduler step
        self._step_schedulers()

        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )
        return {"train_loss": loss}

    def _step_schedulers(self):
        """Step all LR schedulers (handles nested lists / dicts)."""
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for sch in schedulers:
            if isinstance(sch, dict):
                sch = sch.get("scheduler")
            if sch is not None:
                sch.step()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.compute_loss(output, target)

        acc = topk_accuracy(output, target, topk=(1, 5))
        top1_acc = acc["top1_acc"]
        top5_acc = acc["top5_acc"]

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/accuracy", top1_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/top1_acc", top1_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/top5_acc", top5_acc, on_step=False, on_epoch=True, logger=True)
        self.log(
            "val/lips_bound", self.model.get_lips_bound(),
            on_step=False, on_epoch=True, logger=True,
        )
        return {"val_loss": loss, "val_accuracy": top1_acc, "val_top5_acc": top5_acc}

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        loss = self.compute_loss(output, target)

        self._test_step_outputs.append({
            "test_loss": loss,
            "correct": correct,
            "num_samples": target.shape[0],
        })

        self.log("test/loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"test_loss": loss, "correct": correct, "num_samples": target.shape[0]}

    def on_test_epoch_end(self):
        if not self._test_step_outputs:
            return
        avg_loss = torch.stack([x["test_loss"] for x in self._test_step_outputs]).mean()
        total_correct = sum(x["correct"] for x in self._test_step_outputs)
        total_samples = sum(x["num_samples"] for x in self._test_step_outputs)
        test_acc = total_correct / total_samples if total_samples > 0 else 0.0

        self.log("test/loss_epoch", avg_loss, logger=True)
        self.log("test/accuracy", test_acc, logger=True, prog_bar=True)
        self._test_step_outputs.clear()

    # ------------------------------------------------------------------
    # Epoch-end hooks
    # ------------------------------------------------------------------

    def on_train_epoch_end(self):
        """Log per-layer norm-change ratio metrics."""
        for name in list(self._norm_ratio_sums.keys()):
            count = self._norm_ratio_counts[name]
            if count > 0:
                avg_ratio = self._norm_ratio_sums[name] / count
                self.log(
                    f"train/norm_ratio/{name}", avg_ratio,
                    on_step=False, on_epoch=True, logger=True,
                )
            self._norm_ratio_sums[name] = 0.0
            self._norm_ratio_counts[name] = 0
