"""
Training script for Lipschitz-constrained vision models.

Uses the modular ``src`` infrastructure:
  - ``src.utils.config``       -- config loading & CLI overrides
  - ``src.models.registry``    -- model factory
  - ``src.data.dataloaders``   -- data-module factory
  - ``src.training.module``    -- generic Lightning training module
  - ``src.training.losses``    -- loss function factory
"""

import os
import pathlib
import re
import socket
import warnings
from argparse import ArgumentParser

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint  # , EarlyStopping
from lightning.pytorch.loggers.wandb import WandbLogger

from src.data.dataloaders import get_datamodule
from src.models.registry import get_model
from src.training.losses import get_loss_fn
from src.training.module import LipsLightningModule
from src.utils.config import apply_cli_overrides, load_config

torch.set_float32_matmul_precision("medium")

warnings.filterwarnings(
    "ignore",
    message=".*resuming from a checkpoint that ended before the epoch ended.*",
    category=UserWarning,
)

hostname = socket.gethostname()


def run_train(args: ArgumentParser):
    seed_everything(args.seed)

    args.exp_dir = pathlib.Path(args.exp_dir)
    config_path = pathlib.Path(args.config)
    print(f"Loading config from {config_path}")

    config = load_config(config_path)
    apply_cli_overrides(config, args)

    model = get_model(config)
    print(f"Model: {model}")

    datamodule = get_datamodule(config)

    loss_fn = get_loss_fn(config)

    module = LipsLightningModule(model, config, loss_fn=loss_fn)

    exp_root = args.exp_dir / config_path.stem
    checkpoint_dir = exp_root / "checkpoints"
    if not args.no_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = None
    if args.resume_training:
        if args.ckpt_path:
            ckpt_path = args.ckpt_path
        else:
            # Prefer last.ckpt (always-updated resume target)
            last_ckpt = checkpoint_dir / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
            else:
                # Fallback: latest checkpoint by creation time
                ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
                if ckpt_paths:
                    ckpt_path = str(ckpt_paths[-1])
        if ckpt_path:
            print(f"Resuming from checkpoint: {ckpt_path}")
        else:
            print(
                "WARNING: --resume_training set but no checkpoint found. "
                "Starting fresh."
            )

    wandb_id_file = exp_root / "wandb_run_id.txt"

    if args.resume_training and wandb_id_file.exists():
        wandb_run_id = wandb_id_file.read_text().strip()
        wandb_logger = WandbLogger(
            project="LipsVision",
            log_model=False,
            id=wandb_run_id,
            resume="must",
        )
    else:
        if args.lr is not None:
            lr_sci = f"{args.lr:.0e}"
            lr_formatted = re.sub(r"e([+-])0+(\d)", r"e\1\2", lr_sci)
            wandb_name = f"{config_path.stem}_lr{lr_formatted}"
        else:
            wandb_name = config_path.stem
        wandb_logger = WandbLogger(
            project="LipsVision",
            log_model=False,
            name=wandb_name,
        )
        # Persist the run ID so future --resume_training picks it up.
        exp_root.mkdir(parents=True, exist_ok=True)
        wandb_id_file.write_text(wandb_logger.experiment.id)
    # early_stop_callback = EarlyStopping(
    #     monitor="val/loss", mode="min", patience=10,
    # )
    if args.no_checkpoints:
        # callbacks = [early_stop_callback]
        callbacks = []
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            dirpath=checkpoint_dir,
            filename="epoch{epoch:02d}-val_loss{val/loss:.4f}",
            auto_insert_metric_name=False,
        )
        # callbacks = [checkpoint_callback, early_stop_callback]
        callbacks = [checkpoint_callback]

    hparams = config.get("hparams", {})
    trainer = Trainer(
        precision="32",
        default_root_dir=exp_root,
        max_epochs=hparams.get("max_epochs", 20),
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp" if args.gpus > 1 else "auto",
        limit_val_batches=hparams.get("limit_val_batches", 1.0),
        limit_train_batches=hparams.get("limit_train_batches", 1.0),
        val_check_interval=hparams.get("val_check_interval", 1.0),
        gradient_clip_val=hparams.get("gradient_clip_val", None),
        gradient_clip_algorithm=hparams.get("gradient_clip_algorithm", "value"),
        accumulate_grad_batches=hparams.get("accumulate_grad_batches", 1),
        profiler=hparams.get("profiler", None),
        callbacks=callbacks,
        enable_checkpointing=not args.no_checkpoints,
        logger=wandb_logger,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/lipsalexnet_test.json")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable checkpoint saving (e.g. for LR sweeps to avoid checkpoint I/O)",
    )
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
