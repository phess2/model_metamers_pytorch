import json
import os
import pathlib
import re
import socket
from argparse import ArgumentParser

import torch
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers.wandb import WandbLogger

from src.models.lipsvision import get_module


# Use only the new API; do not set torch.backends.cuda/cudnn.allow_tf32 (legacy mix causes RuntimeError).
torch.set_float32_matmul_precision("medium")

hostname = socket.gethostname()


def run_train(args: ArgumentParser):
    seed_everything(args.seed)

    args.exp_dir = pathlib.Path(args.exp_dir)
    config_path = args.config
    print(f"Loading config from {config_path}")

    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    module = get_module(config)

    config["data_settings"]["num_workers"] = args.num_workers
    config["ngpus"] = args.gpus

    # Override learning rate if provided via command line
    if args.lr is not None:
        if config["optim_settings"].get("mult_optimizers", False):
            # Handle multi-optimizer case
            if "linear_optimizer" in config["optim_settings"]:
                config["optim_settings"]["linear_optimizer"]["kwargs"]["lr"] = args.lr
            if "conv_optimizer" in config["optim_settings"]:
                config["optim_settings"]["conv_optimizer"]["kwargs"]["lr"] = args.lr
        else:
            # Single optimizer case
            config["optim_settings"]["optimizer"]["kwargs"]["lr"] = args.lr

    config_path = pathlib.Path(config_path)

    exp_root = args.exp_dir / config_path.stem
    checkpoint_dir = exp_root / "checkpoints"
    if not args.no_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_paths = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getctime)
    else:
        ckpt_paths = []

    ckpt_path = None
    if args.resume_training:
        if args.ckpt_path:
            ckpt_path = args.ckpt_path
        elif ckpt_paths:
            ckpt_path = str(ckpt_paths[-1])

    module = module(config)
    print(f"Module: {module.__str__()}")
    run_idx = len(ckpt_paths)

    # Build wandb run name
    if args.lr is not None:
        # Format LR to avoid scientific notation in name (e.g., 0.0001 -> lr1e-4)
        # Remove leading zeros from exponent (e-04 -> e-4, e+01 -> e+1)
        lr_sci = f"{args.lr:.0e}"
        lr_formatted = re.sub(r"e([+-])0+(\d)", r"e\1\2", lr_sci)
        lr_str = f"lr{lr_formatted}"
        wandb_name = f"{config_path.stem}_{lr_str}_{run_idx}"
    else:
        wandb_name = f"{config_path.stem}_{run_idx}"

    # Define loggers and callbacks
    wandb_logger = WandbLogger(
        project="LipsVision",
        log_model=False,
        name=wandb_name,
    )
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=10,
    )
    if args.no_checkpoints:
        callbacks = [early_stop_callback]
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            dirpath=checkpoint_dir,
            filename="{epoch:02d}-{val/loss:.4f}",
        )
        callbacks = [checkpoint_callback, early_stop_callback]

    trainer = Trainer(
        precision="32",
        default_root_dir=exp_root,
        max_epochs=config["hparams"]["max_epochs"],
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp" if args.gpus > 1 else "auto",
        limit_val_batches=config["hparams"].get("limit_val_batches", 1.0),
        limit_train_batches=config["hparams"].get("limit_train_batches", 1.0),
        val_check_interval=config["hparams"].get("val_check_interval", 1.0),
        gradient_clip_val=config["hparams"].get("gradient_clip_val", None),
        gradient_clip_algorithm=config["hparams"].get(
            "gradient_clip_algorithm", "value"
        ),
        accumulate_grad_batches=config["hparams"].get("accumulate_grad_batches", 1),
        profiler=config["hparams"].get("profiler", None),
        callbacks=callbacks,
        enable_checkpointing=not args.no_checkpoints,
        logger=wandb_logger,
    )
    trainer.fit(module, ckpt_path=ckpt_path)


def main():
    args = ArgumentParser()
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--config", type=str, default="configs/lipsalexnet_test.json")
    args.add_argument("--num_workers", type=int, default=1)
    args.add_argument("--gpus", type=int, default=1)
    args.add_argument("--exp_dir", type=str, default="experiments")
    args.add_argument("--resume_training", action="store_true")
    args.add_argument("--ckpt_path", type=str, default="")
    args.add_argument("--num_nodes", type=int, default=1)
    args.add_argument(
        "--lr", type=float, default=None, help="Override learning rate from config"
    )
    args.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable checkpoint saving (e.g. for LR sweeps to avoid checkpoint I/O)",
    )
    args = args.parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
