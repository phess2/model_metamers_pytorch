import os
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets import get_vision_dataset


class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/rphess/orcd/datasets/imagenet/images_complete/ilsvrc/",
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.image_size = image_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            if (self.train_dataset is None) or (self.val_dataset is None):
                train_dataset, val_dataset = get_vision_dataset(
                    "imagenet", self.data_dir, self.image_size, stage
                )
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset
        elif stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = get_vision_dataset(
                    "imagenet", self.data_dir, self.image_size, stage
                )
        elif stage == "test":
            if self.test_dataset is None:
                self.test_dataset = get_vision_dataset(
                    "imagenet", self.data_dir, self.image_size, "validate"
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @property
    def num_classes(self) -> int:
        if self.train_dataset is None:
            self.setup(stage="fit")
        return len(self.train_dataset.classes)


def get_datamodule(config: dict) -> LightningDataModule:
    """
    Build a ``LightningDataModule`` from a training config dict.

    The config is expected to have a ``data_settings`` section.  The
    ``dataset`` key (default ``"imagenet"``) selects the data module class.
    Remaining keys are forwarded as constructor kwargs.

    Extend this function when adding new datasets (e.g. audio).
    """
    data_settings = config.get("data_settings", {})
    dataset_name = data_settings.get("dataset", "imagenet")

    # Keys accepted by ImageNetDataModule
    _IMAGENET_KEYS = {
        "data_dir",
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "image_size",
    }

    if dataset_name == "imagenet":
        kwargs = {k: v for k, v in data_settings.items() if k in _IMAGENET_KEYS}
        return ImageNetDataModule(**kwargs)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported")
