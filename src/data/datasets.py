from pathlib import Path
from typing import Callable, Optional, Union

from torchvision import datasets

from src.data.transforms import get_vision_transform


class ImageNetFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None,
    ):
        root_path = Path(root)
        if split not in ["train", "val"]:
            raise ValueError(f"Split must be either 'train' or 'val', got {split}")

        split_dir = root_path / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory {split_dir} not found")

        self.dataset = datasets.ImageFolder(split_dir, transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx


def get_vision_dataset(
    dataset_name: str, root_dir: Union[str, Path], image_size: int, stage: str
) -> datasets.ImageFolder:
    if dataset_name == "imagenet":
        train_transform, val_transform = get_vision_transform(dataset_name, image_size)
        if stage == 'validate':
            return ImageNetFolder(root_dir, "val", val_transform)
        else:
            return ImageNetFolder(root_dir, "train", train_transform), ImageNetFolder(
                root_dir, "val", val_transform
            )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
