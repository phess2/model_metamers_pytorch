from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from PIL import Image
from torchvision import datasets

from src.data.imagenet_legacy_split import (
    LEGACY_400_16_CLASS_ORDER,
    LEGACY_400_16_EXCLUDED_VAL_PATHS,
    LEGACY_400_16_VALIDATION_PATHS,
)
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


def _legacy_path_to_dataset_relpath(path_from_validation_paths: str) -> str:
    """Convert legacy ilsvrc/val paths to ImageFolder relpaths: <wnid>/<filename>."""
    parts = Path(path_from_validation_paths).parts
    if len(parts) < 4 or parts[0] != "ilsvrc" or parts[1] != "val":
        raise ValueError(
            f"Unexpected validation path format '{path_from_validation_paths}'. "
            "Expected 'ilsvrc/val/<wnid>/<filename>'."
        )
    return str(Path(parts[2]) / parts[3])


def _build_legacy_400_16_val_relpaths(data_dir: Path) -> list[str]:
    """Reconstruct the legacy curated 400-image ImageNet validation pool."""
    rng = np.random.RandomState(517)
    selected_relpaths: list[str] = []
    target_per_class = 25

    for class_name in LEGACY_400_16_CLASS_ORDER:
        all_candidates = LEGACY_400_16_VALIDATION_PATHS[class_name]
        perm = rng.permutation(len(all_candidates))
        class_total = 0
        check_idx = 0

        while class_total < target_per_class:
            if check_idx >= len(perm):
                raise RuntimeError(
                    f"Could not select {target_per_class} images for class '{class_name}' "
                    "with legacy filtering constraints."
                )

            candidate_legacy_path = all_candidates[perm[check_idx]]
            check_idx += 1

            if candidate_legacy_path in LEGACY_400_16_EXCLUDED_VAL_PATHS:
                continue

            dataset_rel = _legacy_path_to_dataset_relpath(candidate_legacy_path)
            candidate_abs = data_dir / "val" / dataset_rel
            if not candidate_abs.is_file():
                raise FileNotFoundError(
                    "Missing ImageNet validation image for legacy subset reconstruction: "
                    f"{candidate_abs}. Expected ImageNet layout under data_dir/val/<wnid>/<image>.JPEG."
                )

            with Image.open(candidate_abs) as img_pil:
                if img_pil.mode != "RGB":
                    continue
                if min(img_pil.size) < 224:
                    continue

            selected_relpaths.append(dataset_rel)
            class_total += 1

    return selected_relpaths


def _build_dataset_relpath_to_index(dataset) -> dict[str, int]:
    """Map ImageFolder relpath (<wnid>/<filename>) to dataset index."""
    try:
        samples = dataset.dataset.samples
    except AttributeError as exc:
        raise ValueError(
            "Expected torchvision ImageFolder-style dataset with '.dataset.samples'."
        ) from exc

    relpath_to_idx: dict[str, int] = {}
    for idx, (sample_path, _) in enumerate(samples):
        sample_path_obj = Path(sample_path)
        rel_key = str(Path(sample_path_obj.parent.name) / sample_path_obj.name)
        relpath_to_idx[rel_key] = idx
    return relpath_to_idx


def resolve_imagenet_subset_indices(
    dataset, data_dir: Union[str, Path], subset: str
) -> list[int]:
    """Return ordered dataset indices for supported ImageNet subset modes."""
    if subset == "none":
        return []
    if subset != "legacy_400_16_val":
        raise ValueError(
            f"ImageNet subset '{subset}' not supported. "
            "Supported values are 'none' and 'legacy_400_16_val'."
        )

    selected_relpaths = _build_legacy_400_16_val_relpaths(Path(data_dir))
    relpath_to_idx = _build_dataset_relpath_to_index(dataset)

    selected_indices: list[int] = []
    unmatched: list[str] = []
    for rel_path in selected_relpaths:
        if rel_path not in relpath_to_idx:
            unmatched.append(rel_path)
            continue
        selected_indices.append(relpath_to_idx[rel_path])

    if unmatched:
        preview = ", ".join(unmatched[:5])
        raise ValueError(
            f"Could not map {len(unmatched)} legacy subset images to dataset indices. "
            f"First unmatched relpaths: {preview}"
        )

    if subset == "legacy_400_16_val" and len(selected_indices) != 400:
        raise ValueError(
            "Expected exactly 400 resolved indices for legacy_400_16_val, got "
            f"{len(selected_indices)}."
        )

    return selected_indices


def get_vision_dataset(
    dataset_name: str,
    root_dir: Union[str, Path],
    image_size: int,
    stage: str,
    raw: bool = False,
    imagenet_subset: str = "none",
    return_imagenet_subset_indices: bool = False,
) -> Union[
    datasets.ImageFolder,
    tuple[datasets.ImageFolder, list[int]],
    tuple[ImageNetFolder, ImageNetFolder],
]:
    """Load a vision dataset.

    Parameters
    ----------
    raw : bool
        If ``True`` the validation transform returns pixel-space ``[0, 1]``
        tensors without ImageNet normalisation (used for metamer generation).
    """
    if dataset_name == "imagenet":
        train_transform, val_transform = get_vision_transform(
            dataset_name, image_size, raw=raw
        )
        if stage == "validate":
            val_dataset = ImageNetFolder(root_dir, "val", val_transform)
            if imagenet_subset != "none":
                selected_indices = resolve_imagenet_subset_indices(
                    val_dataset, root_dir, imagenet_subset
                )
                if return_imagenet_subset_indices:
                    return val_dataset, selected_indices
            elif return_imagenet_subset_indices:
                return val_dataset, []
            return val_dataset
        else:
            if imagenet_subset != "none":
                raise ValueError(
                    "ImageNet subsets are only supported for stage='validate'."
                )
            return ImageNetFolder(root_dir, "train", train_transform), ImageNetFolder(
                root_dir, "val", val_transform
            )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
