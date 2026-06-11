from typing import Tuple

from torchvision import transforms

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_imagenet_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_imagenet_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_imagenet_raw_transform(image_size: int = 224) -> transforms.Compose:
    """Resize/crop to ``image_size`` and convert to ``[0, 1]`` tensor.

    No ImageNet normalisation — suitable for metamer generation where the
    model normalises internally.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def get_vision_transform(
    dataset_name: str,
    image_size: int = 224,
    raw: bool = False,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_transform, val_transform)`` for *dataset_name*.

    If *raw* is ``True`` the val transform omits normalisation (pixel space
    ``[0, 1]``), useful for metamer generation.
    """
    if dataset_name == "imagenet":
        val_tf = (
            get_imagenet_raw_transform(image_size)
            if raw
            else get_imagenet_val_transform(image_size)
        )
        return get_imagenet_train_transform(image_size), val_tf
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
