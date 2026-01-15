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


def get_vision_transform(
    dataset_name: str, image_size: int = 224
) -> Tuple[transforms.Compose, transforms.Compose]:
    if dataset_name == "imagenet":
        return get_imagenet_train_transform(image_size), get_imagenet_val_transform(
            image_size
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
