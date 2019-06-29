import torchvision.transforms as transforms
import torch

from PIL import Image


def _distillation_transform(
    input_shape: tuple,
    crop_size: tuple,
    mean,
    std,
    flip_prob=0.5,
    low_ratio=None,
    is_test=False,
):
    transform_list = []
    transform_list.append(transforms.Resize(input_shape, interpolation=Image.BICUBIC))

    if low_ratio is not None:
        transform_list.append(transforms.Resize(low_ratio, interpolation=Image.BICUBIC))
        transform_list.append(transforms.Resize(input_shape, interpolation=Image.BICUBIC))

    if is_test is True:
        transform_list.append(transforms.TenCrop(crop_size))
        transform_list.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transform_list.append(transforms.Lambda(lambda images: torch.stack([transforms.Normalize(mean=mean, std=std)(image) for image in images])))
        for t in transform_list:
            print("Transform : {}".format(t))
        print("\n")
        return transforms.Compose(transform_list)

    else:
        transform_list.append(transforms.RandomHorizontalFlip(flip_prob))
        transform_list.append(transforms.RandomCrop(crop_size))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    for t in transform_list:
        print("Transform : {}".format(t))
    print("\n")

    transform = transforms.Compose(transform_list)

    return transform


def build_transforms(**kwargs):
    return _distillation_transform(**kwargs)
