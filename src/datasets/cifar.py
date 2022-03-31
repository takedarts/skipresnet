import pathlib
from typing import Any, List

import torch
import torch.utils.data
import torchvision

from .augmentations import (AutoAugmentCIFAR10, RandAugment,
                            apply_cutmixup_to_dataset)


def prepare_dataset(dataset_name: str, data_path: str) -> None:
    if dataset_name == 'cifar10':
        if not (pathlib.Path(data_path) / 'cifar-10-batches-py').is_dir():
            torchvision.datasets.CIFAR10(data_path, download=True, train=True)
            torchvision.datasets.CIFAR10(data_path, download=True, train=False)
    elif dataset_name == 'cifar100':
        if not (pathlib.Path(data_path) / 'cifar-100-python').is_dir():
            torchvision.datasets.CIFAR100(data_path, download=True, train=True)
            torchvision.datasets.CIFAR100(data_path, download=True, train=False)
    else:
        raise Exception(f'Unsuppoted dataset: {dataset_name}')


def create_train_dataloader(
    dataset_name: str,
    data_path: str,
    batch_size: int,
    num_workers: int,
    num_cores: int,
    pin_memory: bool,
    train_crop: int,
    autoaugment: bool,
    randaugment_prob: float,
    randaugment_num: int,
    randaugment_mag: int,
    randaugment_std: float,
    randomerasing_prob: float,
    randomerasing_type: str,
    mixup_prob: float,
    mixup_alpha: float,
    cutmix_prob: float,
    cutmix_alpha: float,
    labelsmooth: float,
    **kwargs,
) -> torch.utils.data.DataLoader:
    return _create_dataloader(
        dataset_name=dataset_name,
        data_path=pathlib.Path(data_path),
        train=True,
        crop_size=train_crop,
        batch_size=batch_size,
        num_workers=num_workers,
        num_cores=num_cores,
        pin_memory=pin_memory,
        stdaugment=True,
        autoaugment=autoaugment,
        randaugment_prob=randaugment_prob,
        randaugment_num=randaugment_num,
        randaugment_mag=randaugment_mag,
        randaugment_std=randaugment_std,
        randomerasing_prob=randomerasing_prob,
        randomerasing_type=randomerasing_type,
        cutmix_prob=cutmix_prob,
        cutmix_alpha=cutmix_alpha,
        mixup_prob=mixup_prob,
        mixup_alpha=mixup_alpha,
        labelsmooth=labelsmooth)


def create_valid_dataloader(
    dataset_name: str,
    data_path: str,
    batch_size: int,
    num_workers: int,
    num_cores: int,
    pin_memory: bool,
    valid_crop: int,
    **kwargs,
) -> torch.utils.data.DataLoader:
    return _create_dataloader(
        dataset_name=dataset_name,
        data_path=pathlib.Path(data_path),
        batch_size=batch_size,
        num_workers=num_workers,
        num_cores=num_cores,
        pin_memory=pin_memory,
        train=False,
        crop_size=valid_crop,
        stdaugment=False,
        autoaugment=False,
        randaugment_prob=0.0,
        randaugment_num=0,
        randaugment_mag=0,
        randaugment_std=0.0,
        randomerasing_prob=0, randomerasing_type='',
        cutmix_prob=0.0, cutmix_alpha=0.0,
        mixup_prob=0.0, mixup_alpha=0.0,
        labelsmooth=0.0)


def _create_dataloader(
    dataset_name: str,
    data_path: pathlib.Path,
    batch_size: int,
    num_workers: int,
    num_cores: int,
    pin_memory: bool,
    train: bool,
    crop_size: int,
    stdaugment: bool,
    autoaugment: bool,
    randaugment_prob: float,
    randaugment_num: int,
    randaugment_mag: int,
    randaugment_std: float,
    randomerasing_prob: float,
    randomerasing_type: str,
    cutmix_prob: float,
    cutmix_alpha: float,
    mixup_prob: float,
    mixup_alpha: float,
    labelsmooth: float,
) -> torch.utils.data.DataLoader:
    dataset = _create_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        train=train,
        crop_size=crop_size,
        stdaugment=stdaugment,
        autoaugment=autoaugment,
        randaugment_prob=randaugment_prob,
        randaugment_num=randaugment_num,
        randaugment_mag=randaugment_mag,
        randaugment_std=randaugment_std,
        randomerasing_prob=randomerasing_prob,
        randomerasing_type=randomerasing_type)

    dataset = apply_cutmixup_to_dataset(
        dataset=dataset,
        mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
        cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha,
        labelsmooth=labelsmooth)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // num_cores,
        shuffle=train, drop_last=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0)


def _create_dataset(
    dataset_name: str,
    data_path: pathlib.Path,
    train: bool,
    crop_size: int,
    stdaugment: bool,
    autoaugment: bool,
    randaugment_prob: float,
    randaugment_num: int,
    randaugment_mag: int,
    randaugment_std: float,
    randomerasing_prob: float,
    randomerasing_type: str,
) -> torch.utils.data.Dataset:
    if dataset_name == 'cifar10':
        return Cifar10Dataset(
            data_path, train, crop_size, stdaugment, autoaugment,
            randaugment_prob, randaugment_num,
            randaugment_mag, randaugment_std,
            randomerasing_prob, randomerasing_type)
    elif dataset_name == 'cifar100':
        return Cifar100Dataset(
            data_path, train, crop_size, stdaugment, autoaugment,
            randaugment_prob, randaugment_num,
            randaugment_mag, randaugment_std,
            randomerasing_prob, randomerasing_type)
    else:
        raise Exception(f'Unsuppoted dataset: {dataset_name}')


class CifarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Any,
        num_classes: int,
        crop_size: int,
        mean: List[float],
        std: List[float],
        stdaugment: bool,
        autoaugment: bool,
        randaugment_prob: float,
        randaugment_num: int,
        randaugment_mag: int,
        randaugment_std: float,
        random_erasing_prob: float,
        random_erasing_type: str,
    ) -> None:
        super().__init__()
        if stdaugment:
            transforms = [
                torchvision.transforms.RandomCrop(
                    crop_size, padding=4, padding_mode='reflect'),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)]
        else:
            transforms = [
                torchvision.transforms.Resize(crop_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)]

        if autoaugment:
            transforms.insert(0, AutoAugmentCIFAR10())

        if randaugment_prob != 0.0:
            transforms.insert(0, RandAugment(
                randaugment_num, randaugment_mag,
                randaugment_prob, randaugment_std))

        if random_erasing_prob != 0:
            value = 0 if random_erasing_type == 'zero' else 'random'
            transforms.append(torchvision.transforms.RandomErasing(
                p=random_erasing_prob, value=value))

        self.transforms = torchvision.transforms.Compose(transforms)
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        image, label = self.dataset[idx]
        image = self.transforms(image)
        prob = torch.zeros(self.num_classes, dtype=torch.float32)
        prob[label] = 1

        return image, prob


class Cifar10Dataset(CifarDataset):
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616]

    def __init__(
        self,
        path: pathlib.Path,
        train: bool,
        crop_size: int,
        stdaugment: bool,
        autoaugment: bool,
        randaugment_prob: float,
        randaugment_num: int,
        randaugment_mag: int,
        randaugment_std: float,
        random_erasing_prob: float,
        random_erasing_type: str,
    ) -> None:
        super().__init__(
            torchvision.datasets.CIFAR10(str(path), download=False, train=train),
            crop_size=crop_size, num_classes=10, mean=self.MEAN, std=self.STD,
            stdaugment=stdaugment, autoaugment=autoaugment,
            randaugment_prob=randaugment_prob, randaugment_num=randaugment_num,
            randaugment_mag=randaugment_mag, randaugment_std=randaugment_std,
            random_erasing_prob=random_erasing_prob,
            random_erasing_type=random_erasing_type)


class Cifar100Dataset(CifarDataset):
    MEAN = [0.5071, 0.4865, 0.4409]
    STD = [0.2673, 0.2564, 0.2762]

    def __init__(
        self,
        path: pathlib.Path,
        train: bool,
        crop_size: int,
        stdaugment: bool,
        autoaugment: bool,
        randaugment_prob: float,
        randaugment_num: int,
        randaugment_mag: int,
        randaugment_std: float,
        random_erasing_prob: float,
        random_erasing_type: str,
    ) -> None:
        super().__init__(
            torchvision.datasets.CIFAR100(path, download=False, train=train),
            crop_size=crop_size, num_classes=100, mean=self.MEAN, std=self.STD,
            stdaugment=stdaugment, autoaugment=autoaugment,
            randaugment_prob=randaugment_prob, randaugment_num=randaugment_num,
            randaugment_mag=randaugment_mag, randaugment_std=randaugment_std,
            random_erasing_prob=random_erasing_prob,
            random_erasing_type=random_erasing_type)
