import io
import json
import logging
import os
import pathlib
import shutil
import sys
import tarfile
import time
import urllib.parse
from typing import List, Tuple

import numpy as np
import torchvision
import torchvision.datasets

import webdataset as wds

from .augmentations import AutoAugmentImageNet, Lighting, RandAugment
from .webdataset import (Processor, PytorchShardList, apply_augmentations,
                         apply_class_to_tensor, gopen, url_opener)

LOGGER = logging.getLogger(__name__)

TRAIN_SHARDS = 256
VALID_SHARDS = 16


def setup_dataloader(dataset_name: str, data_path: str) -> None:
    if dataset_name == 'imagenet':
        url = urllib.parse.urlparse(data_path)
        if url.scheme == '' or url.scheme == 'file':
            path = pathlib.Path(url.path)
            _setup_imagenet_dataset(
                data_path=path, split_name='train', num_shards=TRAIN_SHARDS)
            _setup_imagenet_dataset(
                data_path=path, split_name='val', num_shards=VALID_SHARDS)
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
) -> wds.Processor:
    return _create_dataloader(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        num_cores=num_cores,
        pin_memory=pin_memory,
        train=True,
        crop_size=train_crop,
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
) -> wds.Processor:
    return _create_dataloader(
        dataset_name=dataset_name,
        data_path=data_path,
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
        randomerasing_prob=0,
        randomerasing_type='',
        cutmix_prob=0.0,
        cutmix_alpha=0.0,
        mixup_prob=0.0,
        mixup_alpha=0.0,
        labelsmooth=0.0)


def _create_dataloader(
    dataset_name: str,
    data_path: str,
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
    mixup_prob: float,
    mixup_alpha: float,
    cutmix_prob: float,
    cutmix_alpha: float,
    labelsmooth: float,
) -> Processor:
    dataset = _create_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        crop_size=crop_size,
        train=train,
        stdaugment=stdaugment,
        autoaugment=autoaugment,
        randaugment_prob=randaugment_prob,
        randaugment_num=randaugment_num,
        randaugment_mag=randaugment_mag,
        randaugment_std=randaugment_std,
        randomerasing_prob=randomerasing_prob,
        randomerasing_type=randomerasing_type)
    length = dataset.length
    sampler = dataset.sampler

    dataset = dataset.then(
        apply_augmentations,
        mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
        cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha,
        labelsmooth=labelsmooth)
    dataset = dataset.batched(batch_size // num_cores, partial=not train)

    if train:
        num_workers = min(max(TRAIN_SHARDS // num_cores, 1), num_workers)
    else:
        num_workers = min(max(VALID_SHARDS // num_cores, 1), num_workers)

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0)

    num_batches = length // batch_size
    loader = loader.repeat(sys.maxsize).with_epoch(num_batches)

    return Processor(loader, num_batches, sampler)


def _create_dataset(
    dataset_name: str,
    data_path: str,
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
) -> Processor:
    if dataset_name == 'imagenet':
        return ImagenetDataset(
            data_path=data_path,
            num_classes=1000,
            crop_size=crop_size,
            train=train,
            stdaugment=stdaugment,
            autoaugment=autoaugment,
            randaugment_prob=randaugment_prob,
            randaugment_num=randaugment_num,
            randaugment_mag=randaugment_mag,
            randaugment_std=randaugment_std,
            randomerasing_prob=randomerasing_prob,
            randomerasing_type=randomerasing_type)
    else:
        raise Exception(f'Unsuppoted dataset: {dataset_name}')


def ImagenetDataset(
    data_path: str,
    num_classes: int,
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
) -> Processor:
    # image transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    lightning = {
        'alphastd': 0.1,
        'eigval': [0.2175, 0.0188, 0.0045],
        'eigvec': [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]]
    }

    if stdaugment:
        transforms = [
            torchvision.transforms.RandomResizedCrop(crop_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4),
            torchvision.transforms.ToTensor(),
            Lighting(**lightning),
            torchvision.transforms.Normalize(mean=mean, std=std)]
    else:
        transforms = [
            torchvision.transforms.Resize(round(crop_size / 224 * 256)),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)]

    if autoaugment:
        transforms.insert(0, AutoAugmentImageNet())

    if randaugment_prob != 0.0:
        transforms.insert(0, RandAugment(
            randaugment_num, randaugment_mag,
            randaugment_prob, randaugment_std))

    if randomerasing_prob != 0.0:
        value = 0 if randomerasing_type == 'zero' else 'random'
        transforms.append(torchvision.transforms.RandomErasing(
            p=randomerasing_prob, value=value))

    transform = torchvision.transforms.Compose(transforms)

    # load meta data
    files, length = _load_imagenet_files(data_path=data_path, train=train)

    # make dataset
    shardlist = PytorchShardList(files, shuffle=train)
    dataset = (
        shardlist
        .then(url_opener)
        .then(wds.tariterators.tar_file_expander)
        .then(wds.tariterators.group_by_keys))

    if train:
        dataset = dataset.shuffle(5000)

    dataset = (
        dataset
        .decode('pil')
        .to_tuple('ppm;jpg;jpeg;png', 'cls')
        .map_tuple(transform, wds.iterators.identity)
        .then(apply_class_to_tensor, num_classes=num_classes))

    return Processor(dataset, length, shardlist)


def _load_imagenet_files(
    data_path: str,
    train: bool,
) -> Tuple[List[str], int]:
    url = urllib.parse.urlparse(data_path)
    meta_file = 'train_shards.json' if train else 'val_shards.json'
    shard_dir = 'train_shards' if train else 'val_shards'

    if url.scheme == '' or url.scheme == 'file':
        path = pathlib.Path(url.path)
        with open(path / meta_file, 'r') as reader:
            meta = json.load(reader)
        files = [str(path / shard_dir / n) for n in meta['files']]
    else:
        data_path = data_path[:-1] if data_path[-1] == '/' else data_path
        with gopen(f'{data_path}/{meta_file}', 'rb') as reader:
            meta = json.load(reader)
        files = [f'{data_path}/{shard_dir}/{n}' for n in meta['files']]

    return files, meta['length']


def _setup_imagenet_dataset(
    data_path: pathlib.Path,
    split_name: str,
    num_shards: int,
) -> None:
    meta_file = data_path / f'{split_name}_shards.json'
    shard_path = data_path / f'{split_name}_shards'

    if meta_file.is_file():
        return

    LOGGER.debug('Expand ImageNet dataset: %s/%s', data_path, split_name)
    dataset = torchvision.datasets.ImageNet(str(data_path), split_name)
    indexes = np.random.permutation(len(dataset))

    os.makedirs(shard_path, exist_ok=True)
    shard_files = []

    for num in range(num_shards):
        shard_file = shard_path / f'{num:05d}.tar'
        shard_files.append(str(shard_file.name))
        LOGGER.debug('Create shard file: %s', shard_file)

        with tarfile.open(shard_file, 'w') as tar_writer:
            for idx in indexes[num::num_shards]:
                key = f'{idx:07d}'
                path, label = dataset.imgs[idx]
                image = pathlib.Path(path).read_bytes()
                value = f'{label}'.encode('ascii')
                mtime = int(time.time())

                for ext, bin in (('jpg', image), ('cls', value)):
                    info = tarfile.TarInfo(f'{key}.{ext}')
                    info.size = len(bin)
                    info.mtime = mtime
                    info.mode = 0o0644
                    tar_writer.addfile(info, io.BytesIO(bin))

    meta = {
        'length': len(dataset),
        'files': shard_files,
    }

    with open(meta_file, 'w') as writer:
        json.dump(meta, writer)

    os.remove(data_path / 'meta.bin')
    shutil.rmtree(data_path / split_name, ignore_errors=True)
