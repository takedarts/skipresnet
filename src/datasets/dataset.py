from typing import Iterable

from . import cifar, dummy, imagenet, imageneta


def prepare_dataset(dataset_name: str, data_path: str) -> None:
    if dataset_name in ('cifar10', 'cifar100'):
        cifar.prepare_dataset(dataset_name, data_path)
    elif dataset_name == 'imagenet':
        imagenet.prepare_dataset(dataset_name, data_path)
    elif dataset_name == 'imageneta':
        imageneta.prepare_dataset(dataset_name, data_path)
    elif dataset_name == 'dummy':
        dummy.prepare_dataset(dataset_name, data_path)
    else:
        raise Exception(f'Unsuppoted dataset: {dataset_name}')


def create_train_dataloader(
    dataset_name: str,
    data_path: str,
    batch_size: int,
    num_workers: int,
    num_cores: int,
    pin_memory: bool,
    **kwargs,
) -> Iterable:
    if dataset_name in ('cifar10', 'cifar100'):
        return cifar.create_train_dataloader(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_cores=num_cores,
            pin_memory=pin_memory,
            **kwargs)
    elif dataset_name == 'imagenet':
        return imagenet.create_train_dataloader(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_cores=num_cores,
            pin_memory=pin_memory,
            **kwargs)
    elif dataset_name == 'dummy':
        return dummy.create_train_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            num_cores=num_cores,
            pin_memory=pin_memory,
            **kwargs)
    else:
        raise Exception(f'Unsuppoted dataset: {dataset_name}')


def create_valid_dataloader(
    dataset_name: str,
    data_path: str,
    batch_size: int,
    num_workers: int,
    num_cores: int,
    pin_memory: bool,
    **kwargs,
) -> Iterable:
    if dataset_name in ('cifar10', 'cifar100'):
        return cifar.create_valid_dataloader(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_cores=num_cores,
            pin_memory=pin_memory,
            **kwargs)
    elif dataset_name == 'imagenet':
        return imagenet.create_valid_dataloader(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_cores=num_cores,
            pin_memory=pin_memory,
            **kwargs)
    elif dataset_name == 'imageneta':
        return imageneta.create_valid_dataloader(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_cores=num_cores,
            pin_memory=pin_memory,
            **kwargs)
    elif dataset_name == 'dummy':
        return dummy.create_valid_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            num_cores=num_cores,
            pin_memory=pin_memory,
            **kwargs)
    else:
        raise Exception(f'Unsuppoted dataset: {dataset_name}')
