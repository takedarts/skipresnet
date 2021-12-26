from typing import Tuple

import torch
import torch.utils.data


def setup_dataloader(dataset_name: str, data_path: str) -> None:
    if dataset_name == 'dummy':
        pass
    else:
        raise Exception(f'Unsuppoted dataset: {dataset_name}')


def create_train_dataloader(
    batch_size: int,
    num_workers: int,
    num_cores: int,
    pin_memory: bool,
    **kwargs,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset=DummyDataset(16),
        batch_size=batch_size // num_cores,
        shuffle=True, drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0)


def create_valid_dataloader(
    batch_size: int,
    num_workers: int,
    num_cores: int,
    pin_memory: bool,
    **kwargs,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset=DummyDataset(8),
        batch_size=batch_size // num_cores,
        shuffle=False, drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        super().__init__()
        self.data = torch.randn([length, 3, 32, 32], dtype=torch.float32)
        self.values = torch.randn([length, 10], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.values[index]
