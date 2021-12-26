import os
import pathlib
import tarfile
from typing import List, Tuple

import PIL.Image
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms
from utils import download_and_verify

ARCHIVE_URL = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar'
ARCHIVE_DIGEST = 'md5:c3e55429088dc681f30d81f4726b6595'
ARCHIVE_SIZE = 687_552_512


def setup_dataloader(dataset_name: str, data_path: str) -> None:
    if dataset_name == 'imageneta':
        ImageNetADataset.prepare(pathlib.Path(data_path))
    else:
        raise Exception(f'Unsuppoted dataset: {dataset_name}')


def create_train_dataloader(*args, **kwargs):
    raise Exception('ImageNetA does not provide training images.')


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
    if dataset_name == 'imageneta':
        dataset = ImageNetADataset(
            path=pathlib.Path(data_path), crop_size=valid_crop)
    else:
        raise Exception(f'Unsuppoted dataset: {dataset_name}')

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // num_cores,
        shuffle=False, drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0)


class ImageNetADataset(torch.utils.data.Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NUM_CLASSES = 1000

    @staticmethod
    def prepare(path: pathlib.Path) -> None:
        archive_path = path / 'imagenet-a.tar'

        os.makedirs(path, exist_ok=True)

        if not archive_path.is_file():
            download_and_verify(
                path=archive_path,
                url=ARCHIVE_URL,
                digest=ARCHIVE_DIGEST,
                size=ARCHIVE_SIZE,
            )

        with tarfile.open(archive_path, 'r') as reader:
            reader.extractall(path)

    def __init__(
        self,
        path: pathlib.Path,
        crop_size: int,
        download: bool = False
    ) -> None:
        super().__init__()

        if download:
            self.prepare(path)

        self.classes = (path / 'wnids.txt').read_text().split('\n')
        self.classes = sorted(c.strip() for c in self.classes if len(c) != 0)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(round(crop_size / 224 * 256)),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.MEAN, std=self.STD)])

        self.imgs: List[Tuple[pathlib.Path, int]] = []

        for wnid_path in (path / 'imagenet-a').iterdir():
            if not wnid_path.is_dir():
                continue
            wnid = wnid_path.name
            for image_path in wnid_path.iterdir():
                self.imgs.append((image_path, self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, label = self.imgs[index]
        image = self.transform(PIL.Image.open(path).convert('RGB'))
        prob = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
        prob[label] = 1

        return image, prob
