from typing import Any

import numpy as np
import torch
import torch.utils.data

from . import augmentations


def apply_augmentations(
    dataset: torch.utils.data.Dataset,
    mixup_prob: float,
    mixup_alpha: float,
    cutmix_prob: float,
    cutmix_alpha: float,
    labelsmooth: float,
) -> torch.utils.data.Dataset:
    if ((mixup_prob > 0.0 and mixup_alpha > 0.0)
            or (cutmix_prob > 0.0 and cutmix_alpha > 0.0)):
        dataset = MixupCutmix(
            dataset=dataset,
            mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
            cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha)

    if labelsmooth > 0.0:
        dataset = LabelSmooth(dataset, labelsmooth=labelsmooth)

    return dataset


class MixupCutmix(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        mixup_prob: float,
        mixup_alpha: float,
        cutmix_prob: float,
        cutmix_alpha: float,
    ) -> None:
        self.dataset = dataset
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob / max(1.0 - mixup_prob, 1e-6)
        self.cutmix_alpha = cutmix_alpha

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx):
        image1, target1 = self.dataset[idx]
        image2, target2 = self.dataset[np.random.randint(len(self.dataset))]
        mixup_prob = self.mixup_prob
        cutmix_prob = self.cutmix_prob

        if mixup_prob + cutmix_prob > 1.0:
            mixup_prob /= mixup_prob + cutmix_prob
            cutmix_prob /= mixup_prob + cutmix_prob

        prob = np.random.rand()

        if prob < mixup_prob:
            return augmentations.apply_mixup(
                image1, target1, image2, target2, self.mixup_alpha)
        elif prob < mixup_prob + cutmix_prob:
            return augmentations.apply_cutmix(
                image1, target1, image2, target2, self.cutmix_alpha)
        else:
            return image1, target1


class LabelSmooth(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        labelsmooth: float,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.labelsmooth = labelsmooth

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int) -> Any:
        image, target = self.dataset[idx]
        return augmentations.apply_labelsmooth(
            image, target, self.labelsmooth)
