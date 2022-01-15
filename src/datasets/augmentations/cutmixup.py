from typing import Any, Generator, Iterable, Tuple

import numpy as np
import torch
import torch.utils.data


@torch.no_grad()
def apply_mixup(
    image1: torch.Tensor,
    target1: torch.Tensor,
    image2: torch.Tensor,
    target2: torch.Tensor,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    lam = np.random.beta(alpha, alpha)
    lam = 1 - lam if lam < 0.5 else lam

    # generate mixed image
    image = lam * image1 + (1 - lam) * image2

    # generate mixed probability
    target = lam * target1 + (1 - lam) * target2

    return image, target


@torch.no_grad()
def apply_cutmix(
    image1: torch.Tensor,
    target1: torch.Tensor,
    image2: torch.Tensor,
    target2: torch.Tensor,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    lam = np.random.beta(alpha, alpha)
    lam = 1 - lam if lam < 0.5 else lam

    # generate mixed image
    bbx1, bby1, bbx2, bby2 = _get_rand_bbox(image1.shape[-1], image1.shape[-2], lam)
    image = image1.clone()
    image[:, bby1:bby2, bbx1:bbx2] = image2[:, bby1:bby2, bbx1:bbx2]

    # generate mixed probability
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image1.shape[-1] * image1.shape[-2]))
    target = lam * target1 + (1 - lam) * target2

    return image, target


def _get_rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
    cut_rat = np.sqrt(1 - lam)
    cut_w = np.int(width * cut_rat)
    cut_h = np.int(height * cut_rat)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


@torch.no_grad()
def apply_labelsmooth(
    image: torch.Tensor,
    target: torch.Tensor,
    labelsmooth: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    target *= 1.0 - labelsmooth
    target += labelsmooth / target.shape[0]
    return image, target


def apply_cutmixup_to_dataset(
    dataset: torch.utils.data.Dataset,
    mixup_prob: float,
    mixup_alpha: float,
    cutmix_prob: float,
    cutmix_alpha: float,
    labelsmooth: float,
) -> torch.utils.data.Dataset:
    if ((mixup_prob > 0.0 and mixup_alpha > 0.0)
            or (cutmix_prob > 0.0 and cutmix_alpha > 0.0)):
        dataset = _CutMixupDataset(
            dataset=dataset,
            mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
            cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha)

    if labelsmooth > 0.0:
        dataset = _LabelSmoothDataset(dataset, labelsmooth=labelsmooth)

    return dataset


class _CutMixupDataset(torch.utils.data.Dataset):
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
            return apply_mixup(
                image1, target1, image2, target2, self.mixup_alpha)
        elif prob < mixup_prob + cutmix_prob:
            return apply_cutmix(
                image1, target1, image2, target2, self.cutmix_alpha)
        else:
            return image1, target1


class _LabelSmoothDataset(torch.utils.data.Dataset):
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
        return apply_labelsmooth(
            image, target, self.labelsmooth)


def apply_cutmixup_to_stream(
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    mixup_prob: float,
    mixup_alpha: float,
    cutmix_prob: float,
    cutmix_alpha: float,
    labelsmooth: float,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if ((mixup_prob > 0.0 and mixup_alpha > 0.0)
            or (cutmix_prob > 0.0 and cutmix_alpha > 0.0)):
        dataset = _apply_cutmixup_to_stream(
            dataset=dataset,
            mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
            cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha)

    if labelsmooth > 0.0:
        dataset = _apply_labelsmooth_to_stream(
            dataset=dataset, labelsmooth=labelsmooth)

    return dataset


@ torch.no_grad()
def _apply_cutmixup_to_stream(
    dataset: Any,
    mixup_prob: float,
    mixup_alpha: float,
    cutmix_prob: float,
    cutmix_alpha: float,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    cutmix_prob /= max(1.0 - mixup_prob, 1e-6)
    image1, target1 = None, None
    image2, target2 = None, None

    for values in dataset:
        image1, target1 = image2, target2
        image2, target2 = values

        if image1 is None:
            continue

        if mixup_prob + cutmix_prob > 1.0:
            mixup_prob /= mixup_prob + cutmix_prob
            cutmix_prob /= mixup_prob + cutmix_prob

        prob = np.random.rand()

        if prob < mixup_prob:
            yield apply_mixup(
                image1, target1, image2, target2, mixup_alpha)
        elif prob < mixup_prob + cutmix_prob:
            yield apply_cutmix(
                image1, target1, image2, target2, cutmix_alpha)
        else:
            yield image1, target1

    if image2 is not None and target2 is not None:
        yield image2, target2


@torch.no_grad()
def _apply_labelsmooth_to_stream(
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    labelsmooth: float,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    for image, target in dataset:
        yield apply_labelsmooth(image, target, labelsmooth)
