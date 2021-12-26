import torch
import numpy as np
from typing import Tuple


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
