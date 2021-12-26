import torch
import numpy as np
from typing import Tuple


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
