from typing import Tuple

import torch


def apply_labelsmooth(
    image: torch.Tensor,
    target: torch.Tensor,
    labelsmooth: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    target *= 1.0 - labelsmooth
    target += labelsmooth / target.shape[0]
    return image, target
