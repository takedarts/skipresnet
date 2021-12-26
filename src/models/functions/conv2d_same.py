from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .padding import pad_same


def conv2d_same(
    x: torch.Tensor,
    kernel_size: Tuple[int, int],
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    return F.conv2d(
        pad_same(x, kernel_size, stride, dilation),
        weight, bias, stride, (0, 0), dilation, groups)
