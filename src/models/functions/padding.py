from typing import Tuple

import torch
import torch.nn.functional as F


def get_same_padding(
    kernel: int,
    stride: int,
    dilation: int,
    insize: int,
) -> Tuple[int, int]:
    outsize = (insize - 1) // stride * stride + 1 + (kernel - 1) * dilation
    padding = max(outsize - insize, 0)

    prev_padding = padding // 2
    post_padding = padding - prev_padding

    return prev_padding, post_padding


def pad_same(
    x: torch.Tensor,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int] = (1, 1),
    value: float = 0
) -> torch.Tensor:
    in_height, in_width = x.size()[-2:]
    padding_height = get_same_padding(kernel[0], stride[0], dilation[0], in_height)
    padding_width = get_same_padding(kernel[1], stride[1], dilation[1], in_width)
    padding = padding_width + padding_height

    if padding != (0, 0, 0, 0):
        x = F.pad(x, padding, value=value)

    return x
