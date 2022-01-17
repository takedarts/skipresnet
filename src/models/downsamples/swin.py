from typing import Callable

import torch
import torch.nn as nn


class SwinDownsample(nn.Module):
    '''
    Downsample class for Swin Transformers.
    https://arxiv.org/abs/2103.14030

    This class does Patch Margning.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        normalization: Callable[..., nn.Module],
        **kwargs,
    ) -> None:
        super().__init__()
        self.enabled = in_channels != out_channels or stride != 1

        if self.enabled:
            channels = in_channels * stride * stride

            self.norm = normalization(channels)
            self.reduction = nn.Linear(channels, out_channels, bias=False)
            self.channels = channels
            self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        x = x.reshape(*x.shape[:2], -1, self.stride, x.shape[3])
        x = x.reshape(*x.shape[:4], -1, self.stride)
        x = x.permute(0, 2, 4, 5, 3, 1)
        height, width = x.shape[1:3]

        x = x.reshape(x.shape[0], -1, self.channels)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(x.shape[0], height, width, -1)
        x = x.permute(0, 3, 1, 2)

        return x
