from typing import Callable

import torch
import torch.nn as nn


class SwinPatchStem(nn.Module):
    '''
    Stem class for Swin Transformers.
    https://arxiv.org/abs/2103.14030
    '''

    def __init__(
        self,
        out_channels: int,
        patch_size: int,
        image_size: int,
        normalization: Callable[..., nn.Module],
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            3, out_channels, kernel_size=patch_size, stride=patch_size)
        self.norm = normalization(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        B, C, H, W = x.shape[:]
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x
