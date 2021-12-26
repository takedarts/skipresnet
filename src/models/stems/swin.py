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
        normalization: Callable[..., nn.Module],
        dropout_prob: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            3, out_channels, kernel_size=patch_size, stride=patch_size)
        self.norm = normalization(out_channels)
        self.drop = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        size = x.shape[:]
        x = x.reshape(size[0], size[1], -1).permute(0, 2, 1)
        x = self.norm(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).reshape(size)
        return x
