from typing import Callable

import torch
import torch.nn as nn


class SwinHead(nn.Module):
    '''
    Head class for Swin Transformer.
    https://arxiv.org/abs/2103.14030
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__()
        self.norm = normalization(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x
