from typing import Callable

import torch
import torch.nn as nn


class ViTHead(nn.Module):
    '''
    Head class for Vision Transformer.
    https://arxiv.org/abs/2010.11929
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
        x = x.squeeze(3).permute(0, 2, 1)
        x = self.norm(x)
        x = x[:, 0, :, None, None]
        return x
