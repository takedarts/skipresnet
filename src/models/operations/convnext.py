from typing import Callable

import torch
import torch.nn as nn


class ConvNeXtOperation(nn.Module):
    '''
    Operation class for ConvNeXt.
    hhttps://arxiv.org/abs/2201.03545
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        layer_scale_init_value: float,
        dropout_prob: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(  # depthwise conv
            out_channels, out_channels, kernel_size=7,
            padding=3, groups=out_channels)
        self.norm = normalization(out_channels)
        self.pwconv1 = nn.Linear(out_channels, 4 * out_channels)  # pointwise conv
        self.act = activation(inplace=True)
        self.pwconv2 = nn.Linear(4 * out_channels, out_channels)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        return x
