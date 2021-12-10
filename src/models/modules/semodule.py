import collections
from typing import Callable
import math
import torch.nn as nn


class SEModule(nn.Module):
    '''
    Squeeze and excitation module.
    https://arxiv.org/abs/1709.01507
    '''

    def __init__(
        self,
        channels: int,
        reduction: float,
        divisor: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        sigmoid: Callable[..., nn.Module] = nn.Sigmoid,
        round_or_ceil: str = 'round',
    ) -> None:
        super().__init__()
        if round_or_ceil == 'ceil':
            hidden_channels = math.ceil(channels / reduction / divisor)
        else:
            hidden_channels = round(channels / reduction / divisor)

        hidden_channels = max(hidden_channels, 1) * divisor

        self.op = nn.Sequential(collections.OrderedDict([
            ('pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('conv1', nn.Conv2d(channels, hidden_channels, kernel_size=1, padding=0)),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(hidden_channels, channels, kernel_size=1, padding=0)),
            ('sigmoid', sigmoid()),
        ]))

    def forward(self, x):
        return x * self.op(x)
