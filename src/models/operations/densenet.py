import collections
from typing import Callable

import torch.nn as nn

from ..modules import DropBlock


class DenseNetOperation(nn.Sequential):
    '''
    Operation class for DenseNets.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        growth: int,
        expansion: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        dropblock: bool,
        **kwargs,
    ) -> None:
        if stride != 1:
            super().__init__(collections.OrderedDict((n, m) for n, m in [
                ('norm1', normalization(in_channels)),
                ('act1', activation(inplace=True)),
                ('conv1', nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False)),
                ('pool1', nn.AvgPool2d(kernel_size=2, stride=stride)),
            ] if m is not None))
        else:
            channels = growth * expansion
            super().__init__(collections.OrderedDict((n, m) for n, m in [
                ('norm1', normalization(in_channels)),
                ('drop1', None if not dropblock else DropBlock()),
                ('act1', activation(inplace=True)),
                ('conv1', nn.Conv2d(
                    in_channels, channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False)),
                ('norm2', normalization(channels)),
                ('drop2', None if not dropblock else DropBlock()),
                ('act2', activation(inplace=True)),
                ('conv2', nn.Conv2d(
                    channels, growth, kernel_size=3, padding=1,
                    stride=1, bias=False)),
            ] if m is not None))
