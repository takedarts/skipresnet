import collections
from typing import Callable, Union

import torch.nn as nn

from ..modules import DropBlock


class SingleActivationBasicOperation(nn.Sequential):
    '''
    Operation class with 2 convolution layers for single-activation ResNets.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        groups: int,
        bottleneck: Union[int, float],
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        dropblock: bool,
        **kwargs,
    ) -> None:
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('norm1', normalization(in_channels)),
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m is not None))


class SingleActivationBottleneckOperation(nn.Sequential):
    '''
    Operation class with 3 convolution layers for single-activation ResNets.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        groups: int,
        bottleneck: Union[int, float],
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        dropblock: bool,
        **kwargs,
    ) -> None:
        channels = round(out_channels / bottleneck * groups)

        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('norm1', normalization(in_channels)),
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm3', normalization(channels)),
            ('drop3', None if not dropblock else DropBlock()),
            ('act3', activation(inplace=True)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm4', normalization(out_channels)),
            ('drop4', None if not dropblock else DropBlock()),
        ] if m is not None))
