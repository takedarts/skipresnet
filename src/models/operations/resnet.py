import collections
from typing import Callable, Union

import torch.nn as nn

from ..modules import BlurPool2d, DropBlock, SEModule


class BasicOperation(nn.Sequential):
    '''
    Operation class with 2 convolution layers.
    This module is used in ResNet family networks (ex: ResNet-18).
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
        semodule: bool,
        semodule_reduction: int,
        semodule_divisor: int,
        semodule_activation: Callable[..., nn.Module],
        semodule_sigmoid: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False)),
            ('norm2', normalization(out_channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('semodule', None if not semodule else SEModule(
                out_channels,
                reduction=semodule_reduction,
                divisor=semodule_divisor,
                activation=semodule_activation,
                sigmoid=semodule_sigmoid)),
        ] if m is not None))


class BottleneckOperation(nn.Sequential):
    '''
    Operation class with 3 convolution layers.
    This module is used in ResNet family networks (ex: ResNet-50).
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
        semodule: bool,
        semodule_reduction: int,
        semodule_divisor: int,
        semodule_activation: Callable[..., nn.Module],
        semodule_sigmoid: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        channels = round(out_channels / bottleneck * groups)

        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
            ('semodule', None if not semodule else SEModule(
                out_channels,
                reduction=semodule_reduction,
                divisor=semodule_divisor,
                activation=semodule_activation,
                sigmoid=semodule_sigmoid)),
        ] if m is not None))


class BlurPoolBottleneckOperation(nn.Sequential):
    '''
    Operation class with 3 convolution layers and blur pooling.
    This module is known as a residual operation in ResNet-D.
    https://arxiv.org/abs/1812.01187
    '''

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        groups,
        bottleneck,
        normalization,
        activation,
        dropblock,
        **kwargs,
    ) -> None:
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('pool', None if stride == 1 else BlurPool2d(channels, stride=stride)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m is not None))
