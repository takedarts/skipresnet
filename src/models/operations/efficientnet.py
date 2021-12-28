import collections
from typing import Callable, Union

import torch.nn as nn

from ..modules import Conv2dSame, DropBlock, SEModule


def _create_conv(
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int,
    normalization: Callable[..., nn.Module],
    activation: Callable[..., nn.Module],
    dropblock: bool,
    **kwargs,
) -> collections.OrderedDict:
    return collections.OrderedDict((n, m) for n, m in [
        ('conv', Conv2dSame(
            in_channels, out_channels, kernel_size=kernel,
            stride=stride, bias=False, image_size=256)),
        ('norm', normalization(out_channels)),
        ('drop', None if not dropblock else DropBlock()),
        ('act', activation(inplace=True)),
    ] if m is not None)


def _create_edge(
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int,
    expansion: Union[int, float],
    normalization: Callable[..., nn.Module],
    activation: Callable[..., nn.Module],
    dropblock: bool,
    semodule: bool,
    semodule_reduction: int,
    semodule_divisor: int,
    semodule_activation: Callable[..., nn.Module],
    semodule_sigmoid: Callable[..., nn.Module],
    **kwargs,
) -> collections.OrderedDict:
    channels = round(in_channels * expansion)

    return collections.OrderedDict((n, m) for n, m in [
        ('conv1', Conv2dSame(
            in_channels, channels, kernel_size=kernel,
            stride=stride, bias=False, image_size=256)),
        ('norm1', normalization(channels)),
        ('drop1', None if not dropblock else DropBlock()),
        ('act1', activation(inplace=True)),
        ('semodule', None if not semodule else SEModule(
            channels,
            reduction=semodule_reduction,
            divisor=semodule_divisor,
            activation=semodule_activation,
            sigmoid=semodule_sigmoid,
            round_or_ceil='ceil')),
        ('conv2', nn.Conv2d(
            channels, out_channels, kernel_size=1,
            padding=0, stride=1, bias=False)),
        ('norm2', normalization(out_channels)),
        ('drop2', None if not dropblock else DropBlock()),
    ] if m is not None)


def _create_inverted(
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int,
    expansion: Union[int, float],
    normalization: Callable[..., nn.Module],
    activation: Callable[..., nn.Module],
    dropblock: bool,
    semodule: bool,
    semodule_reduction: int,
    semodule_divisor: int,
    semodule_activation: Callable[..., nn.Module],
    semodule_sigmoid: Callable[..., nn.Module],
    **kwargs,
) -> collections.OrderedDict:
    channels = round(in_channels * expansion)

    return collections.OrderedDict((n, m) for n, m in [
        ('conv1', nn.Conv2d(
            in_channels, channels, kernel_size=1,
            padding=0, stride=1, bias=False)),
        ('norm1', normalization(channels)),
        ('drop1', None if not dropblock else DropBlock()),
        ('act1', activation(inplace=True)),
        ('conv2', Conv2dSame(
            channels, channels, kernel_size=kernel, groups=channels,
            stride=stride, bias=False, image_size=256)),
        ('norm2', normalization(channels)),
        ('drop2', None if not dropblock else DropBlock()),
        ('act2', activation(inplace=True)),
        ('semodule', None if not semodule else SEModule(
            channels,
            reduction=semodule_reduction,
            divisor=semodule_divisor,
            activation=semodule_activation,
            sigmoid=semodule_sigmoid,
            round_or_ceil='ceil')),
        ('conv3', nn.Conv2d(
            channels, out_channels, kernel_size=1,
            padding=0, stride=1, bias=False)),
        ('norm3', normalization(out_channels)),
        ('drop3', None if not dropblock else DropBlock()),
    ] if m is not None)


def _create_depthwise(
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int,
    expansion: Union[int, float],
    normalization: Callable[..., nn.Module],
    activation: Callable[..., nn.Module],
    dropblock: bool,
    semodule: bool,
    semodule_reduction: int,
    semodule_divisor: int,
    semodule_activation: Callable[..., nn.Module],
    semodule_sigmoid: Callable[..., nn.Module],
    **kwargs,
) -> collections.OrderedDict:
    channels = round(in_channels * expansion)

    return collections.OrderedDict((n, m) for n, m in [
        ('conv1', Conv2dSame(
            in_channels, channels, kernel_size=kernel, groups=in_channels,
            stride=stride, bias=False, image_size=256)),
        ('norm1', normalization(channels)),
        ('drop1', None if not dropblock else DropBlock()),
        ('act1', activation(inplace=True)),
        ('semodule', None if not semodule else SEModule(
            channels,
            reduction=semodule_reduction,
            divisor=semodule_divisor,
            activation=semodule_activation,
            sigmoid=semodule_sigmoid,
            round_or_ceil='ceil')),
        ('conv2', nn.Conv2d(
            channels, out_channels, kernel_size=1,
            padding=0, stride=1, bias=False)),
        ('norm2', normalization(out_channels)),
        ('drop2', None if not dropblock else DropBlock()),
    ] if m is not None)


class EfficientNetOperation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style: str,
        **kwargs,
    ) -> None:
        super().__init__()

        create_fn: Callable[..., collections.OrderedDict]
        if style == 'conv':
            create_fn = _create_conv
        elif style == 'edge':
            create_fn = _create_edge
        elif style == 'inverted':
            create_fn = _create_inverted
        elif style == 'depthwise':
            create_fn = _create_depthwise
        else:
            raise Exception(f'Unknown operation: {style}')

        super().__init__(create_fn(in_channels, out_channels, **kwargs))
