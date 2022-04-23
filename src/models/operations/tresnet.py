import collections
from typing import Callable, List, Tuple, Union

import torch.nn as nn

from ..modules import BlurPool2d, DropBlock, SEModule


def _create_basic_operation(
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
) -> List[Tuple[str, nn.Module]]:
    channels = round(out_channels / bottleneck)

    if out_channels / semodule_reduction < 64:
        semodule_divisor = 64

    return [
        (n, m) for n, m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=1, groups=groups, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('blur', None if stride == 1 else BlurPool2d(
                channels, filter_size=3, stride=stride, padding=1)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False)),
            ('norm2', normalization(out_channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('semodule', None if not semodule else SEModule(
                channels=out_channels,
                reduction=semodule_reduction,
                divisor=semodule_divisor,
                activation=semodule_activation,
                sigmoid=semodule_sigmoid)),
        ] if m is not None
    ]


def _create_bottleneck_operation(
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
) -> List[Tuple[str, nn.Module]]:
    channels = round(out_channels / bottleneck)

    if out_channels / semodule_reduction < 64:
        semodule_divisor = 64

    return [
        (n, m) for n, m in [
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
            ('blur', None if stride == 1 else BlurPool2d(
                channels, filter_size=3, stride=stride, padding=1)),
            ('semodule', None if not semodule else SEModule(
                channels=channels,
                reduction=semodule_reduction,
                divisor=semodule_divisor,
                activation=semodule_activation,
                sigmoid=semodule_sigmoid)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m is not None
    ]


class TResNetOperation(nn.Sequential):
    '''Operation class for TResNets.
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
        style: str,
        **kwargs
    ) -> None:
        if style == 'basic':
            create_operation = _create_basic_operation
        elif style == 'bottleneck':
            create_operation = _create_bottleneck_operation

        super().__init__(collections.OrderedDict(create_operation(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            groups=groups,
            bottleneck=bottleneck,
            normalization=normalization,
            activation=activation,
            dropblock=dropblock,
            semodule=semodule,
            semodule_reduction=semodule_reduction,
            semodule_divisor=semodule_divisor,
            semodule_activation=semodule_activation,
            semodule_sigmoid=semodule_sigmoid,
        )))
