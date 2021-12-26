import collections
from typing import Callable, Union

import torch
import torch.nn as nn


from ..modules import DropBlock, Multiply, SEModule, ScaledStdConv2dSame


class NFOperation(nn.Module):
    '''
    Operation class for Normalizer-Free ResNets (NFNets).
    https://arxiv.org/abs/2102.06171
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        groups: int,
        bottleneck: Union[int, float],
        activation: Callable[..., nn.Module],
        dropblock: bool,
        semodule: bool,
        semodule_reduction: int,
        semodule_divisor: int,
        semodule_activation: Callable[..., nn.Module],
        semodule_sigmoid: Callable[..., nn.Module],
        alpha: float,
        beta: float,
        gamma: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gain = nn.Parameter(torch.tensor(0.0))

        channels = max(round(out_channels // bottleneck), 1)

        self.op = nn.Sequential(collections.OrderedDict((n, m) for n, m in [
            ('act1', activation(inplace=True)),
            ('gamma1', Multiply(gamma, inplace=True)),
            ('beta', Multiply(beta, inplace=True)),
            ('conv1', ScaledStdConv2dSame(
                in_channels, channels, kernel_size=1,
                stride=1, groups=1, eps=1e-5, image_size=256)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('gamma2', Multiply(gamma, inplace=True)),
            ('conv2', ScaledStdConv2dSame(
                channels, channels, kernel_size=3,
                stride=stride, groups=groups, eps=1e-5, image_size=256)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act3', activation(inplace=True)),
            ('gamma3', Multiply(gamma, inplace=True)),
            ('conv3', ScaledStdConv2dSame(
                channels, channels, kernel_size=3,
                stride=1, groups=groups, eps=1e-5, image_size=256)),
            ('drop3', None if not dropblock else DropBlock()),
            ('act4', activation(inplace=True)),
            ('gamma4', Multiply(gamma, inplace=True)),
            ('conv4', ScaledStdConv2dSame(
                channels, out_channels, kernel_size=1,
                stride=1, groups=1, eps=1e-5, image_size=256)),
            ('drop4', None if not dropblock else DropBlock()),
            ('semodule', None if not semodule else SEModule(
                out_channels,
                reduction=semodule_reduction,
                divisor=semodule_divisor,
                activation=semodule_activation,
                sigmoid=semodule_sigmoid)),
        ] if m is not None))

    def forward(self, x):
        return self.op(x) * self.gain * self.alpha
