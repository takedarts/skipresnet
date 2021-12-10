import collections
from typing import Callable, List, Optional, Tuple

import torch.nn as nn
from timm.models.layers import ScaledStdConv2dSame

from ..modules import DropBlock, Multiply


class NFDownsample(nn.Sequential):
    '''
    Downsample class for Normalizer-Free ResNets (NFNets).
    https://arxiv.org/abs/2102.06171
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        activation: Callable[..., nn.Module],
        dropblock: bool,
        beta: float,
        gamma: float,
        **kwargs,
    ) -> None:
        modules: List[Tuple[str, Optional[nn.Module]]] = []

        if stride != 1 or in_channels != out_channels:
            modules.extend([
                ('act1', activation(inplace=True)),
                ('gamma', Multiply(gamma, inplace=True)),
                ('beta', Multiply(beta, inplace=True)),
            ])

            if stride != 1:
                modules.append(('pool', nn.AvgPool2d(
                    kernel_size=stride, stride=stride, ceil_mode=True)))

            modules.extend([
                ('conv', ScaledStdConv2dSame(
                    in_channels, out_channels, kernel_size=1,
                    stride=1, padding='same', eps=1e-5)),
                ('drop', None if not dropblock else DropBlock()),
            ])

        super().__init__(collections.OrderedDict(
            (n, m) for n, m in modules if m is not None))
