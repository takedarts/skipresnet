import collections
import math
from typing import Callable

import torch.nn as nn

from ..modules import Multiply, ScaledStdConv2dSame


class NFNetStem(nn.Sequential):
    '''
    Stem class for Normalizer-Free ResNets (NFNets).
    https://arxiv.org/abs/2102.06171
    '''

    def __init__(
        self,
        out_channels: int,
        activation: Callable[..., nn.Module],
        gamma: float,
        **kwargs
    ) -> None:
        mid_channels = max(out_channels // 8, 1)
        mid_channels = math.ceil(mid_channels / 8) * 8

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', ScaledStdConv2dSame(
                3, mid_channels * 1, kernel_size=3,
                stride=2, eps=1e-05, image_size=256)),
            ('act1', activation(inplace=True)),
            ('mul1', Multiply(gamma, inplace=True)),
            ('conv2', ScaledStdConv2dSame(
                mid_channels * 1, mid_channels * 2,
                kernel_size=3, stride=1, eps=1e-05, image_size=256)),
            ('act2', activation(inplace=True)),
            ('mul2', Multiply(gamma, inplace=True)),
            ('conv3', ScaledStdConv2dSame(
                mid_channels * 2, mid_channels * 4,
                kernel_size=3, stride=1, eps=1e-05, image_size=256)),
            ('act3', activation(inplace=True)),
            ('mul3', Multiply(gamma, inplace=True)),
            ('conv4', ScaledStdConv2dSame(
                mid_channels * 4, out_channels, kernel_size=3,
                stride=2, eps=1e-05, image_size=256)),
        ] if m[1] is not None))
