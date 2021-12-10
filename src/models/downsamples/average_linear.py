import collections
from typing import Callable, List, Optional, Tuple

import torch.nn as nn

from ..modules import DropBlock


class AverageLinearDownsample(nn.Sequential):
    '''
    Downsample class with average pooling and linear mapping.
    This module is known as a downsample for ResNet-D.
    https://arxiv.org/abs/1812.01187
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        normalization: Callable[..., nn.Module],
        dropblock: bool,
        **kwargs,
    ) -> None:
        modules: List[Tuple[str, Optional[nn.Module]]] = []

        if stride != 1:
            modules.append(('pool', nn.AvgPool2d(
                kernel_size=stride, stride=stride, ceil_mode=True)))

        if in_channels != out_channels:
            modules.extend([
                ('conv', nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0, bias=False)),
                ('norm', normalization(out_channels)),
                ('drop', None if not dropblock else DropBlock()),
            ])

        super().__init__(collections.OrderedDict(
            (n, m) for n, m in modules if m is not None))
