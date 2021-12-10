import collections
from typing import Callable

import torch.nn as nn

from ..modules import DropBlock


class LinearDownsample(nn.Sequential):
    '''
    Downsample class with linear mapping.
    This is a default donwsample mudule for ResNets.
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
        modules = []

        if stride != 1 or in_channels != out_channels:
            modules.extend([
                ('conv', nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, padding=0, bias=False)),
                ('norm', normalization(out_channels)),
                ('drop', None if not dropblock else DropBlock()),
            ])

        super().__init__(collections.OrderedDict(
            (n, m) for n, m in modules if m is not None))
