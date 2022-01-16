import collections
from typing import Callable

import torch.nn as nn


class ConvNeXtDownsample(nn.Sequential):
    '''
    Downsample class with ConvNeXT.
    https://arxiv.org/abs/2201.03545
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        normalization: Callable[..., nn.Module],
        **kwargs,
    ) -> None:
        modules = []

        if stride != 1 or in_channels != out_channels:
            modules.extend([
                ('norm', normalization(in_channels)),
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)),
            ])

        super().__init__(collections.OrderedDict(
            (n, m) for n, m in modules if m is not None))
