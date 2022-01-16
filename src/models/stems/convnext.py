import collections
from typing import Callable

import torch.nn as nn


class ConvNeXtStem(nn.Sequential):
    '''
    Stem class for ConvNeXt.
    https://arxiv.org/abs/2201.03545
    '''

    def __init__(
        self,
        out_channels: int,
        patch_size: int,
        normalization: Callable[..., nn.Module],
        **kwargs,
    ) -> None:
        super().__init__(collections.OrderedDict([
            ('conv', nn.Conv2d(
                3, out_channels, kernel_size=patch_size, stride=patch_size)),
            ('norm', normalization(out_channels)),
        ]))
