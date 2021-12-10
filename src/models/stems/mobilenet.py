import collections
from typing import Callable

import torch.nn as nn


class MobileNetStem(nn.Sequential):
    '''
    Stem class for MobileNets or EfficientNets.
    '''

    def __init__(
        self,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('conv', nn.Conv2d(
                3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
        ] if m is not None))
