import collections
from typing import Callable

import torch.nn as nn


class PreActSmallStem(nn.Sequential):
    '''
    Stem class without strides for pre-activation ResNets.
    '''

    def __init__(
        self,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('conv', nn.Conv2d(
                3, out_channels, kernel_size=3, padding=1, bias=False)),
            ('norm', normalization(out_channels)),
        ] if m is not None))
