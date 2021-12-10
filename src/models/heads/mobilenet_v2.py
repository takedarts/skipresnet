import collections
from typing import Callable

import torch.nn as nn


class MobileNetV2Head(nn.Sequential):
    '''
    Head class for MobileNet-V2.
    https://arxiv.org/abs/1801.04381
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('conv', nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                padding=0, stride=1, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
        ] if m is not None))
