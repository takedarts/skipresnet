import collections
from typing import Callable

import torch.nn as nn

from ..modules import Conv2dSame


class EfficientNetV2Stem(nn.Sequential):
    '''
    Stem class for EfficientNetV2s.
    '''

    def __init__(
        self,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('conv', Conv2dSame(
                3, out_channels, kernel_size=3,
                stride=2, bias=False, image_size=256)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
        ] if m is not None))
