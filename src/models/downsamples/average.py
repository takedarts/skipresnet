import collections
from typing import List, Tuple

import torch.nn as nn

from ..modules import ChannelPad


class AverageDownsample(nn.Sequential):
    '''
    Downsample class with average pooling.
    Required channels are padded if input channels are less than output channels.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        **kwargs,
    ) -> None:
        modules: List[Tuple[str, nn.Module]] = []

        if stride != 1:
            modules.append(('pool', nn.AvgPool2d(
                kernel_size=stride, stride=stride, ceil_mode=True)))

        if in_channels != out_channels:
            modules.append(('pad', ChannelPad(out_channels - in_channels)))

        super().__init__(collections.OrderedDict(modules))
