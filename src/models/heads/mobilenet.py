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


class MobileNetV3Head(nn.Sequential):
    '''
    Head class for MobileNet-V3.
    https://arxiv.org/abs/1905.02244
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        channels = round(out_channels * 0.75)

        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1,
                padding=0, stride=1, bias=False)),
            ('norm1', normalization(channels)),
            ('act1', activation(inplace=True)),
            ('pool', nn.AdaptiveAvgPool2d(1)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=1,
                padding=0, stride=1, bias=True)),
            ('act2', activation(inplace=True)),
        ] if m is not None))
