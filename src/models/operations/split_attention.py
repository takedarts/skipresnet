import collections
from typing import Callable, Union
import torch.nn as nn
from ..modules import DropBlock, SplitAttentionModule


class SplitAttentionOperation(nn.Sequential):
    '''
    Operation class for Split Attention Networks (ResNeSt).
    https://arxiv.org/abs/2004.08955
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        radix: int,
        groups: int,
        bottleneck: Union[int, float],
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        dropblock: bool,
        avg_first: bool,
        **kwargs,
    ) -> None:
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('pool1', None if stride == 1 or not avg_first else nn.AvgPool2d(
                kernel_size=3, stride=stride, padding=1)),
            ('conv2', nn.Conv2d(
                channels, channels * radix, kernel_size=3, padding=1,
                stride=1, groups=groups * radix, bias=False)),
            ('norm2', normalization(channels * radix)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('attention', SplitAttentionModule(
                channels, radix=radix, groups=groups,
                normalization=normalization, activation=activation)),
            ('pool2', None if stride == 1 or avg_first else nn.AvgPool2d(
                kernel_size=3, stride=stride, padding=1)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m is not None))
