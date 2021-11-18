import collections

import torch.nn as nn

from .modules import ChannelPad, DropBlock, Multiply, ScaledStdConv2d


class NoneDownsample(nn.Identity):

    def __init__(self, in_channels, out_channels, stride,
                 normalization, activation, dropblock, ** kwargs):
        super().__init__()


class BasicDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride,
                 normalization, activation, dropblock, **kwargs):
        modules = []

        if stride != 1 or in_channels != out_channels:
            modules.extend([
                ('conv', nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, padding=0, bias=False)),
                ('norm', normalization(out_channels)),
                ('drop', None if not dropblock else DropBlock()),
            ])

        super().__init__(collections.OrderedDict(m for m in modules if m[1] is not None))


class TweakedDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride,
                 normalization, activation, dropblock, **kwargs):
        modules = []

        if stride != 1:
            modules.append(('pool', nn.AvgPool2d(
                kernel_size=stride, stride=stride, ceil_mode=True)))

        if in_channels != out_channels:
            modules.extend([
                ('conv', nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0, bias=False)),
                ('norm', normalization(out_channels)),
                ('drop', None if not dropblock else DropBlock()),
            ])

        super().__init__(collections.OrderedDict(m for m in modules if m[1] is not None))


class AverageDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride,
                 normalization, activation, dropblock, **kwargs):
        modules = []

        if stride != 1:
            modules.append(('pool', nn.AvgPool2d(
                kernel_size=stride, stride=stride, ceil_mode=True)))

        if in_channels != out_channels:
            modules.append(('pad', ChannelPad(out_channels - in_channels)))

        super().__init__(collections.OrderedDict(m for m in modules if m[1] is not None))


class NFDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride,
                 activation, dropblock, beta, gamma, **kwargs):
        modules = []

        if stride != 1 or in_channels != out_channels:
            modules.extend([
                ('act', activation(inplace=True)),
                ('beta', Multiply(beta)),
            ])

            if stride != 1:
                modules.append(('pool', nn.AvgPool2d(
                    kernel_size=stride, stride=stride, ceil_mode=True)))

            modules.extend([
                ('conv', ScaledStdConv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0, gamma=gamma)),
                ('drop', None if not dropblock else DropBlock()),
            ])

        super().__init__(collections.OrderedDict(m for m in modules if m[1] is not None))
