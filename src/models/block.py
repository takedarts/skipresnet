from .modules import ShakeDrop, SignalAugmentation, SEModule, StochasticDepth
from .downsample import NoneDownsample
from .junction import NoneJunction

import torch.nn as nn
import collections
from typing import List, Tuple, Callable


class _Block(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        operation: Callable[..., nn.Module],
        downsample: Callable[..., nn.Module],
        junction: Callable[..., nn.Module],
        subsequent: nn.Module,
        semodule: bool,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        dropblock: float,
        shakedrop: float,
        stochdepth: float,
        signalaugment: float,
        **kwargs
    ) -> None:
        super().__init__()
        in_channels, out_channels, stride = settings[index]

        # downsample
        self.downsample = downsample(
            in_channels, out_channels, stride=stride,
            normalization=normalization, activation=activation,
            dropblock=dropblock, **kwargs)

        # convolution layers
        self.operation = operation(
            in_channels, out_channels, stride=stride,
            normalization=normalization, activation=activation,
            dropblock=dropblock, **kwargs)

        # attention modules
        if semodule:
            self.semodule = SEModule(out_channels, **kwargs)

        # noise
        self.noise = nn.Sequential(collections.OrderedDict([
            ('signalaugment', SignalAugmentation(std=signalaugment)),
            ('shakedrop', ShakeDrop(drop_prob=shakedrop * (index + 1) / len(settings))),
            ('stochdepth', StochasticDepth(drop_prob=stochdepth * (index + 1) / len(settings))),
        ]))

        # junction
        self.junction = junction(index, settings, **kwargs)

        # activation after a block
        self.subsequent = subsequent

    def forward(self, x):
        # operation
        z = self.downsample(x[-1])
        y = self.operation(x[-1])

        # attention
        if hasattr(self, 'semodule'):
            y = self.semodule(y)

        # noise
        y = self.noise(y)

        # junction
        x[-1] = z
        y = self.junction(y, x)

        # output
        x.append(self.subsequent(y))

        return x


class BasicBlock(_Block):
    def __init__(self, index, settings, operation, downsample, junction, **kwargs):
        super().__init__(
            index, settings, operation, downsample, junction, nn.ReLU(inplace=True), **kwargs)


class PreActBlock(_Block):
    def __init__(self, index, settings, operation, downsample, junction, **kwargs):
        super().__init__(
            index, settings, operation, downsample, junction, nn.Identity(), **kwargs)


class MobileNetBlock(_Block):
    def __init__(self, index, settings, operation, downsample, junction, **kwargs):
        in_channels, out_channels, stride = settings[index]

        if downsample == NoneDownsample:
            if stride != 1 or in_channels != out_channels:
                junction = NoneJunction

        super().__init__(
            index, settings, operation, downsample, junction, nn.Identity(), **kwargs)


class DenseNetBlock(_Block):
    def __init__(self, index, settings, operation, downsample, junction, **kwargs):
        in_channels, _, stride = settings[index]

        if stride != 1:
            junction = NoneJunction
            kwargs['semodule'] = False
            kwargs['shakedrop'] = 0.0
            kwargs['stochdepth'] = 0.0
            kwargs['signalaugment'] = 0.0

        super().__init__(
            index, settings, operation, downsample, junction, nn.Identity(), **kwargs)


class ViTBlock(_Block):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        operation: Callable[..., nn.Module],
        downsample: Callable[..., nn.Module],
        junction: Callable[..., nn.Module],
        **kwargs,
    ) -> None:
        super().__init__(
            index, settings, operation, downsample, junction, nn.Identity(), **kwargs)
