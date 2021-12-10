import collections
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from ..modules import ShakeDrop, SignalAugmentation, StochasticDepth


class _Block(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        operation: Callable[..., nn.Module],
        downsample: Callable[..., nn.Module],
        junction: Callable[..., nn.Module],
        preprocess: nn.Module,
        postprocess: nn.Module,
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

        # process before a block
        self.preprocess = preprocess

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

        # noise
        self.noise = nn.Sequential(
            collections.OrderedDict((n, m) for n, m in [
                ('signalaugment',
                 SignalAugmentation(std=signalaugment)
                 if signalaugment > 0.0 else None),
                ('shakedrop',
                 ShakeDrop(drop_prob=shakedrop * (index + 1) / len(settings))
                 if shakedrop > 0.0 else None),
                ('stochdepth',
                 StochasticDepth(drop_prob=stochdepth * (index + 1) / len(settings))
                 if stochdepth > 0.0 else None),
            ] if m is not None))

        # junction
        self.junction = junction(index, settings, **kwargs)

        # process after a block
        self.postprocess = postprocess

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # preprocess
        x[-1] = self.preprocess(x[-1])

        # operation
        z = self.downsample(x[-1])
        y = self.operation(x[-1])

        # noise
        y = self.noise(y)

        # junction
        x[-1] = z
        y = self.junction(y, x)

        # post process
        y = self.postprocess(y)

        # output
        x.append(y)

        return x


class BasicBlock(_Block):
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
            index=index,
            settings=settings,
            operation=operation,
            downsample=downsample,
            junction=junction,
            preprocess=nn.Identity(),
            postprocess=nn.ReLU(inplace=True),
            **kwargs)
