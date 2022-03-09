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
        shakedrop_prob: float,
        stochdepth_prob: float,
        signalaugment: float,
        downsample_before_block: bool,
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

        # layers
        self.downsample_before_block = downsample_before_block

        if downsample_before_block:
            self.operation = operation(
                out_channels, out_channels, stride=1,
                normalization=normalization, activation=activation,
                dropblock=dropblock, **kwargs)
        else:
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
                 ShakeDrop(drop_prob=shakedrop_prob * (index + 1) / len(settings))
                 if shakedrop_prob > 0.0 else None),
                ('stochdepth',
                 StochasticDepth(drop_prob=stochdepth_prob * (index + 1) / len(settings))
                 if stochdepth_prob > 0.0 else None),
            ] if m is not None))

        # junction
        self.junction = junction(index, settings, **kwargs)

        # process after a block
        self.postprocess = postprocess

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # preprocess
        x[-1] = self.preprocess(x[-1])

        # downsample
        z = self.downsample(x[-1])

        # layers
        if self.downsample_before_block:
            y = self.operation(z)
        else:
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
        activation: Callable[..., nn.Module],
        **kwargs,
    ) -> None:
        super().__init__(
            index=index,
            settings=settings,
            operation=operation,
            downsample=downsample,
            junction=junction,
            activation=activation,
            preprocess=nn.Identity(),
            postprocess=activation(inplace=True),
            downsample_before_block=False,
            **kwargs)


class PreActivationBlock(_Block):
    '''
    Block class for pre-actiovation ResNets.
    '''

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
            postprocess=nn.Identity(),
            downsample_before_block=False,
            **kwargs)
