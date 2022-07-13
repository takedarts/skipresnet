import collections
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..modules import ShakeDrop, SignalAugmentation, StochasticDepth


class BaseBlock(nn.Module):
    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        operation: Callable[..., nn.Module],
        downsample: Callable[..., nn.Module],
        junction: Callable[..., nn.Module],
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        dropblock: float,
        shakedrop_prob: float,
        stochdepth_prob: float,
        signalaugment: float,
        preprocess: nn.Module = nn.Identity(),
        postprocess: nn.Module = nn.Identity(),
        downsample_before_block: bool = False,
        use_checkpoint: bool = False,
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

        # if True, checkpoint is used to reduce memory consumption.
        self.use_checkpoint = use_checkpoint

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # preprocess
        x[-1] = self.preprocess(x[-1])

        # downsample
        z = self.downsample(x[-1])

        # layers
        arg = z if self.downsample_before_block else x[-1]

        if self.use_checkpoint:
            y = checkpoint(self.operation, arg)
        else:
            y = self.operation(arg)

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
