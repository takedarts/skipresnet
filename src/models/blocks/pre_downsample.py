from typing import Callable, List, Tuple

import torch.nn as nn

from .base import _Block
from ..downsamples import NoneDownsample


class PreDownsampleBlock(_Block):
    '''
    Block class which performs the downsample before maim process.
    '''

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
        **kwargs,
    ) -> None:
        in_channels, out_channels, stride = settings[index]

        preprocess = downsample(
            in_channels, out_channels, stride=stride,
            normalization=normalization, activation=activation,
            dropblock=dropblock, **kwargs)

        super().__init__(
            index=index,
            settings=settings,
            operation=operation,
            downsample=NoneDownsample,
            junction=junction,
            preprocess=preprocess,
            postprocess=nn.Identity(),
            normalization=normalization,
            activation=activation,
            dropblock=dropblock,
            **kwargs)
