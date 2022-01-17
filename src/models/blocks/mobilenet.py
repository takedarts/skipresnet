from typing import Callable, List, Tuple

import torch.nn as nn

from ..downsamples import NoneDownsample
from ..junctions import NoneJunction
from .base import _Block


class MobileNetBlock(_Block):
    '''
    Block class for MobileNets and EfficientNets.
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
        in_channels, out_channels, stride = settings[index]

        if downsample == NoneDownsample:
            if stride != 1 or in_channels != out_channels:
                junction = NoneJunction

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
