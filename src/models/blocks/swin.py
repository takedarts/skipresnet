from typing import Callable, List, Tuple

import torch.nn as nn

from .base import _Block


class SwinBlock(_Block):
    '''
    Block class for Swin Transformers.
    Downsample is performed before the main block.
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
        super().__init__(
            index=index,
            settings=settings,
            operation=operation,
            downsample=downsample,
            junction=junction,
            preprocess=nn.Identity(),
            postprocess=nn.Identity(),
            normalization=normalization,
            activation=activation,
            dropblock=dropblock,
            downsample_before_block=True,
            **kwargs)
