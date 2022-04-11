from typing import Callable, List, Tuple
from .base import BaseBlock
import torch.nn as nn
from ..junctions import NoneJunction


class DenseNetBlock(BaseBlock):
    '''
    Block class for DenseNets.
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
        _, _, stride = settings[index]

        if stride != 1:
            junction = NoneJunction
            kwargs['semodule'] = False
            kwargs['shakedrop'] = 0.0
            kwargs['stochdepth'] = 0.0
            kwargs['signalaugment'] = 0.0

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
