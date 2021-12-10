from typing import Callable, List, Tuple

import torch.nn as nn

from ..modules import PatchMerging
from .base import _Block


class SwinBlock(_Block):
    '''
    Block class for Swin Transformers.
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

        if stride != 1 or in_channels != out_channels:
            preprocess: nn.Module = PatchMerging(
                in_channels, out_channels, stride, **kwargs)
        else:
            preprocess = nn.Identity()

        super().__init__(
            index=index,
            settings=settings,
            operation=operation,
            downsample=downsample,
            junction=junction,
            preprocess=preprocess,
            postprocess=nn.Identity(),
            **kwargs)
