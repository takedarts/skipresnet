from typing import Callable, List, Tuple

import torch.nn as nn

from ..downsamples import NoneDownsample
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

        # PatchMerging is applied before the block.
        # Here, the number of channels is chaned from `in_channels` to `out_channels`.
        if stride != 1 or in_channels != out_channels:
            preprocess: nn.Module = PatchMerging(
                in_channels, out_channels, stride, **kwargs)
        else:
            preprocess = nn.Identity()

        # remove the downsample module
        # LinearDownsample is specified for SkipJunctions (Skip-SwinTransformer),
        # but the donwsammple modules are not needed for SwinTransformers.
        downsample = NoneDownsample

        super().__init__(
            index=index,
            settings=settings,
            operation=operation,
            downsample=downsample,
            junction=junction,
            preprocess=preprocess,
            postprocess=nn.Identity(),
            **kwargs)
