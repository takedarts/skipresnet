from typing import Callable, List, Tuple

import torch.nn as nn

from .base import _Block


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
            **kwargs)
