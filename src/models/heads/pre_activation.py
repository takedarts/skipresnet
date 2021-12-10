import collections
from typing import Callable

import torch.nn as nn


class PreActivationHead(nn.Sequential):
    '''
    Head class for pre-activation ResNets.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('norm', normalization(in_channels)),
            ('act', activation(inplace=True)),
        ] if m is not None))
