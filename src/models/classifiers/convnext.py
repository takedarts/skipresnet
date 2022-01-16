import collections
from typing import Callable

import torch.nn as nn


class ConvNextClassifier(nn.Sequential):
    '''
    A simple classifiler class.
    This class does only mapping from features to logits.
    '''

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_prob: float,
        normalization: Callable[..., nn.Module],
        **kwargs,
    ) -> None:
        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('norm', normalization(in_channels)),
            ('dout', nn.Dropout2d(p=dropout_prob, inplace=True)),
            ('conv', nn.Conv2d(
                in_channels, num_classes,
                kernel_size=1, padding=0, bias=True)),
        ] if m is not None))
