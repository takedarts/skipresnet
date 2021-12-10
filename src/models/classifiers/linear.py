import torch.nn as nn
import collections


class LinearClassifier(nn.Sequential):
    '''
    A simple classifiler class.
    This class does only mapping from features to logits.
    '''

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        **kwargs,
    ) -> None:
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(
                in_channels, num_classes,
                kernel_size=1, padding=0, bias=True)),
        ] if m[1] is not None))
