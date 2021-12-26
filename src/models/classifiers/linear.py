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
        dropout_prob: float,
        dropout_enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(collections.OrderedDict((n, m) for n, m in [
            ('dout', None if not dropout_enabled else nn.Dropout2d(
                p=dropout_prob, inplace=True)),
            ('conv', nn.Conv2d(
                in_channels, num_classes,
                kernel_size=1, padding=0, bias=True)),
        ] if m is not None))


class LinearClassifierWithoutDropout(LinearClassifier):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_prob: float,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            dropout_enabled=False,
            **kwargs)
