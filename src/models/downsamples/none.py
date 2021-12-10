import torch.nn as nn


class NoneDownsample(nn.Identity):
    '''
    This class does nothing.
    This is specified when a downsample module is not used.
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__()
