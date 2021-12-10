import torch.nn as nn


class NoneHead(nn.Sequential):
    '''
    This head module does nothing.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs
    ) -> None:
        super().__init__()
