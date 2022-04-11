import torch.nn as nn

from .base import BaseBlock


class ResNetBlock(BaseBlock):
    '''
    Block class for ResNets.
    '''

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            postprocess=nn.ReLU(inplace=True),
            **kwargs)
