from ..functions import swish, h_swish
import torch.nn as nn


class Swish(nn.Module):

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, inplace=self.inplace)

    def extra_repr(self):
        return 'inplace={}'.format(self.inplace)


class HSwish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return h_swish(x, inplace=self.inplace)

    def extra_repr(self):
        return 'inplace={}'.format(self.inplace)
