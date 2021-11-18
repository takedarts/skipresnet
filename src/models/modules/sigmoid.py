from ..functions import h_sigmoid
import torch.nn as nn


class HSigmoid(nn.Module):

    def __init__(self, inplace=False, *args, **kwargs):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return h_sigmoid(x, inplace=self.inplace)

    def extra_repr(self):
        return 'inplace={}'.format(self.inplace)
