from ..functions import signal_augment
import torch.nn as nn


class SignalAugmentation(nn.Module):

    def __init__(self, std, dims=1):
        super().__init__()
        self.std = std
        self.dims = dims

    def forward(self, x):
        if self.training and self.std != 0:
            return signal_augment(x, self.std, self.dims)
        else:
            return x

    def extra_repr(self):
        return 'std={}, dim={}'.format(self.std, self.dims)
