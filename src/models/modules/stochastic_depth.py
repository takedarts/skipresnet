import torch.nn as nn
import numpy


class StochasticDepth(nn.Module):

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0:
            return x
        elif not self.training:
            return x * (1.0 - self.drop_prob)
        elif numpy.random.rand() < self.drop_prob:
            return x * 0
        else:
            return x

    def extra_repr(self):
        return 'drop={}'.format(self.drop_prob)
