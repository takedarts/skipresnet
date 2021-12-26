import torch.nn as nn
import numpy as np


class StochasticDepth(nn.Module):

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            d = float(np.random.rand() >= self.drop_prob)
            r = 1.0 - self.drop_prob
            return x * d / r

    def extra_repr(self):
        return 'drop={}'.format(self.drop_prob)
