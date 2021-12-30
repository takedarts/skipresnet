import torch
import torch.nn as nn


class StochasticDepth(nn.Module):

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            m = (torch.rand_like(x[:, 0, 0, 0]) >= self.drop_prob).float()
            r = 1.0 - self.drop_prob
            return x * m[:, None, None, None] / r

    def extra_repr(self):
        return 'drop={}'.format(self.drop_prob)
