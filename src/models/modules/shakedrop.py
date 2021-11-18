from ..functions import shakedrop
import torch.nn as nn


class ShakeDrop(nn.Module):

    def __init__(self, drop_prob, alpha_range=[-1, 1]):
        super().__init__()
        self.drop_prob = drop_prob
        self.alpha_range = alpha_range

    def forward(self, x):
        if self.drop_prob != 0:
            return shakedrop(x, self.drop_prob, self.alpha_range, self.training)
        else:
            return x

    def extra_repr(self):
        return 'drop_prob={}, alpha_range={}'.format(self.drop_prob, self.alpha_range)
