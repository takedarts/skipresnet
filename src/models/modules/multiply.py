import torch.nn as nn


class Multiply(nn.Module):
    def __init__(self, multiply, inplace=False):
        super().__init__()
        self.multiply = multiply
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.multiply)
        else:
            return x * self.multiply

    def extra_repr(self):
        return '{}, inplace={}'.format(self.multiply, self.inplace)
