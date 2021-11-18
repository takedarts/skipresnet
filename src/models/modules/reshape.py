import torch.nn as nn


class Reshape(nn.Module):

    def __init__(self, *shape):
        super().__init__()

        if len(shape) == 1 and hasattr(shape[0], '__getitem__'):
            self.shape = tuple(shape[0])
        else:
            self.shape = tuple(shape)

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

    def extra_repr(self):
        return str(self.shape)


class ChannelPad(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        if self.size > 0:
            return nn.functional.pad(x, (0, 0, 0, 0, 0, self.size))
        elif self.size < 0:
            return x[:, :self.size]
        else:
            return x

    def extra_repr(self):
        return str(self.size)
