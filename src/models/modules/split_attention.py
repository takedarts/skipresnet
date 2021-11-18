import torch.nn as nn
import collections
import math


class SplitAttentionModule(nn.Module):

    def __init__(self, out_channels, radix, groups,
                 normalization, activation, reduction=4):
        super().__init__()
        channels = max(out_channels * radix // reduction, 1)
        channels = math.ceil(channels / 8) * 8

        self.op = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(
                out_channels, channels, 1, padding=0, groups=groups, bias=True)),
            ('norm1', normalization(channels)),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels * radix, 1, padding=0, groups=groups, bias=True)),
        ]))

        self.radix = radix

    def forward(self, x):
        w = x.reshape(x.shape[0], self.radix, -1, *x.shape[2:])
        w = w.sum(dim=1).mean(dim=(2, 3), keepdims=True)
        w = self.op(w)
        w = w.reshape(w.shape[0], self.radix, -1, *w.shape[2:])
        w = w.softmax(dim=1)

        x = x.reshape(*w.shape[:3], *x.shape[2:])
        x = (x * w).sum(dim=1)

        return x
