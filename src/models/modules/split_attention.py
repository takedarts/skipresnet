import torch.nn as nn
import collections


class SplitAttentionModule(nn.Module):
    def __init__(self, out_channels, radix, groups,
                 normalization, activation, reduction=4):
        super().__init__()

        channels = max(out_channels * radix // reduction, 32)
        channels = (channels // 8) * 8

        self.op = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(
                out_channels, channels, 1, padding=0, groups=groups, bias=True)),
            ('norm1', normalization(channels)),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels * radix, 1, padding=0, groups=groups, bias=True)),
        ]))

        self.radix = radix
        self.groups = groups

    def forward(self, x):
        B, _, H, W = x.shape
        w = x.reshape(B, self.radix, -1, H, W)
        w = w.sum(dim=1).mean(dim=(2, 3), keepdims=True)
        w = self.op(w)

        if self.radix > 1:
            w = w.reshape(B, self.groups, self.radix, -1).transpose(1, 2)
            w = w.softmax(dim=1)
            w = w.reshape(B, self.radix, -1, 1, 1)
            x = x.reshape(B, self.radix, -1, H, W)
            x = (x * w).sum(dim=1)
        else:
            w = w.sigmoid()
            x = x * w

        return x
