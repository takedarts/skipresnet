import torch.nn as nn
import collections


class SEModule(nn.Module):

    def __init__(self, channels, semodule_reduction, semodule_activation=nn.ReLU,
                 semodule_sigmoid=nn.Sigmoid, semodule_gain=1.0, **kwargs):
        super().__init__()
        hidden_channels = max(channels // semodule_reduction, 1)

        self.op = nn.Sequential(collections.OrderedDict([
            ('pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('conv1', nn.Conv2d(channels, hidden_channels, kernel_size=1, padding=0)),
            ('act1', semodule_activation(inplace=True)),
            ('conv2', nn.Conv2d(hidden_channels, channels, kernel_size=1, padding=0)),
            ('sigmoid', semodule_sigmoid()),
        ]))
        self.gain = semodule_gain

    def forward(self, x):
        return (x * self.op(x)).mul_(self.gain)
