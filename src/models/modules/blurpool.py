import torch
import torch.nn as nn


class BlurPool2d(nn.Module):
    def __init__(self, channels, filter_size=3, stride=2, padding=1):
        super().__init__()

        if not hasattr(stride, '__getitem__'):
            stride = (stride, stride)

        if not hasattr(padding, '__getitem__'):
            padding = (padding, padding, padding, padding)
        elif len(padding) != 4:
            padding = (padding[1], padding[1], padding[0], padding[0])

        self.channels = channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        filter = [1]
        for _ in range(1, filter_size):
            filter = [0] + filter + [0]
            filter = [x + y for x, y in zip(filter[:-1], filter[1:])]

        filter = torch.Tensor(filter)
        filter = filter[:, None] * filter[None, :]
        filter = filter / filter.sum()
        self.register_buffer('filter', filter[None, None, :, :].repeat(channels, 1, 1, 1))

    def forward(self, x):
        if sum(self.padding) != 0:
            x = nn.functional.pad(x, self.padding, 'reflect')

        return nn.functional.conv2d(x, self.filter, stride=self.stride, groups=x.shape[1])

    def extra_repr(self):
        return '{}, filter_size={}, stride={}, padding={}'.format(
            self.channels, self.filter_size, self.stride, self.padding)
