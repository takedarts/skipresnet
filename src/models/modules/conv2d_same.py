from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functions import conv2d_same, get_same_padding


class Conv2dSame(nn.Conv2d):
    '''
    Convolution 2D class which does padding just like Tensorflow 'SAME' padding.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        image_size: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if (stride[0] == 1 and stride[1] == 1
                and (dilation[0] * (kernel_size[0] - 1)) % 2 == 0
                and (dilation[1] * (kernel_size[1] - 1)) % 2 == 0):
            pad_y = (dilation[0] * (kernel_size[0] - 1)) // 2
            pad_x = (dilation[1] * (kernel_size[1] - 1)) // 2

            self.padding_style = 'static'
            self.padding_size = None
            padding = (pad_y, pad_x)
        elif image_size[0] > 0 and image_size[1] > 0:
            pad_ys = get_same_padding(
                kernel_size[0], stride[0], dilation[0], image_size[0])
            pad_xs = get_same_padding(
                kernel_size[1], stride[1], dilation[1], image_size[1])

            self.padding_style = 'static'
            self.padding_size = pad_xs + pad_ys
            padding = (0, 0)
        else:
            self.padding_style = 'dynamic'
            self.padding_size = None
            padding = (0, 0)

        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.kernel_size: Tuple[int, int] = kernel_size
        self.stride: Tuple[int, int] = stride
        self.dilation: Tuple[int, int] = dilation
        self.groups: int = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_style == 'dynamic':
            return conv2d_same(
                x=x,
                kernel_size=self.kernel_size,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups)
        elif self.padding_size is not None:
            x = F.pad(x, self.padding_size, mode='constant', value=0.0)
            return super().forward(x)
        else:
            return super().forward(x)

    def extra_repr(self) -> str:
        if self.padding_style == 'dynamic':
            return '{}, dynamic'.format(super().extra_repr())
        elif self.padding_size is not None:
            return '{}, padding={}'.format(
                super().extra_repr(), self.padding_size)
        else:
            return super().extra_repr()
