"""
This is an implementation of Conv2d layer with Scaled Weight Standardization
which is based on an implementation in pytorch-image-models:
https://github.com/rwightman/pytorch-image-models.
"""
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functions import pad_same, get_same_padding


class ScaledStdConv2d(nn.Conv2d):
    '''
    Conv2d layer with Scaled Weight Standardization.
    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692
    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int]] = (0, 0),
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        groups: int = 1,
        bias: bool = True,
        gamma: float = 1.0,
        eps: float = 1e-6,
        gain_init: float = 1.0,
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias)

        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(1, self.out_channels, -1)
        weight = F.batch_norm(
            weight, running_mean=None, running_var=None,
            weight=(self.gain * self.scale).view(-1),
            training=True, momentum=0, eps=self.eps)
        weight = weight.reshape_as(self.weight)

        return F.conv2d(
            x, weight=weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups)


class ScaledStdConv2dSame(ScaledStdConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = (1, 1),
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        groups: int = 1,
        bias: bool = True,
        gamma: float = 1.0,
        eps: float = 1e-6,
        gain_init: float = 1.0,
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
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
            gamma=gamma, eps=eps, gain_init=gain_init)

        self.kernel_size: Tuple[int, int] = kernel_size
        self.stride: Tuple[int, int] = stride
        self.dilation: Tuple[int, int] = dilation
        self.groups: int = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_style == 'dynamic':
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
            return super().forward(x)
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
