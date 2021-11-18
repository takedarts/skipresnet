import collections
from typing import Callable

import torch
import torch.nn as nn

from .modules import (BlurPool2d, DropBlock, Multiply, ScaledStdConv2d,
                      SEModule, SKConv2d, SplitAttentionModule)


class BasicOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False)),
            ('norm2', normalization(out_channels)),
            ('drop2', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class BottleneckOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck * groups)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class SelectedKernelOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck * groups)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=stride, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', SKConv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, radix=radix, groups=groups)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class PreActBasicOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('norm1', normalization(in_channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False)),
        ] if m[1] is not None))


class SingleActBasicOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('norm1', normalization(in_channels)),
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class SingleActBottleneckOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck * groups)

        super().__init__(collections.OrderedDict(m for m in [
            ('norm1', normalization(in_channels)),
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm3', normalization(channels)),
            ('drop3', None if not dropblock else DropBlock()),
            ('act3', activation(inplace=True)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm4', normalization(out_channels)),
            ('drop4', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class TweakedBottleneckOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('pool', None if stride == 1 else BlurPool2d(channels, stride=stride)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class TweakedSlectedKernelOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', SKConv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, radix=radix, groups=groups)),
            ('drop2', None if not dropblock else DropBlock()),
            ('pool', None if stride == 1 else BlurPool2d(channels, stride=stride)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class MobileNetOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, expansion,
                 normalization, activation, dropblock, seoperation,
                 seoperation_reduction, seoperation_sigmoid, **kwargs):
        channels = int(in_channels * expansion)
        modules = []

        if in_channels != channels:
            modules.extend([
                ('conv1', nn.Conv2d(
                    in_channels, channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False)),
                ('norm1', normalization(channels)),
                ('drop1', None if not dropblock else DropBlock()),
                ('act1', activation(inplace=True)),
            ])

        modules.extend([
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=kernel, padding=kernel // 2,
                stride=stride, groups=channels, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('semodule', None if not seoperation else SEModule(
                channels,
                semodule_reduction=seoperation_reduction,
                semodule_activation=activation,
                semodule_sigmoid=seoperation_sigmoid)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ])

        super().__init__(collections.OrderedDict(m for m in modules if m[1] is not None))


class SplitAttentionOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels * radix, kernel_size=3, padding=1,
                stride=1, groups=groups * radix, bias=False)),
            ('norm2', normalization(channels * radix)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('attention', SplitAttentionModule(
                channels, radix=radix, groups=groups,
                normalization=normalization, activation=activation)),
            ('downsample', None if stride == 1 else nn.AvgPool2d(
                kernel_size=3, stride=stride, padding=1)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class DenseNetOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, growth, expansion,
                 normalization, activation, dropblock, **kwargs):
        if stride != 1:
            super().__init__(collections.OrderedDict(m for m in [
                ('norm1', normalization(in_channels)),
                ('act1', activation(inplace=True)),
                ('conv1', nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False)),
                ('pool1', nn.AvgPool2d(kernel_size=2, stride=stride)),
            ] if m[1] is not None))
        else:
            channels = growth * expansion
            super().__init__(collections.OrderedDict(m for m in [
                ('norm1', normalization(in_channels)),
                ('drop1', None if not dropblock else DropBlock()),
                ('act1', activation(inplace=True)),
                ('conv1', nn.Conv2d(
                    in_channels, channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False)),
                ('norm2', normalization(channels)),
                ('drop2', None if not dropblock else DropBlock()),
                ('act2', activation(inplace=True)),
                ('conv2', nn.Conv2d(
                    channels, growth, kernel_size=3, padding=1,
                    stride=1, bias=False)),
            ] if m[1] is not None))


class NFOperation(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 activation, dropblock, seoperation, seoperation_reduction, seoperation_sigmoid,
                 alpha, beta, gamma, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('act1', activation(inplace=True)),
            ('beta', Multiply(beta, inplace=True)),
            ('conv1', ScaledStdConv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, gamma=gamma)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv2', ScaledStdConv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, gamma=gamma)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act3', activation(inplace=True)),
            ('conv3', ScaledStdConv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, groups=groups, gamma=gamma)),
            ('drop3', None if not dropblock else DropBlock()),
            ('semodule', None if not seoperation else SEModule(
                channels, reduction=seoperation_reduction,
                activation=nn.ReLU, sigmoid=seoperation_sigmoid)),
            ('act4', activation(inplace=True)),
            ('conv4', ScaledStdConv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, gamma=gamma)),
            ('drop4', None if not dropblock else DropBlock()),
            ('alpha', Multiply(alpha, inplace=True)),
        ] if m[1] is not None))


class ViTOperation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attn_heads: int,
        mlp_ratio: float,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        operation_type: str,
        **kwargs,
    ) -> None:
        super().__init__()
        self.attn_op = (operation_type.lower() == 'attn')

        if self.attn_op:
            self.attn_heads = attn_heads
            self.attn_scale = (in_channels // attn_heads) ** -0.5
            self.attn_norm = normalization(in_channels)
            self.attn_qkv = nn.Linear(in_channels, in_channels * 3)
            self.attn_proj = nn.Linear(in_channels, in_channels)
        else:
            mid_channels = int(in_channels * mlp_ratio)
            self.mlp_norm = normalization(in_channels)
            self.mlp_fc1 = nn.Linear(in_channels, mid_channels)
            self.mlp_act = activation(inplace=True)
            self.mlp_fc2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).squeeze(3)

        if self.attn_op:  # attention
            x = self.attn_norm(x)
            b, n, c = x.shape
            qkv = self.attn_qkv(x)
            qkv = qkv.reshape(b, n, 3, self.attn_heads, c // self.attn_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.attn_scale
            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(b, n, c)
            x = self.attn_proj(x)
        else:  # mlp
            x = self.mlp_norm(x)
            x = self.mlp_fc1(x)
            x = self.mlp_act(x)
            x = self.mlp_fc2(x)

        return x.permute(0, 2, 1).unsqueeze(3)
