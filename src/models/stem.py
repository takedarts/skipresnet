from .modules import ScaledStdConv2d

import torch
import torch.nn as nn
import collections
import math


class BasicSmallStem(nn.Sequential):
    def __init__(
        self,
        out_channels: int,
        normalization: nn.Module,
        activation: nn.Module,
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
        ] if m[1] is not None))


class PreActSmallStem(nn.Sequential):
    def __init__(
        self,
        out_channels: int,
        normalization: nn.Module,
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False)),
            ('norm', normalization(out_channels)),
        ] if m[1] is not None))


class BasicLargeStem(nn.Sequential):
    def __init__(
        self,
        out_channels: int,
        normalization: nn.Module,
        activation: nn.Module,
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, padding=1, stride=2)),
        ] if m[1] is not None))


class TweakedLargeStem(nn.Sequential):
    def __init__(
        self,
        out_channels: int,
        normalization: nn.Module,
        activation: nn.Module,
        **kwargs
    ) -> None:
        mid_channels = max(out_channels // 2, 1)
        mid_channels = math.ceil(mid_channels / 8) * 8

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(3, mid_channels, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm1', normalization(mid_channels)),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            ('norm2', normalization(mid_channels)),
            ('act2', activation(inplace=True)),
            ('conv3', nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('act3', activation(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, padding=1, stride=2)),
        ] if m[1] is not None))


class MobileNetStem(nn.Sequential):
    def __init__(
        self,
        out_channels: int,
        normalization: nn.Module,
        activation: nn.Module,
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
        ] if m[1] is not None))


class NFNetStem(nn.Sequential):
    def __init__(
        self,
        out_channels: int,
        activation: nn.Module,
        gamma: float,
        **kwargs
    ) -> None:
        mid_channels = max(out_channels // 8, 1)
        mid_channels = math.ceil(mid_channels / 8) * 8

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', ScaledStdConv2d(
                3, mid_channels * 1,
                kernel_size=3, stride=2, padding=1, gamma=gamma)),
            ('act1', activation(inplace=True)),
            ('conv2', ScaledStdConv2d(
                mid_channels * 1, mid_channels * 2,
                kernel_size=3, stride=1, padding=1, gamma=gamma)),
            ('act2', activation(inplace=True)),
            ('conv3', ScaledStdConv2d(
                mid_channels * 2, mid_channels * 4,
                kernel_size=3, stride=1, padding=1, gamma=gamma)),
            ('act3', activation(inplace=True)),
            ('conv4', ScaledStdConv2d(
                mid_channels * 4, out_channels,
                kernel_size=3, stride=2, padding=1, gamma=gamma)),
        ] if m[1] is not None))


class ViTPatchStem(nn.Module):
    def __init__(
        self,
        out_channels: int,
        patch_size: int,
        num_patches: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            3, out_channels, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.pos_embed = nn.Parameter(torch.zeros(1, out_channels, num_patches + 1, 1))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = self.cls_token.expand(x.shape[0], -1, 1, 1)
        x = self.conv(x)
        x = x.reshape(x.shape[0], x.shape[1], -1, 1)
        x = torch.cat((cls_token, x), dim=2)
        x += self.pos_embed
        return x
