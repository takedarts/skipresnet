import collections
from typing import Callable

import torch
import torch.nn as nn

from .modules import ScaledStdConv2d


class BasicHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs
    ) -> None:
        super().__init__()


class PreActHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict(m for m in [
            ('norm', normalization(in_channels)),
            ('act', activation(inplace=True)),
        ] if m[1] is not None))


class MobileNetV2Head(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
        ] if m[1] is not None))


class MobileNetV3Head(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        channels = round(out_channels * 0.75)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0, stride=1, bias=False)),
            ('norm1', normalization(channels)),
            ('act1', activation(inplace=True)),
            ('pool', nn.AdaptiveAvgPool2d(1)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0, stride=1, bias=True)),
            ('act2', activation(inplace=True)),
        ] if m[1] is not None))


class NFHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Callable[..., nn.Module],
        gamma: float,
        **kwargs,
    ) -> None:
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', ScaledStdConv2d(
                in_channels, out_channels, kernel_size=1, padding=0, stride=1, gamma=gamma)),
            ('act', activation(inplace=True)),
        ] if m[1] is not None))


class ViTHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__()
        self.norm = normalization(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(3).permute(0, 2, 1)
        x = self.norm(x)
        x = x[:, 0, :, None, None]
        return x
