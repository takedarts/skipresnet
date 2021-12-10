import torch
import torch.nn as nn
from typing import Callable
import itertools


class PatchMerging(nn.Module):
    '''Patch Merging Layer.'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        normalization: Callable[..., nn.Module],
        **kwargs,
    ) -> None:
        super().__init__()
        self.norm = normalization(in_channels * stride * stride)
        self.reduction = nn.Linear(
            in_channels * stride * stride, out_channels, bias=False)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        xs = [
            x[:, h::self.stride, w::self.stride, :]
            for w, h in itertools.product(range(self.stride), repeat=2)]
        x = torch.cat(xs, dim=3).reshape(B, -1, C * self.stride * self.stride)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(B, H // self.stride, W // self.stride, -1)
        x = x.permute(0, 3, 1, 2)

        return x
