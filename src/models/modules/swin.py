import torch
import torch.nn as nn
from typing import Callable, Tuple, Union


class PatchMerging(nn.Module):
    '''Patch Merging Layer.'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int]],
        normalization: Callable[..., nn.Module],
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride)

        channels = in_channels * stride[0] * stride[1]

        self.norm = normalization(channels)
        self.reduction = nn.Linear(channels, out_channels, bias=False)
        self.channels = channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:2], -1, self.stride[0], x.shape[3])
        x = x.reshape(*x.shape[:4], -1, self.stride[1])
        x = x.permute(0, 2, 4, 5, 3, 1)
        height, width = x.shape[1:3]

        x = x.reshape(x.shape[0], -1, self.channels)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(x.shape[0], height, width, -1)
        x = x.permute(0, 3, 1, 2)

        return x
