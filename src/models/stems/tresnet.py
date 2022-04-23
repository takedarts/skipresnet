from typing import Callable
import torch.nn as nn
import torch

try:
    from inplace_abn.functions import inplace_abn
    has_iabn = True
except ImportError:
    has_iabn = False

    def inplace_abn(*args, **kwargs):
        raise ImportError(
            "Please install InplaceABN:'pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.12'")


@torch.jit.script
def space_to_depth(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.size()
    x = x.view(n, c, h // 4, 4, w // 4, 4)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    x = x.view(n, c * 16, h // 4, w // 4)
    return x


class TResNetStem(nn.Module):
    '''Stem class for TResNets.
    '''

    def __init__(
        self,
        out_channels: int,
        normalization: Callable[..., nn.Module],
        **kwargs
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            48, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = normalization(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = space_to_depth(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
