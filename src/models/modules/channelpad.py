import torch
import torch.nn as nn

from ..functions import channelpad


class ChannelPad(nn.Module):
    def __init__(self, padding: int) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return channelpad(x, self.padding)

    def extra_repr(self) -> str:
        return str(self.padding)
