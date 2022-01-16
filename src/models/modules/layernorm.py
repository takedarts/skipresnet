import torch
import torch.nn as nn


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, 'Shape of input tensor must be (B, C, H, W)'
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x
