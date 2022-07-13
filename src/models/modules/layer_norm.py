import torch
import torch.nn as nn


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, 'Shape of input tensor must be (B, C, H, W)'

        s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.eps)
        if self.weight is not None:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]

        return x
