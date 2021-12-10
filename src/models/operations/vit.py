from typing import Callable

import torch
import torch.nn as nn


class ViTOperation(nn.Module):
    '''
    Operation class for Vision Transformers.
    https://arxiv.org/abs/2010.11929
    '''

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
