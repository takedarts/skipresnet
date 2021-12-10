from typing import Callable

import timm.models.swin_transformer
import torch
import torch.nn as nn


class SwinOperation(nn.Module):
    '''
    Operation class for Swin Transformers.
    https://arxiv.org/abs/2103.14030
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int,
        window_size: int,
        shift_size: int,
        attn_heads: int,
        mlp_ratio: float,
        normalization: Callable[..., nn.Module],
        activation: Callable[..., nn.Module],
        operation_type: str,
        **kwargs,
    ) -> None:
        super().__init__()
        self.attn_op = (operation_type.lower() == 'attn')
        self.window_size = window_size
        self.shift_size = shift_size

        if self.attn_op:
            self.attn_norm = normalization(out_channels)
            self.attn_attn = timm.models.swin_transformer.WindowAttention(
                out_channels, (window_size, window_size), attn_heads)
        else:
            mid_channels = int(out_channels * mlp_ratio)
            self.mlp_norm = normalization(out_channels)
            self.mlp_fc1 = nn.Linear(out_channels, mid_channels)
            self.mlp_act = activation(inplace=True)
            self.mlp_fc2 = nn.Linear(mid_channels, out_channels)

        if shift_size > 0:
            # calculate attention mask for SW-MSA
            attn_mask = torch.zeros((1, feature_size, feature_size, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[:, h, w, :] = cnt
                    cnt += 1

            B, H, W, C = attn_mask.shape
            attn_mask = attn_mask.view(B, H // window_size, window_size, W // window_size, window_size, C)
            attn_mask = attn_mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

            attn_mask = attn_mask.view(-1, self.window_size * self.window_size)
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.register_buffer("attn_mask", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_op:  # attention
            return self.forward_attn(x)
        else:  # mlp
            return self.forward_mlp(x)

    def forward_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)

        # normalize
        x = self.attn_norm(x)
        x = x.reshape(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        # nW*B, window_size, window_size, C
        x = x.reshape(
            B, H // self.window_size, self.window_size,
            W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.window_size, self.window_size, C)
        # nW*B, window_size*window_size, C
        x = x.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        x = self.attn_attn(x, mask=self.attn_mask)

        # merge windows
        x = x.reshape(-1, self.window_size, self.window_size, C)
        # B H' W' C
        x = x.view(
            B, H // self.window_size, W // self.window_size,
            self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.permute(0, 3, 1, 2)

        return x

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        x = self.mlp_norm(x)
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x
