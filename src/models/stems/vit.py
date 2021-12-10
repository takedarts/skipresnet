import torch
import torch.nn as nn


class ViTPatchStem(nn.Module):
    '''
    Stem class for Vision Transformers.
    https://arxiv.org/abs/2010.11929
    '''

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
