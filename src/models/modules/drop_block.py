import torch
import torch.nn as nn
import functools
import operator
import itertools


class DropBlock(nn.Module):
    '''Drop 2D spacial blocks from input features.

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    '''

    def __init__(self, drop_prob: float = 0.0, block_size: int = 7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        bsize = [min(self.block_size, s) for s in x.shape[2:]]
        cshape = [max(s - b + 1, 1) for s, b in zip(x.shape[2:], bsize)]

        gamma = self.drop_prob / functools.reduce(operator.mul, bsize)
        gamma *= functools.reduce(operator.mul, [s / c for s, c in zip(x.shape[2:], cshape)])

        mask = torch.rand([x.shape[0], 1] + cshape, device=x.device)
        mask = (mask < gamma).float()
        mask = nn.functional.pad(
            mask, list(itertools.chain.from_iterable([b - 1] * 2 for b in bsize[::-1])))
        mask = nn.functional.max_pool2d(mask, kernel_size=bsize, stride=1)
        mask = 1 - mask
        mask_sum = mask.sum()

        if mask_sum != 0:
            mask *= mask.numel() / mask.sum()

        return x * mask

    def extra_repr(self):
        return 'drop_prob={}, block_size={}'.format(self.drop_prob, self.block_size)
