"""
This code is imported from a repository of pytorch-image-models.
https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.
    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692
    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 gamma=1.0, eps=1e-5, use_layernorm=False):
        if padding is None:
            padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm  # experimental, slightly faster/less GPU memory to hijack LN kernel

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)

        return self.gain * weight

    def forward(self, x):
        return F.conv2d(
            x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
