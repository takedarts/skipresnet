import torch
import torch.nn as nn

try:
    from inplace_abn.functions import inplace_abn
except ImportError:
    inplace_abn = None


class InplaceNorm(nn.Module):
    '''Inplace Batch Normalization.
    This class uses `In-Place Activated BatchNorm` without activation
    to build an inplace batch normalization layer.
    https://github.com/mapillary/inplace_abn
    '''

    def __init__(
        self,
        channels: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        if inplace_abn is None:
            raise ImportError(
                'In-Place Activated BatchNorm is required.'
                ' Please install:'
                ' pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.12')

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))
        self.channels = channels
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = inplace_abn(
            x=x,
            weight=self.weight,
            bias=self.bias,
            running_mean=self.running_mean,
            running_var=self.running_var,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
            activation='identity')

        if isinstance(x, tuple):
            return x[0]
        else:
            return x

    def extra_repr(self):
        return '{}, momentum={}, eps={}'.format(
            self.channels, self.momentum, self.eps)
