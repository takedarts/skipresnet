import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *shape) -> None:
        super().__init__()

        if len(shape) == 1 and hasattr(shape[0], '__getitem__'):
            self.shape = tuple(shape[0])
        else:
            self.shape = tuple(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], *self.shape)

    def extra_repr(self) -> str:
        return str(self.shape)
