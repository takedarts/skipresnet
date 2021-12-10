from typing import Any, List, Tuple

import torch
import torch.nn as nn


class ConcatJunction(nn.Module):
    '''
    Junction class which concatenates the previous tensor.
    This module is used for DenseNets.
    '''

    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        **kwargs
    ) -> None:
        super().__init__()

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        return torch.cat([x[-1], y], dim=1)
