from typing import Any, List, Tuple

import torch
import torch.nn as nn


class AddJunction(nn.Module):
    '''
    Junction class which adds the previous tensor.
    This module is a default junction of ResNets.
    '''

    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        **kwargs
    ) -> None:
        super().__init__()

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        return y + x[-1]
