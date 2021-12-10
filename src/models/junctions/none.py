
from typing import Any, List, Tuple

import torch
import torch.nn as nn


class NoneJunction(nn.Module):
    '''
    Junction class which does nothing.
    '''

    def __init__(
        self,
        index: int,
        settings: List[Tuple[int, int, int]],
        **kwargs
    ) -> None:
        super().__init__()

    def forward(self, y: torch.Tensor, x: List[Any]) -> torch.Tensor:
        return y
