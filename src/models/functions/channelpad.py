import torch
import torch.nn.functional as F


def channelpad(x: torch.Tensor, padding: int) -> torch.Tensor:
    if padding > 0:
        return F.pad(x, (0, 0, 0, 0, 0, padding))
    elif padding < 0:
        return x[:, :padding]
    else:
        return x
