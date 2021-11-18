import torch.nn.functional as F


def swish(x, inplace: bool = False):
    if inplace:
        return x.mul_(x.sigmoid())
    else:
        return x.mul(x.sigmoid())


def h_swish(x, inplace=False):
    y = F.relu6(x + 3.).div_(6.)
    if inplace:
        return x.mul_(y)
    else:
        return x.mul(y)
