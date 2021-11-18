import torch.nn as nn


def h_sigmoid(x, inplace=False):
    return nn.functional.relu6(x + 3, inplace=inplace) / 6
