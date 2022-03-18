import math

import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


def create_scheduler(
    optimizer: optim.Optimizer,
    train_schedule: str,
    train_epoch: int,
    train_warmup: int,
    train_lastlr: float,
    **kwargs,
) -> _LRScheduler:
    if train_schedule == 'constant':
        return ConstantLR(optimizer)
    elif train_schedule == 'cosine':
        return CosineAnnealingLR(optimizer, train_epoch, train_warmup, train_lastlr)
    elif train_schedule == 'exponential':
        return ExponentialLR(optimizer, train_epoch, train_warmup, train_lastlr)
    else:
        raise Exception('unsupported scheduler: {}'.format(train_schedule))


class ConstantLR(_LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        last_epoch: int = -1
    ) -> None:
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs


class CosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        T_wup: int,
        eta_min: float,
        last_epoch: int = -1
    ) -> None:
        self.T_max = T_max
        self.T_wup = T_wup
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [self.eta_min + (base_lr - self.eta_min)
               * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
               for base_lr in self.base_lrs]

        if self.last_epoch < self.T_wup:
            lrs = [(self.last_epoch + 1) / (self.T_wup + 1) * lr for lr in lrs]

        return lrs


class ExponentialLR(_LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        T_wup: int,
        eta_min: float,
        last_epoch: int = -1
    ) -> None:
        if eta_min == 0.0:
            rate = 0.01
        else:
            rate = eta_min / optimizer.param_groups[0]['lr']

        self.T_max = T_max
        self.T_wup = T_wup
        self.gamma = rate ** (1 / T_max)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]

        if self.last_epoch < self.T_wup:
            lrs = [(self.last_epoch + 1) / (self.T_wup + 1) * lr for lr in lrs]

        return lrs
