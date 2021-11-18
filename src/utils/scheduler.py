import math

import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


def create_scheduler(
    optimizer: optim.Optimizer,
    train_schedule: str,
    train_epoch: int,
    train_warmup: int,
    **kwargs,
) -> _LRScheduler:
    if train_schedule == 'cosine':
        return CosineAnnealingLR(optimizer, train_epoch, train_warmup)
    elif train_schedule == 'exponential':
        return ExponentialLR(optimizer, train_epoch, train_warmup)
    else:
        raise Exception('unsupported scheduler: {}'.format(train_schedule))


class CosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        T_wup: int,
        eta_min: float = 0.0,
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

    def load_state_dict(self, state_dict: dict) -> None:
        return super().load_state_dict(state_dict)


class ExponentialLR(_LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        T_wup: int,
        rate: float = 0.05,
        last_epoch: int = -1
    ) -> None:
        self.T_max = T_max
        self.T_wup = T_wup

        ghigh, glow = 1.0, 0.0

        for _ in range(20):
            gmid = (ghigh + glow) * 0.5

            if gmid ** T_max > rate:
                ghigh = gmid
            else:
                glow = gmid

        self.gamma = (ghigh + glow) * 0.5

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]

        if self.last_epoch < self.T_wup:
            lrs = [(self.last_epoch + 1) / (self.T_wup + 1) * lr for lr in lrs]

        return lrs
