import torch.nn as nn
import torch.optim as optim
import torch.nn.modules.batchnorm
from .optimizers import SAM

NORM_CLASSES = (
    torch.nn.modules.batchnorm._BatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
)

BIAS_CLASSES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
)


def create_optimizer(
    model: nn.Module,
    train_optim: str,
    train_lr: float,
    train_momentum: float,
    train_eps: float,
    train_alpha: float,
    train_wdecay: float,
    train_bdecay: bool,
    **kwargs,
) -> optim.Optimizer:
    if train_bdecay:
        parameters = [{
            'params': [p for p in model.parameters() if p.requires_grad],
            'weight_decay': train_wdecay}]
    else:
        norm_names = set()
        bias_names = set()
        nodecay_params = []
        decay_params = []

        for name, module in model.named_modules():
            if isinstance(module, NORM_CLASSES):
                norm_names.add(name)
            elif isinstance(module, BIAS_CLASSES):
                bias_names.add(f'{name}.bias')

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            elif (name.rsplit('.', maxsplit=1)[0] in norm_names
                  or name in bias_names):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        parameters = [
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': decay_params, 'weight_decay': train_wdecay}]

    if train_optim == 'sgd':
        return optim.SGD(
            parameters, lr=train_lr, momentum=train_momentum, nesterov=True)
    elif train_optim == 'rmsprop':
        return optim.RMSprop(
            parameters, lr=train_lr,
            alpha=train_alpha, momentum=train_momentum, eps=train_eps)
    elif train_optim == 'adamw':
        return optim.AdamW(parameters, lr=train_lr, eps=train_eps)
    elif train_optim == 'sam':
        return SAM(
            parameters, lr=train_lr, momentum=train_momentum, nesterov=True)
    else:
        raise Exception('unsupported optimizer: {}'.format(train_optim))
