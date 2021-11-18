import torch.nn as nn
import torch.optim as optim

from .optimizers import SAM

NORM_CLASSES = set([
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
])

BIAS_CLASSES = set([
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
])


def _isinstance(obj, classes):
    for cls in classes:
        if isinstance(obj, cls):
            return True

    return False


def create_optimizer(
    model: nn.Module,
    train_optim: str,
    train_lr: float,
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
            if _isinstance(module, NORM_CLASSES):
                norm_names.add(name)
            elif _isinstance(module, BIAS_CLASSES):
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
        return optim.SGD(parameters, lr=train_lr, momentum=0.9, nesterov=True)
    elif train_optim == 'rmsprop':
        return optim.RMSprop(parameters, lr=train_lr, alpha=0.99, momentum=0.9)
    elif train_optim == 'adamw':
        return optim.AdamW(parameters, lr=train_lr)
    elif train_optim == 'sam':
        return SAM(parameters, lr=train_lr, momentum=0.9, nesterov=True)
    else:
        raise Exception('unsupported optimizer: {}'.format(train_optim))
