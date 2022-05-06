from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from models import Model
from models.operations.swin import WindowAttention

from .optimizers import SAM, RMSpropTF

# Normalization classes.
NORM_CLASSES = (
    torch.nn.modules.batchnorm._BatchNorm,
    torch.nn.modules.instancenorm._InstanceNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    nn.CrossMapLRN2d,
)


def create_optimizer(
    model: Model,
    train_optim: str,
    train_lr: float,
    train_layerlr_ratio: float,
    train_momentum: float,
    train_eps: float,
    train_alpha: float,
    train_wdecay: float,
    train_bdecay: bool,
    **kwargs,
) -> optim.Optimizer:
    # Make parameter groups.
    if train_layerlr_ratio != 1.0:
        parameters = _create_layer_decayed_parameter_group(
            model=model,
            train_lr=train_lr,
            train_wdecay=train_wdecay,
            train_bdecay=train_bdecay,
            train_layerlr_ratio=train_layerlr_ratio,
        )
    else:
        parameters = _create_parameter_group(
            modules=[model],
            train_lr=train_lr,
            train_wdecay=train_wdecay,
            train_bdecay=train_bdecay)

    # Make the optimizer.
    if train_optim == 'sgd':
        return optim.SGD(
            parameters, lr=train_lr, momentum=train_momentum, nesterov=True)
    elif train_optim == 'adamw':
        return optim.AdamW(parameters, lr=train_lr, eps=train_eps)
    elif train_optim == 'rmsprop':
        return optim.RMSprop(
            parameters, lr=train_lr, alpha=train_alpha,
            momentum=train_momentum, eps=train_eps)
    elif train_optim == 'rmsproptf':
        return RMSpropTF(
            parameters, lr=train_lr, alpha=train_alpha,
            momentum=train_momentum, eps=train_eps)
    elif train_optim == 'sam':
        return SAM(
            parameters, lr=train_lr, momentum=train_momentum, nesterov=True)
    else:
        raise Exception('unsupported optimizer: {}'.format(train_optim))


def _create_layer_decayed_parameter_group(
    model: Model,
    train_lr: float,
    train_wdecay: float,
    train_bdecay: bool,
    train_layerlr_ratio: float,
) -> List[Dict[str, Any]]:
    '''Create parameter groups when layer-wise learning decay is applied.
    A argument `train_layerdecay` means the ratio of the learning rate for the
    stem block to the learing rate for the classifier.

    Args:
        model: Model object
        train_lr: Initial learning rate
        train_wdecay: Weight decay
        train_bdecay: True if weight decay is applied to biases
        train_layerlr_ratio: Input/Output LR ratio of Layer-wise learning rate decay

    Returns:
        List of parameter groups
    '''
    decay = train_layerlr_ratio**(1 / (len(model.blocks) + 1))

    parameters = _create_parameter_group(
        modules=[model.head, model.classifier],
        train_lr=train_lr,
        train_wdecay=train_wdecay,
        train_bdecay=train_bdecay)

    for i, block in enumerate(model.blocks[::-1]):
        parameters.extend(_create_parameter_group(
            modules=[block],
            train_lr=train_lr * decay**(i + 1),
            train_wdecay=train_wdecay,
            train_bdecay=train_bdecay
        ))

    parameters.extend(_create_parameter_group(
        modules=[model.stem],
        train_lr=train_lr * decay**(len(model.blocks) + 1),
        train_wdecay=train_wdecay,
        train_bdecay=train_bdecay
    ))

    return parameters


def _create_parameter_group(
    modules: List[nn.Module],
    train_lr: float,
    train_wdecay: float,
    train_bdecay: bool,
) -> List[Dict[str, Any]]:
    '''Create parameter groups.

    Args:
        modules: List of target modules
        train_lr: Initial learning rate
        train_wdecay: Weight decay
        train_bdecay: True if weight decay is applied to biases

    Returns:
        List of parameter groups
    '''

    # If weight decay is applied to all parameters,
    # a parameter group which contains all parameters is created.
    if train_bdecay:
        params: List[nn.Parameter] = []
        for module in modules:
            params.extend(p for p in module.parameters() if p.requires_grad)
        return [dict(params=params, lr=train_lr, weight_decay=train_wdecay)]

    # If weight decay is not applied to nomalizations and biases,
    # following two parameter groups are created:
    # nodecay_group: parameters which are updated without weight decay.
    # decay_group: parameters which are updated with weight decay.
    # A parameter `relative_position_bias_table` in WindowAttention is
    # considered a bias parameter.
    nodecay_group: List[nn.Parameter] = []
    decay_group: List[nn.Parameter] = []

    for module in modules:
        children = {n: m for n, m in module.named_modules()}
        children[''] = module

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if '.' in name:
                parent_name, param_name = name.rsplit('.', maxsplit=1)
            else:
                parent_name, param_name = '', name

            if (isinstance(children[parent_name], NORM_CLASSES)
                    or param_name == 'bias'
                    or (isinstance(children[parent_name], WindowAttention)
                        and param_name == 'relative_position_bias_table')):
                nodecay_group.append(param)
            else:
                decay_group.append(param)

    return [
        dict(params=nodecay_group, lr=train_lr, weight_decay=0.0),
        dict(params=decay_group, lr=train_lr, weight_decay=train_wdecay),
    ]
