from typing import Dict, List, Tuple

import models
import pytest
import torch
import torch.nn as nn
from models.operations.swin import WindowAttention
from utils import create_optimizer

# Scenarios: model_name, layerlrdecay, bdecay
TEST_PARAMETERS: List[Tuple[str, float, bool]] = [
    ('ResNet-50', 1.0, True),
    ('ResNet-50', 0.9, True),
    ('ResNet-50', 1.0, False),
    ('ResNet-50', 0.9, False),
    ('ViTSmallPatch16-224', 1.0, False),
    ('ViTSmallPatch16-224', 0.9, False),
    ('SwinTinyPatch4-224', 1.0, False),
    ('SwinTinyPatch4-224', 0.9, False),
]

# Normalization classes
NORM_CLASSES = (
    torch.nn.modules.batchnorm._BatchNorm,
    torch.nn.modules.instancenorm._InstanceNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    nn.CrossMapLRN2d,
)


@pytest.mark.parametrize(
    'model_name,layerlr_ratio,bdecay',
    TEST_PARAMETERS,
    ids=[f'{n}-{d}-{b}' for n, d, b in TEST_PARAMETERS]
)
def test_optimizer(model_name: str, layerlr_ratio: float, bdecay: bool) -> None:
    _test_optimizer(
        model=models.create_model(model_name, 'imagenet'),
        layerlr_ratio=layerlr_ratio,
        bdecay=bdecay,
    )


def _test_optimizer(
    model: models.Model,
    layerlr_ratio: float,
    bdecay: bool,
) -> None:
    # Make a list of valid settings.
    parameters: Dict[int, Tuple[str, float, float]] = {}
    decay = layerlr_ratio**(1 / (len(model.blocks) + 1))

    parameters.update(
        {k: (n, decay**(len(model.blocks) + 1), d)
         for k, n, d
         in _collect_parameters(model.stem, 'stem', bdecay)})
    parameters.update(
        {k: (n, 1.0, d)
         for k, n, d
         in _collect_parameters(model.head, 'head', bdecay)})
    parameters.update(
        {k: (n, 1.0, d)
         for k, n, d
         in _collect_parameters(model.classifier, 'classifier', bdecay)})

    for i, block in enumerate(model.blocks):
        parameters.update(
            {k: (n, decay**(len(model.blocks) - i), d)
             for k, n, d
             in _collect_parameters(block, f'blocks.{i}', bdecay)})

    # Make an optimizer.
    optimizer = create_optimizer(
        model=model,
        train_optim='sgd',
        train_lr=1.0,
        train_momentum=0.9,
        train_eps=1e-08,
        train_alpha=0.99,
        train_wdecay=0.1,
        train_bdecay=bdecay,
        train_layerlr_ratio=layerlr_ratio,
    )

    # Assert the number of parameter groups.
    if layerlr_ratio == 1.0:
        num_of_groups = 1
    else:
        num_of_groups = len(model.blocks) + 2

    if not bdecay:
        num_of_groups *= 2

    assert len(optimizer.param_groups) == num_of_groups

    # Assert optimizer settings.
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            key = id(param)
            name, learning_rate, weight_decay = parameters[key]
            assert learning_rate == param_group['lr'], name
            assert weight_decay == param_group['weight_decay'], name
            del parameters[key]

    # Assert the remainig parameters.
    assert len(parameters) == 0

    # Assert the learning rate for logging.
    assert optimizer.param_groups[0]['lr'] == 1.0


def _collect_parameters(
    module: nn.Module,
    module_name: str,
    bdecay: bool,
) -> List[Tuple[int, str, float]]:
    # Make a list of parameters.
    parameters: List[Tuple[int, str, float]] = []

    for name, param in module.named_parameters(recurse=False):
        if not param.requires_grad:
            continue

        key = id(param)
        decay = 0.1

        if not bdecay:
            if (isinstance(module, NORM_CLASSES)
                    or name == 'bias'
                    or (isinstance(module, WindowAttention)
                        and name == 'relative_position_bias_table')):
                decay = 0.0

        parameters.append((key, f'{module_name}.{name}', decay))

    # Search parameters recursively.
    for name, child in module.named_children():
        parameters.extend(
            _collect_parameters(child, f'{module_name}.{name}', bdecay))

    return parameters
