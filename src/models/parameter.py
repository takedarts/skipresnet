from typing import Any, Dict

from . import parameters
from .junctions import (AddJunction, DenseJunction, DynamicJunction,
                        MeanJunction, NoneJunction, SkipJunction,
                        StaticJunction, SumJunction)


def update_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def update_models(models, **kwargs):
    new_models = {}

    for name, params in models.items():
        new_models[name] = update_params(params, **kwargs)

    return new_models


def skip_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()
        new_params['semodule'] = False

        if (new_params['junction'] == AddJunction
                or new_params['junction'] == NoneJunction):
            new_params['junction'] = SkipJunction
        else:
            continue

        new_models[f'Skip-{name}'] = new_params

    return new_models


def dense_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()

        if new_params['junction'] == AddJunction:
            new_params['junction'] = DenseJunction
        else:
            continue

        new_models[f'Dense-{name}'] = new_params

    return new_models


def dynamic_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()
        new_params['semodule'] = False

        if new_params['junction'] == AddJunction:
            new_params['junction'] = DynamicJunction
        else:
            continue

        new_models[f'Dynamic-{name}'] = new_params

    return new_models


def static_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()

        if new_params['junction'] == AddJunction:
            new_params['junction'] = StaticJunction
        else:
            continue

        new_models[f'Static-{name}'] = new_params

    return new_models


def mean_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()
        new_params['semodule'] = False

        if new_params['junction'] == AddJunction:
            new_params['junction'] = MeanJunction
        else:
            continue

        new_models[f'Mean-{name}'] = new_params

    return new_models


def sum_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()
        new_params['semodule'] = False

        if new_params['junction'] == AddJunction:
            new_params['junction'] = SumJunction
        else:
            continue

        new_models[f'Sum-{name}'] = new_params

    return new_models


imagenet_base_models = {}
imagenet_base_models.update(parameters.resnet.imagenet_models)
imagenet_base_models.update(parameters.resnest.imagenet_models)
imagenet_base_models.update(parameters.regnet.imagenet_models)
imagenet_base_models.update(parameters.efficientnet.imagenet_models)
imagenet_base_models.update(parameters.densenet.imagenet_models)
imagenet_base_models.update(parameters.nfnet.imagenet_models)
imagenet_base_models.update(parameters.swin.imagenet_models)
imagenet_base_models.update(parameters.vit.imagenet_models)

cifar_base_models = {}
cifar_base_models.update(parameters.resnet.cifar_models)
cifar_base_models.update(parameters.resnest.cifar_models)
cifar_base_models.update(parameters.pyramidnet.cifar_models)

imagenet_models = {}
imagenet_models.update(imagenet_base_models)
imagenet_models.update(skip_models(imagenet_base_models))
imagenet_models.update(dense_models(imagenet_base_models))
imagenet_models.update(dynamic_models(imagenet_base_models))
imagenet_models.update(static_models(imagenet_base_models))
imagenet_models.update(mean_models(imagenet_base_models))
imagenet_models.update(sum_models(imagenet_base_models))

cifar_models = {}
cifar_models.update(cifar_base_models)
cifar_models.update(skip_models(cifar_base_models))
cifar_models.update(dense_models(cifar_base_models))
cifar_models.update(dynamic_models(cifar_base_models))
cifar_models.update(static_models(cifar_base_models))
cifar_models.update(mean_models(cifar_base_models))
cifar_models.update(sum_models(cifar_base_models))

PARAMETERS: Dict[str, Dict[str, Any]] = {
    'imagenet': update_models(imagenet_models, num_classes=1000),
    'cifar100': update_models(cifar_models, num_classes=100),
    'cifar10': update_models(cifar_models, num_classes=10),
    'dummy': update_models(cifar_models, num_classes=10),
}
