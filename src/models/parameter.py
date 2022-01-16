from typing import Any, Callable, Dict

import torch.nn as nn

from . import parameters
from .downsamples import LinearDownsample, NoneDownsample
from .junctions import (AddJunction, DenseJunction, DynamicJunction,
                        MeanJunction, SkipJunction, StaticJunction,
                        SumJunction)


def clone_params(params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def _new_models(
    prefix: str,
    models: Dict[str, Any],
    junction: Callable[..., nn.Module],
) -> Dict[str, Any]:
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()

        if new_params['junction'] == AddJunction:
            new_params['junction'] = junction

            if new_params['downsample'] == NoneDownsample:
                new_params['downsample'] = LinearDownsample
        else:
            continue

        new_models[f'{prefix}-{name}'] = new_params

    return new_models


def skip_models(models: Dict[str, Any]) -> Dict[str, Any]:
    return _new_models('Skip', models, SkipJunction)


def dense_models(models: Dict[str, Any]) -> Dict[str, Any]:
    return _new_models('Dense', models, DenseJunction)


def dynamic_models(models: Dict[str, Any]) -> Dict[str, Any]:
    return _new_models('Dynamic', models, DynamicJunction)


def static_models(models: Dict[str, Any]) -> Dict[str, Any]:
    return _new_models('Static', models, StaticJunction)


def mean_models(models: Dict[str, Any]) -> Dict[str, Any]:
    return _new_models('Mean', models, MeanJunction)


def sum_models(models: Dict[str, Any]) -> Dict[str, Any]:
    return _new_models('Sum', models, SumJunction)


imagenet_base_models: Dict[str, Any] = {}
imagenet_base_models.update(parameters.resnet.imagenet_models)
imagenet_base_models.update(parameters.resnest.imagenet_models)
imagenet_base_models.update(parameters.regnet.imagenet_models)
imagenet_base_models.update(parameters.efficientnet.imagenet_models)
imagenet_base_models.update(parameters.mobilenet.imagenet_models)
imagenet_base_models.update(parameters.densenet.imagenet_models)
imagenet_base_models.update(parameters.nfnet.imagenet_models)
imagenet_base_models.update(parameters.swin.imagenet_models)
imagenet_base_models.update(parameters.vit.imagenet_models)
imagenet_base_models.update(parameters.convnext.imagenet_models)

cifar_base_models: Dict[str, Any] = {}
cifar_base_models.update(parameters.resnet.cifar_models)
cifar_base_models.update(parameters.resnest.cifar_models)
cifar_base_models.update(parameters.pyramidnet.cifar_models)

imagenet_models: Dict[str, Any] = {}
imagenet_models.update(imagenet_base_models)
imagenet_models.update(skip_models(imagenet_base_models))
imagenet_models.update(dense_models(imagenet_base_models))
imagenet_models.update(dynamic_models(imagenet_base_models))
imagenet_models.update(static_models(imagenet_base_models))
imagenet_models.update(mean_models(imagenet_base_models))
imagenet_models.update(sum_models(imagenet_base_models))

cifar_models: Dict[str, Any] = {}
cifar_models.update(cifar_base_models)
cifar_models.update(skip_models(cifar_base_models))
cifar_models.update(dense_models(cifar_base_models))
cifar_models.update(dynamic_models(cifar_base_models))
cifar_models.update(static_models(cifar_base_models))
cifar_models.update(mean_models(cifar_base_models))
cifar_models.update(sum_models(cifar_base_models))

PARAMETERS: Dict[str, Any] = {}
PARAMETERS.update(imagenet_models)
PARAMETERS.update(cifar_models)
