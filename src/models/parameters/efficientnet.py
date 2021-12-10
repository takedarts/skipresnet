import math
from typing import Any, Callable, Dict, List, Tuple

import torch.nn as nn

from ..blocks import MobileNetBlock
from ..classifiers import LinearClassifier
from ..downsamples import NoneDownsample
from ..heads import MobileNetV2Head, MobileNetV3Head
from ..junctions import AddJunction
from ..loaders import (load_efficientnet_parameters,
                       load_mobilenetv2_parameters,
                       load_mobilenetv3_parameters)
from ..operations import MobileNetOperation
from ..stems import MobileNetStem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_mobilenet_layer(
    kernel: int,
    channels: int,
    stride: int,
    expansion: int,
    divisor: int,
    activation: Callable[..., nn.Module],
    semodule: bool,
    semodule_reduction: int,
    semodule_divisor: int,
) -> Tuple[int, int, Dict[str, Any]]:
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)

    if new_channels < 0.9 * channels:
        new_channels += divisor

    params = {
        'kernel': kernel,
        'expansion': expansion,
        'semodule': semodule,
        'semodule_reduction': semodule_reduction,
        'semodule_divisor': semodule_divisor,
        'activation': activation}

    return (new_channels, stride, params)


def make_mobilenet_layers(
    settings: List,
    width: float,
    depth: float,
    divisor: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    layers = []

    for (
        kernel,
        channels,
        stride,
        expansion,
        semodule,
        semodule_reduction,
        semodule_divisor,
        activation,
        repeats,
    ) in settings:
        repeats = math.ceil(repeats * depth)
        params = {
            'kernel': kernel,
            'channels': int(channels * width),
            'expansion': expansion,
            'semodule': semodule,
            'semodule_reduction': semodule_reduction,
            'semodule_divisor': semodule_divisor,
            'activation': activation,
            'divisor': divisor,
        }
        layers.append(make_mobilenet_layer(stride=stride, **params))
        layers.extend(make_mobilenet_layer(stride=1, **params) for _ in range(repeats - 1))

    return layers


def make_mobilenetv2_layers(width):
    settings = [
        # kernel, channels, stride, expansion,
        # se-module, se-reduction,se-divisor,
        # activation, repeats
        [3, 16, 1, 1, False, 0, 0, nn.ReLU6, 1],
        [3, 24, 2, 6, False, 0, 0, nn.ReLU6, 2],
        [3, 32, 2, 6, False, 0, 0, nn.ReLU6, 3],
        [3, 64, 2, 6, False, 0, 0, nn.ReLU6, 4],
        [3, 96, 1, 6, False, 0, 0, nn.ReLU6, 3],
        [3, 160, 2, 6, False, 0, 0, nn.ReLU6, 3],
        [3, 320, 1, 6, False, 0, 0, nn.ReLU6, 1]]

    return make_mobilenet_layers(settings, width, 1.0, 8)


def make_mobilenetv3_large_layers(width):
    settings = [
        # kernel, channels, stride, expansion,
        # se-module, se-reduction,se-divisor,
        # activation, repeats
        [3, 16, 1, 1, False, 0, 0, nn.ReLU, 1],
        [3, 24, 2, 4, False, 0, 0, nn.ReLU, 1],
        [3, 24, 1, 3, False, 0, 0, nn.ReLU, 1],
        [5, 40, 2, 3, True, 4, 8, nn.ReLU, 3],
        [3, 80, 2, 6, False, 0, 0, nn.Hardswish, 1],
        [3, 80, 1, 2.5, False, 0, 0, nn.Hardswish, 1],
        [3, 80, 1, 2.3, False, 0, 0, nn.Hardswish, 2],
        [3, 112, 1, 6, True, 4, 8, nn.Hardswish, 2],
        [5, 160, 2, 6, True, 4, 8, nn.Hardswish, 3]]

    return make_mobilenet_layers(settings, width, 1.0, 8)


def make_efficientnet_layers(width, depth):
    settings = [
        # kernel, channels, stride, expansion,
        # se-module, se-reduction,se-divisor,
        # activation, repeats
        [3, 16, 1, 1, True, 4, 1, nn.SiLU, 1],
        [3, 24, 2, 6, True, 24, 1, nn.SiLU, 2],
        [5, 40, 2, 6, True, 24, 1, nn.SiLU, 2],
        [3, 80, 2, 6, True, 24, 1, nn.SiLU, 3],
        [5, 112, 1, 6, True, 24, 1, nn.SiLU, 3],
        [5, 192, 2, 6, True, 24, 1, nn.SiLU, 4],
        [3, 320, 1, 6, True, 24, 1, nn.SiLU, 1]]

    return make_mobilenet_layers(settings, width, depth, 8)


imagenet_params = dict(
    stem=MobileNetStem,
    block=MobileNetBlock,
    operation=MobileNetOperation,
    downsample=NoneDownsample,
    junction=AddJunction,
    classifier=LinearClassifier,
)


imagenet_models = {
    'MobileNetV2-1.0': clone_params(
        imagenet_params,
        layers=make_mobilenetv2_layers(1.0),
        stem_channels=32, head_channels=1280,
        head=MobileNetV2Head,
        activation=nn.ReLU6,
        semodule_activation=nn.ReLU,
        timm_name='mobilenetv2_100',
        timm_loader=load_mobilenetv2_parameters),

    'MobileNetV2-1.4': clone_params(
        imagenet_params,
        layers=make_mobilenetv2_layers(1.4),
        stem_channels=48, head_channels=1792,
        head=MobileNetV2Head,
        activation=nn.ReLU6,
        semodule_activation=nn.ReLU,
        timm_name='mobilenetv2_140',
        timm_loader=load_mobilenetv2_parameters),

    'MobileNetV3-large': clone_params(
        imagenet_params,
        layers=make_mobilenetv3_large_layers(1.0),
        stem_channels=16, head_channels=1280,
        head=MobileNetV3Head,
        activation=nn.Hardswish,
        semodule_activation=nn.ReLU,
        semodule_sigmoid=lambda: nn.Hardsigmoid(inplace=True),
        timm_name='mobilenetv3_large_100',
        timm_loader=load_mobilenetv3_parameters),

    'EfficientNet-B0': clone_params(
        imagenet_params,
        layers=make_efficientnet_layers(1.0, 1.0),
        stem_channels=32, head_channels=1280,
        head=MobileNetV2Head,
        activation=nn.SiLU,
        semodule_activation=nn.SiLU,
        timm_name='efficientnet_b0',
        timm_loader=load_efficientnet_parameters),

    'EfficientNet-B1': clone_params(
        imagenet_params,
        layers=make_efficientnet_layers(1.0, 1.1),
        stem_channels=32, head_channels=1280,
        head=MobileNetV2Head,
        activation=nn.SiLU,
        semodule_activation=nn.SiLU,
        timm_name='efficientnet_b1',
        timm_loader=load_efficientnet_parameters),

    'EfficientNet-B2': clone_params(
        imagenet_params,
        layers=make_efficientnet_layers(1.1, 1.2),
        stem_channels=32, head_channels=1408,
        head=MobileNetV2Head,
        activation=nn.SiLU,
        semodule_activation=nn.SiLU,
        timm_name='efficientnet_b2',
        timm_loader=load_efficientnet_parameters),

    'EfficientNet-B3': clone_params(
        imagenet_params,
        layers=make_efficientnet_layers(1.2, 1.4),
        stem_channels=40, head_channels=1536,
        head=MobileNetV2Head,
        activation=nn.SiLU,
        semodule_activation=nn.SiLU,
        timm_name='efficientnet_b3',
        timm_loader=load_efficientnet_parameters),
}
