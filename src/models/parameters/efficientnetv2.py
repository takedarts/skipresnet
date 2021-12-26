import functools
from typing import Any, Callable, Dict, List, Tuple

import torch.nn as nn

from ..blocks import MobileNetBlock
from ..classifiers import LinearClassifier
from ..downsamples import NoneDownsample
from ..heads import MobileNetV2Head
from ..junctions import AddJunction
from ..loaders import load_efficientnetv2_parameters
from ..operations import EfficientNetV2Operation
from ..stems import EfficientNetV2Stem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_efficientnetv2_layer(
    style: str,
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

    params = dict(
        style=style,
        kernel=kernel,
        expansion=expansion,
        semodule=semodule,
        semodule_reduction=semodule_reduction,
        semodule_divisor=semodule_divisor,
        activation=activation)

    return (new_channels, stride, params)


def make_efficientnetv2_layers(
    settings: List,
    divisor: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    layers = []

    for (
        style, kernel, channels, stride, expansion, activation, repeats,
        semodule, semodule_reduction, semodule_divisor,
    ) in settings:
        params = dict(
            style=style,
            kernel=kernel,
            channels=channels,
            expansion=expansion,
            divisor=divisor,
            activation=activation,
            semodule=semodule,
            semodule_reduction=semodule_reduction,
            semodule_divisor=semodule_divisor)

        layers.append(
            make_efficientnetv2_layer(stride=stride, **params))
        layers.extend(
            make_efficientnetv2_layer(stride=1, **params)
            for _ in range(repeats - 1))

    return layers


def make_efficientnetv2_s_layers(divisor: int):
    settings = [
        # style, kernel, channels, stride, expansion, activation, repeats
        # se-module, se-reduction,se-divisor
        ['conv', 3, 24, 1, 1, nn.SiLU, 2, False, 0, 0],
        ['edge', 3, 48, 2, 4, nn.SiLU, 4, False, 0, 0],
        ['edge', 3, 64, 2, 4, nn.SiLU, 4, False, 0, 0],
        ['inverted', 3, 128, 2, 4, nn.SiLU, 6, True, 16, 1],
        ['inverted', 3, 160, 1, 6, nn.SiLU, 9, True, 24, 1],
        ['inverted', 3, 256, 2, 6, nn.SiLU, 15, True, 24, 1],
    ]

    return make_efficientnetv2_layers(settings, divisor=divisor)


def make_efficientnetv2_m_layers(divisor: int):
    settings = [
        # style, kernel, channels, stride, expansion, activation, repeats
        # se-module, se-reduction,se-divisor
        ['conv', 3, 24, 1, 1, nn.SiLU, 3, False, 0, 0],
        ['edge', 3, 48, 2, 4, nn.SiLU, 5, False, 0, 0],
        ['edge', 3, 80, 2, 4, nn.SiLU, 5, False, 0, 0],
        ['inverted', 3, 160, 2, 4, nn.SiLU, 7, True, 16, 1],
        ['inverted', 3, 176, 1, 6, nn.SiLU, 14, True, 24, 1],
        ['inverted', 3, 304, 2, 6, nn.SiLU, 18, True, 24, 1],
        ['inverted', 3, 512, 1, 6, nn.SiLU, 5, True, 24, 1],
    ]

    return make_efficientnetv2_layers(settings, divisor=divisor)


def make_efficientnetv2_l_layers(divisor: int):
    settings = [
        # style, kernel, channels, stride, expansion, activation, repeats
        # se-module, se-reduction,se-divisor
        ['conv', 3, 32, 1, 1, nn.SiLU, 4, False, 0, 0],
        ['edge', 3, 64, 2, 4, nn.SiLU, 7, False, 0, 0],
        ['edge', 3, 96, 2, 4, nn.SiLU, 7, False, 0, 0],
        ['inverted', 3, 192, 2, 4, nn.SiLU, 10, True, 16, 8],
        ['inverted', 3, 224, 1, 6, nn.SiLU, 19, True, 24, 8],
        ['inverted', 3, 384, 2, 6, nn.SiLU, 25, True, 24, 8],
        ['inverted', 3, 640, 1, 6, nn.SiLU, 7, True, 24, 8],
    ]

    return make_efficientnetv2_layers(settings, divisor=divisor)


imagenet_params = dict(
    stem=EfficientNetV2Stem,
    block=MobileNetBlock,
    operation=EfficientNetV2Operation,
    downsample=NoneDownsample,
    junction=AddJunction,
    head=MobileNetV2Head,
    classifier=LinearClassifier,
    normalization=functools.partial(nn.BatchNorm2d, eps=1e-3),
    activation=nn.SiLU,
    semodule_activation=nn.SiLU,
)


imagenet_models = {
    'EfficientNetV2-S': clone_params(
        imagenet_params,
        layers=make_efficientnetv2_s_layers(divisor=8),
        stem_channels=24, head_channels=1280,
        timm_name='tf_efficientnetv2_s',
        timm_loader=load_efficientnetv2_parameters),

    'EfficientNetV2-M': clone_params(
        imagenet_params,
        layers=make_efficientnetv2_m_layers(divisor=8),
        stem_channels=24, head_channels=1280,
        timm_name='tf_efficientnetv2_m',
        timm_loader=load_efficientnetv2_parameters),

    'EfficientNetV2-L': clone_params(
        imagenet_params,
        layers=make_efficientnetv2_l_layers(divisor=8),
        stem_channels=32, head_channels=1280,
        timm_name='tf_efficientnetv2_l',
        timm_loader=load_efficientnetv2_parameters),
}
