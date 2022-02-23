import functools
from typing import Any, Callable, Dict, List, Tuple
import math
import torch.nn as nn

from ..blocks import MobileNetBlock
from ..classifiers import LinearClassifier
from ..downsamples import NoneDownsample
from ..heads import MobileNetV2Head
from ..junctions import AddJunction
from ..loaders import load_efficientnet_parameters
from ..operations import EfficientNetOperation
from ..stems import EfficientNetStem


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
    settings: List[Tuple[str, int, int, int, int, Any, int, bool, int, int]],
    divisor: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    layers = []

    for (style, kernel, channels, stride, expansion, activation, repeats,
         semodule, semodule_reduction, semodule_divisor) in settings:
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


def make_efficientnet_layers(
    width: float,
    depth: float,
    divisor: int
) -> List[Tuple[int, int, Dict[str, Any]]]:
    bases: List[Tuple[str, int, int, int, int, Any, int, bool, int, int]] = [
        # style, kernel, channels, stride, expansion, activation, repeats
        # semodule, semodule_reduction, semodule_divisor
        ('depthwise', 3, 16, 1, 1, nn.SiLU, 1, True, 4, 1),
        ('inverted', 3, 24, 2, 6, nn.SiLU, 2, True, 24, 1),
        ('inverted', 5, 40, 2, 6, nn.SiLU, 2, True, 24, 1),
        ('inverted', 3, 80, 2, 6, nn.SiLU, 3, True, 24, 1),
        ('inverted', 5, 112, 1, 6, nn.SiLU, 3, True, 24, 1),
        ('inverted', 5, 192, 2, 6, nn.SiLU, 4, True, 24, 1),
        ('inverted', 3, 320, 1, 6, nn.SiLU, 1, True, 24, 1),
    ]
    settings = []

    for (style, kernel, channels, stride, expansion, activation, repeats,
         semodule, semodule_reduction, semodule_divisor) in bases:
        repeats = math.ceil(repeats * depth)
        channels = int(channels * width)
        settings.append((
            style, kernel, channels, stride, expansion, activation, repeats,
            semodule, semodule_reduction, semodule_divisor))

    return make_efficientnetv2_layers(settings, divisor=divisor)


def make_efficientnetv2_s_layers(
    divisor: int
) -> List[Tuple[int, int, Dict[str, Any]]]:
    settings: List[Tuple[str, int, int, int, int, Any, int, bool, int, int]] = [
        # style, kernel, channels, stride, expansion, activation, repeats
        # semodule, semodule_reduction, semodule_divisor
        ('conv', 3, 24, 1, 1, nn.SiLU, 2, False, 0, 0),
        ('edge', 3, 48, 2, 4, nn.SiLU, 4, False, 0, 0),
        ('edge', 3, 64, 2, 4, nn.SiLU, 4, False, 0, 0),
        ('inverted', 3, 128, 2, 4, nn.SiLU, 6, True, 16, 1),
        ('inverted', 3, 160, 1, 6, nn.SiLU, 9, True, 24, 1),
        ('inverted', 3, 256, 2, 6, nn.SiLU, 15, True, 24, 1),
    ]

    return make_efficientnetv2_layers(settings, divisor=divisor)


def make_efficientnetv2_m_layers(
    divisor: int
) -> List[Tuple[int, int, Dict[str, Any]]]:
    settings: List[Tuple[str, int, int, int, int, Any, int, bool, int, int]] = [
        # style, kernel, channels, stride, expansion, activation, repeats
        # semodule, semodule_reduction, semodule_divisor
        ('conv', 3, 24, 1, 1, nn.SiLU, 3, False, 0, 0),
        ('edge', 3, 48, 2, 4, nn.SiLU, 5, False, 0, 0),
        ('edge', 3, 80, 2, 4, nn.SiLU, 5, False, 0, 0),
        ('inverted', 3, 160, 2, 4, nn.SiLU, 7, True, 16, 1),
        ('inverted', 3, 176, 1, 6, nn.SiLU, 14, True, 24, 1),
        ('inverted', 3, 304, 2, 6, nn.SiLU, 18, True, 24, 1),
        ('inverted', 3, 512, 1, 6, nn.SiLU, 5, True, 24, 1),
    ]

    return make_efficientnetv2_layers(settings, divisor=divisor)


def make_efficientnetv2_l_layers(
    divisor: int
) -> List[Tuple[int, int, Dict[str, Any]]]:
    settings: List[Tuple[str, int, int, int, int, Any, int, bool, int, int]] = [
        # style, kernel, channels, stride, expansion, activation, repeats
        # se-module, se-reduction,se-divisor
        ('conv', 3, 32, 1, 1, nn.SiLU, 4, False, 0, 0),
        ('edge', 3, 64, 2, 4, nn.SiLU, 7, False, 0, 0),
        ('edge', 3, 96, 2, 4, nn.SiLU, 7, False, 0, 0),
        ('inverted', 3, 192, 2, 4, nn.SiLU, 10, True, 16, 8),
        ('inverted', 3, 224, 1, 6, nn.SiLU, 19, True, 24, 8),
        ('inverted', 3, 384, 2, 6, nn.SiLU, 25, True, 24, 8),
        ('inverted', 3, 640, 1, 6, nn.SiLU, 7, True, 24, 8),
    ]

    return make_efficientnetv2_layers(settings, divisor=divisor)


imagenet_params = dict(
    stem=EfficientNetStem,
    block=MobileNetBlock,
    operation=EfficientNetOperation,
    downsample=NoneDownsample,
    junction=AddJunction,
    head=MobileNetV2Head,
    classifier=LinearClassifier,
    normalization=functools.partial(nn.BatchNorm2d, eps=1e-3),
    activation=nn.SiLU,
    semodule_activation=nn.SiLU,
    gate_normalization=functools.partial(nn.BatchNorm2d, eps=1e-3),
    gate_activation=nn.SiLU,
)


imagenet_models = {
    'EfficientNet-B0': clone_params(
        imagenet_params,
        layers=make_efficientnet_layers(1.0, 1.0, divisor=8),
        stem_channels=32, head_channels=1280,
        timm_name='tf_efficientnet_b0',
        timm_loader=load_efficientnet_parameters),

    'EfficientNet-B1': clone_params(
        imagenet_params,
        layers=make_efficientnet_layers(1.0, 1.1, divisor=8),
        stem_channels=32, head_channels=1280,
        timm_name='tf_efficientnet_b1',
        timm_loader=load_efficientnet_parameters),

    'EfficientNet-B2': clone_params(
        imagenet_params,
        layers=make_efficientnet_layers(1.1, 1.2, divisor=8),
        stem_channels=32, head_channels=1408,
        timm_name='tf_efficientnet_b2',
        timm_loader=load_efficientnet_parameters),

    'EfficientNet-B3': clone_params(
        imagenet_params,
        layers=make_efficientnet_layers(1.2, 1.4, divisor=8),
        stem_channels=40, head_channels=1536,
        timm_name='tf_efficientnet_b3',
        timm_loader=load_efficientnet_parameters),

    'EfficientNetV2-S': clone_params(
        imagenet_params,
        layers=make_efficientnetv2_s_layers(divisor=8),
        stem_channels=24, head_channels=1280,
        timm_name='tf_efficientnetv2_s_in21ft1k',
        timm_loader=load_efficientnet_parameters),

    'EfficientNetV2-M': clone_params(
        imagenet_params,
        layers=make_efficientnetv2_m_layers(divisor=8),
        stem_channels=24, head_channels=1280,
        timm_name='tf_efficientnetv2_m_in21ft1k',
        timm_loader=load_efficientnet_parameters),

    'EfficientNetV2-L': clone_params(
        imagenet_params,
        layers=make_efficientnetv2_l_layers(divisor=8),
        stem_channels=32, head_channels=1280,
        timm_name='tf_efficientnetv2_l_in21ft1k',
        timm_loader=load_efficientnet_parameters),
}
