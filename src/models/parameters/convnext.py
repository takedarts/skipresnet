import functools

import torch.nn as nn

from ..blocks import SwinBlock
from ..classifiers import ConvNextClassifier
from ..downsamples import ConvNeXtDownsample
from ..heads import NoneHead
from ..junctions import AddJunction
from ..loaders import load_convnext_parameters
from ..modules import LayerNorm2d
from ..operations import ConvNeXtOperation
from ..stems import ConvNeXtStem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_convnext_layers(depths, channels):
    layers = []

    for i, depth in enumerate(depths):
        layers.append((channels, 1 if i == 0 else 2, {}))
        layers.extend((channels, 1, {}) for _ in range(depth - 1))
        channels *= 2

    return layers


imagenet_params = dict(
    stem=ConvNeXtStem,
    block=SwinBlock,
    operation=ConvNeXtOperation,
    downsample=ConvNeXtDownsample,
    junction=AddJunction,
    head=NoneHead,
    classifier=ConvNextClassifier,
    normalization=functools.partial(LayerNorm2d, eps=1e-6),
    activation=lambda *args, **kwargs: nn.GELU(),
    gate_normalization=functools.partial(LayerNorm2d, eps=1e-6),
    gate_activation=lambda *args, **kwargs: nn.GELU(),
)

imagenet_models = {
    'ConvNeXt-T': clone_params(
        imagenet_params,
        layers=make_convnext_layers([3, 3, 9, 3], 96),
        stem_channels=96, head_channels=768,
        patch_size=4, layer_scale_init_value=1e-6,
        timm_name='convnext_tiny',
        timm_loader=load_convnext_parameters),

    'ConvNeXt-S': clone_params(
        imagenet_params,
        layers=make_convnext_layers([3, 3, 27, 3], 96),
        stem_channels=96, head_channels=768,
        patch_size=4, layer_scale_init_value=1e-6,
        timm_name='convnext_small',
        timm_loader=load_convnext_parameters),

    'ConvNeXt-B': clone_params(
        imagenet_params,
        layers=make_convnext_layers([3, 3, 27, 3], 128),
        stem_channels=128, head_channels=1024,
        patch_size=4, layer_scale_init_value=1e-6,
        timm_name='convnext_base',
        timm_loader=load_convnext_parameters),

    'ConvNeXt-L': clone_params(
        imagenet_params,
        layers=make_convnext_layers([3, 3, 27, 3], 192),
        stem_channels=192, head_channels=1536,
        patch_size=4, layer_scale_init_value=1e-6,
        timm_name='convnext_large',
        timm_loader=load_convnext_parameters),
}

imagenet_models.update({
    'ConvNeXt-T-22k': clone_params(
        imagenet_models['ConvNeXt-T'],
        timm_name='convnext_tiny_in22k',
    ),

    'ConvNeXt-S-22k': clone_params(
        imagenet_models['ConvNeXt-S'],
        timm_name='convnext_small_in22k',
    ),

    'ConvNeXt-B-22k': clone_params(
        imagenet_models['ConvNeXt-B'],
        timm_name='convnext_base_in22k',
    ),

    'ConvNeXt-L-22k': clone_params(
        imagenet_models['ConvNeXt-L'],
        timm_name='convnext_large_in22k',
    ),

    'ConvNeXt-XL-22k': clone_params(
        imagenet_params,
        layers=make_convnext_layers([3, 3, 27, 3], 256),
        stem_channels=256, head_channels=2048,
        patch_size=4, layer_scale_init_value=1e-6,
        timm_name='convnext_xlarge_in22k',
        timm_loader=load_convnext_parameters),
})
