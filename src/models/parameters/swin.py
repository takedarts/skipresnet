import functools
from typing import Any, Dict, List, Tuple

import torch.nn as nn

from ..blocks import SwinBlock
from ..classifiers import LinearClassifier
from ..downsamples import NoneDownsample
from ..heads import SwinHead
from ..junctions import AddJunction
from ..loaders import load_swin_parameters
from ..modules import LayerNorm2d
from ..operations import SwinOperation
from ..stems import SwinPatchStem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_swin_layers(
    depths: List[int],
    channels: int,
    feature_size: int,
    window_size: int,
    attn_heads: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    layers = []
    attn_params = {'operation_type': 'attn', 'window_size': window_size}
    mlp_params = {'operation_type': 'mlp', 'window_size': 0,
                  'shift_size': 0, 'attn_heads': 0}

    for i, depth in enumerate(depths):
        attn_params['feature_size'] = feature_size
        attn_params['attn_heads'] = attn_heads
        mlp_params['feature_size'] = feature_size

        for d in range(depth):
            stride = 2 if i != 0 and d == 0 else 1

            if d % 2 == 1 and feature_size > window_size:
                attn_params['shift_size'] = window_size // 2
            else:
                attn_params['shift_size'] = 0

            layers.append((channels, stride, attn_params.copy()))
            layers.append((channels, 1, mlp_params.copy()))

        channels *= 2
        feature_size //= 2
        attn_heads *= 2

    return layers


imagenet_params = dict(
    stem=SwinPatchStem,
    block=SwinBlock,
    operation=SwinOperation,
    downsample=NoneDownsample,
    junction=AddJunction,
    head=SwinHead,
    classifier=LinearClassifier,
    normalization=functools.partial(nn.LayerNorm, eps=1e-5),
    activation=lambda *args, **kwargs: nn.GELU(),
    gate_normalization=functools.partial(LayerNorm2d, eps=1e-5),
    gate_activation=lambda *args, **kwargs: nn.GELU(),
)

imagenet_models = {
    'SwinTinyPatch4-224': clone_params(
        imagenet_params,
        layers=make_swin_layers(
            depths=[2, 2, 6, 2], channels=96,
            feature_size=56, window_size=7, attn_heads=3),
        stem_channels=96, head_channels=768,
        patch_size=4, mlp_ratio=4.0,
        timm_name='swin_tiny_patch4_window7_224',
        timm_loader=load_swin_parameters),

    'SwinSmallPatch4-224': clone_params(
        imagenet_params,
        layers=make_swin_layers(
            depths=[2, 2, 18, 2], channels=96,
            feature_size=56, window_size=7, attn_heads=3),
        stem_channels=96, head_channels=768,
        patch_size=4, mlp_ratio=4.0,
        timm_name='swin_small_patch4_window7_224',
        timm_loader=load_swin_parameters),

    'SwinBasePatch4-224': clone_params(
        imagenet_params,
        layers=make_swin_layers(
            depths=[2, 2, 18, 2], channels=128,
            feature_size=56, window_size=7, attn_heads=4),
        stem_channels=128, head_channels=1024,
        patch_size=4, mlp_ratio=4.0,
        timm_name='swin_base_patch4_window7_224',
        timm_loader=load_swin_parameters),
}
