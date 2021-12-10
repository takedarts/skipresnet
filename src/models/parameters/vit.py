import functools
from typing import Any, Dict, List, Tuple

import torch.nn as nn

from ..blocks import ViTBlock
from ..classifiers import LinearClassifier
from ..downsamples import LinearDownsample
from ..heads import ViTHead
from ..junctions import AddJunction
from ..loaders import load_vit_parameters
from ..operations import ViTOperation
from ..stems import ViTPatchStem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_vit_layers(
    depths: List[int],
    channels: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    layers = []

    for i, depth in enumerate(depths):
        for d in range(depth):
            stride = 2 if i != 0 and d == 0 else 1
            layers.append((channels, stride, {'operation_type': 'attn'}))
            layers.append((channels, 1, {'operation_type': 'mlp'}))
        channels *= 2

    return layers


imagenet_params = dict(
    stem=ViTPatchStem,
    block=ViTBlock,
    operation=ViTOperation,
    downsample=LinearDownsample,
    junction=AddJunction,
    head=ViTHead,
    classifier=LinearClassifier,
    normalization=functools.partial(nn.LayerNorm, eps=1e-6),
    activation=lambda *args, **kwargs: nn.GELU(),
    gate_normalization=lambda channels: nn.LayerNorm([channels, 1, 1], eps=1e-6),
)

imagenet_models = {
    'ViTSmallPatch16-224': clone_params(
        imagenet_params,
        layers=make_vit_layers([12], 384),
        stem_channels=384, head_channels=384,
        patch_size=16, num_patches=196, attn_heads=6, mlp_ratio=4.0,
        timm_name='vit_small_patch16_224',
        timm_loader=load_vit_parameters),

    'ViTBasePatch16-224': clone_params(
        imagenet_params,
        layers=make_vit_layers([12], 768),
        stem_channels=768, head_channels=768,
        patch_size=16, num_patches=196, attn_heads=12, mlp_ratio=4.0,
        timm_name='vit_base_patch16_224',
        timm_loader=load_vit_parameters),
}
