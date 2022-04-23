from typing import Any, Dict, List, Tuple
from ..loaders import load_tresnet_parameters
from ..stems import TResNetStem
from ..blocks import ResNetBlock
from ..downsamples import AverageLinearDownsample
from ..heads import NoneHead
from ..classifiers import LinearClassifier
from ..junctions import AddJunction
from ..operations import TResNetOperation
from ..modules import InplaceNorm
import torch.nn as nn
import functools


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_tresnet_layers(
    depths: List[int],
    channels: int,
    groups: int,
    bottleneck: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    layers = []

    for i, depth in enumerate(depths):
        stride = 1 if i == 0 else 2

        if i < 2:
            expand = 1
            params = dict(
                groups=groups,
                bottleneck=1,
                semodule_reduction=4,
                style='basic')
        else:
            expand = bottleneck
            params = dict(
                groups=groups,
                bottleneck=bottleneck,
                semodule_reduction=max(8 // bottleneck, 1),
                style='bottleneck')

        if i < 3:
            params['semodule'] = True
        else:
            params['semodule'] = False

        layers.append(
            (round(channels * expand), stride, params.copy()))
        layers.extend(
            (round(channels * expand), 1, params.copy())
            for _ in range(depth - 1))
        channels *= 2

    return layers


imagenet_params = dict(
    stem=TResNetStem,
    block=ResNetBlock,
    operation=TResNetOperation,
    downsample=AverageLinearDownsample,
    junction=AddJunction,
    head=NoneHead,
    classifier=LinearClassifier,
    normalization=InplaceNorm,
    activation=functools.partial(nn.LeakyReLU, negative_slope=1e-3),
)

imagenet_models = {
    'TResNet-M': clone_params(
        imagenet_params,
        layers=make_tresnet_layers([3, 4, 11, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        timm_name='tresnet_m',
        timm_loader=load_tresnet_parameters),

    'TResNet-L': clone_params(
        imagenet_params,
        layers=make_tresnet_layers([4, 5, 18, 3], 76, 1, 4),
        stem_channels=76, head_channels=2432,
        timm_name='tresnet_l',
        timm_loader=load_tresnet_parameters),

    'TResNet-XL': clone_params(
        imagenet_params,
        layers=make_tresnet_layers([4, 5, 24, 3], 83, 1, 4),
        stem_channels=83, head_channels=2656,
        timm_name='tresnet_xl',
        timm_loader=load_tresnet_parameters),
}
