from typing import Any, Dict, List, Tuple

import numpy as np

from ..blocks import BasicBlock
from ..classifiers import LinearClassifier
from ..downsamples import LinearDownsample
from ..heads import NoneHead
from ..junctions import AddJunction
from ..loaders import load_regnet_parameters
from ..operations import RegnetOperation
from ..stems import MobileNetStem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_regnet_layers(
    depth: int,
    channels: int,
    slope: float,
    mult: float,
    group_channels: int,
    bottleneck_ratio: float,
    divisor: int = 8,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    channels_reals = np.arange(depth) * slope + channels
    channels_exps = np.round(np.log(channels_reals / channels) / np.log(mult))
    channels_list = channels * (mult ** channels_exps)
    channels_list = (np.round(channels_list / divisor) * divisor).astype(np.int32)

    layers: List[Tuple[int, int, Dict[str, Any]]] = []
    in_channels = 0

    for out_channels in channels_list:
        groups = round(out_channels / group_channels)
        bottleneck = bottleneck_ratio * groups

        layers.append((
            group_channels * groups,
            2 if out_channels != in_channels else 1,
            {'groups': groups, 'bottleneck': bottleneck}
        ))

        in_channels = out_channels

    return layers


imagenet_params = dict(
    stem=MobileNetStem,
    head=NoneHead,
    classifier=LinearClassifier,
    block=BasicBlock,
    operation=RegnetOperation,
    downsample=LinearDownsample,
    junction=AddJunction,
)

imagenet_models = {
    'RegNetX-0.8': clone_params(
        imagenet_params,
        layers=make_regnet_layers(16, 56, 35.73, 2.28, 16, 1.0),
        stem_channels=32, head_channels=672,
        timm_name='regnetx_008',
        timm_loader=load_regnet_parameters),

    'RegNetX-1.6': clone_params(
        imagenet_params,
        layers=make_regnet_layers(18, 80, 34.01, 2.25, 24, 1.0),
        stem_channels=32, head_channels=912,
        timm_name='regnetx_016',
        timm_loader=load_regnet_parameters),


    'RegNetX-3.2': clone_params(
        imagenet_params,
        layers=make_regnet_layers(25, 88, 26.31, 2.25, 48, 1.0),
        stem_channels=32, head_channels=1008,
        timm_name='regnetx_032',
        timm_loader=load_regnet_parameters),


    'RegNetY-0.8': clone_params(
        imagenet_params,
        layers=make_regnet_layers(14, 56, 38.84, 2.4, 16, 1.0),
        stem_channels=32, head_channels=768,
        semodule=True, semodule_reduction=4,
        timm_name='regnety_008',
        timm_loader=load_regnet_parameters),

    'RegNetY-1.6': clone_params(
        imagenet_params,
        layers=make_regnet_layers(27, 48, 20.71, 2.65, 24, 1.0),
        stem_channels=32, head_channels=888,
        semodule=True, semodule_reduction=4,
        timm_name='regnety_016',
        timm_loader=load_regnet_parameters),

    'RegNetY-3.2': clone_params(
        imagenet_params,
        layers=make_regnet_layers(21, 80, 42.63, 2.66, 24, 1.0),
        stem_channels=32, head_channels=1512,
        semodule=True, semodule_reduction=4,
        timm_name='regnety_032',
        timm_loader=load_regnet_parameters),
}
