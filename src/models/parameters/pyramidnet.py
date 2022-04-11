import itertools
from typing import Any, Dict, List, Tuple

from ..blocks import BaseBlock
from ..classifiers import LinearClassifier
from ..downsamples import AverageDownsample
from ..heads import PreActivationHead
from ..junctions import AddJunction
from ..operations import (SingleActivationBasicOperation,
                          SingleActivationBottleneckOperation)
from ..stems import PreActSmallStem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_pyramid_layers(
    depths: List[int],
    base: int,
    alpha: int,
    groups: int,
    bottleneck: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    params = {'groups': groups, 'bottleneck': bottleneck}
    depths = list(itertools.accumulate(depths))
    layers: List[Tuple[int, int, Dict[str, Any]]] = []

    for i in range(depths[-1]):
        channels = round(base + alpha * (i + 1) / depths[-1])
        stride = 2 if i in depths[:-1] else 1
        layers.append((round(channels * bottleneck), stride, params))

    return layers


cifar_params = dict(
    stem=PreActSmallStem,
    block=BaseBlock,
    operation=SingleActivationBasicOperation,
    downsample=AverageDownsample,
    junction=AddJunction,
    head=PreActivationHead,
    classifier=LinearClassifier,
)

cifar_models = {
    'PyramidNet-110-a48': clone_params(
        cifar_params,
        layers=make_pyramid_layers([18, 18, 18], 16, 48, 1, 1),
        stem_channels=16, head_channels=64),

    'PyramidNet-110-a270': clone_params(
        cifar_params,
        layers=make_pyramid_layers([18, 18, 18], 16, 270, 1, 1),
        stem_channels=16, head_channels=286),

    'PyramidNet-200-a240': clone_params(
        cifar_params,
        layers=make_pyramid_layers([22, 22, 22], 16, 240, 1, 4),
        stem_channels=16, head_channels=1024,
        operation=SingleActivationBottleneckOperation),

    'PyramidNet-272-a200': clone_params(
        cifar_params,
        layers=make_pyramid_layers([30, 30, 30], 16, 200, 1, 4),
        stem_channels=16, head_channels=864,
        operation=SingleActivationBottleneckOperation),
}
