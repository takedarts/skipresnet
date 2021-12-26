import torch.nn as nn

from ..blocks import PreActivationBlock
from ..classifiers import LinearClassifier
from ..downsamples import NFDownsample
from ..heads import NFHead
from ..junctions import AddJunction
from ..loaders import load_nfnet_parameters
from ..operations import NFOperation
from ..stems import NFNetStem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_nfnet_layers(depths, channels, groups, bottleneck, alpha):
    widths = [channels, channels * 2, channels * 6, channels * 6]
    strides = [1] + [2] * (len(depths) - 1)
    expected_var = 1.0
    layers = []

    for d, w, s in zip(depths, widths, strides):
        for i in range(d):
            params = {
                'groups': w // channels * groups,
                'bottleneck': bottleneck,
                'beta': 1.0 / expected_var ** 0.5
            }

            layers.append((round(w * bottleneck), s if i == 0 else 1, params))

            if i == 0:
                expected_var = 1.0

            expected_var += alpha ** 2

    return layers


imagenet_params = dict(
    stem=NFNetStem,
    block=PreActivationBlock,
    operation=NFOperation,
    downsample=NFDownsample,
    junction=AddJunction,
    head=NFHead,
    classifier=LinearClassifier,
    normalization=lambda *args, **kwargs: nn.Identity(),
    activation=lambda *args, **kwargs: nn.GELU(),
)


imagenet_models = {
    # In the paper, semodule_gain = 2.0, alpha = 0.2.
    # But in this implementation, due to same procedure,
    # semodule_gain = 1.0, alpha = 0.4 in this implementation
    # because alpha is multiplied to features before se-module.
    'NFNet-F0': clone_params(
        imagenet_params,
        layers=make_nfnet_layers([1, 2, 6, 3], 128, 1, 2, alpha=0.2),
        stem_channels=128, head_channels=3072,
        semodule=True, semodule_reduction=2,
        alpha=0.4, gamma=1.7015043497085571,
        timm_name='dm_nfnet_f0',
        timm_loader=load_nfnet_parameters),

    'NFNet-F1': clone_params(
        imagenet_params,
        layers=make_nfnet_layers([2, 4, 12, 6], 128, 1, 2, alpha=0.2),
        stem_channels=128, head_channels=3072,
        semodule=True, semodule_reduction=2,
        alpha=0.4, gamma=1.7015043497085571,
        timm_name='dm_nfnet_f1',
        timm_loader=load_nfnet_parameters),
}
