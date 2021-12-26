from ..stems import LargeStem
from ..blocks import DenseNetBlock
from ..heads import PreActivationHead
from ..operations import DenseNetOperation
from ..downsamples import NoneDownsample
from ..junctions import ConcatJunction
from ..classifiers import LinearClassifierWithoutDropout
from ..loaders import load_densenet_parameters


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_densenet_layers(depths, channels, growth, expansion):
    params = {'growth': growth, 'expansion': expansion}
    layers = []

    for i, depth, in enumerate(depths):
        if i != 0:
            channels //= 2
            layers.append((channels, 2, params))

        for _ in range(depth):
            channels += growth
            layers.append((channels, 1, params))

    return layers


imagenet_params = dict(
    stem=LargeStem,
    block=DenseNetBlock,
    operation=DenseNetOperation,
    downsample=NoneDownsample,
    junction=ConcatJunction,
    head=PreActivationHead,
    classifier=LinearClassifierWithoutDropout,
)

imagenet_models = {
    'DenseNet-121': clone_params(
        imagenet_params,
        layers=make_densenet_layers([6, 12, 24, 16], 64, 32, 4),
        stem_channels=64, head_channels=1024,
        timm_name='densenet121',
        timm_loader=load_densenet_parameters),

    'DenseNet-169': clone_params(
        imagenet_params,
        layers=make_densenet_layers([6, 12, 32, 32], 64, 32, 4),
        stem_channels=64, head_channels=1664,
        timm_name='densenet169',
        timm_loader=load_densenet_parameters),
}
