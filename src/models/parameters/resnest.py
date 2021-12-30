from ..blocks import BasicBlock
from ..classifiers import LinearClassifier
from ..downsamples import AverageLinearDownsample
from ..heads import NoneHead
from ..junctions import AddJunction
from ..loaders import load_resnest_parameters
from ..operations import SplitAttentionOperation
from ..stems import DeepLargeStem, SmallStem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_resnest_layers(depths, channels, radix, groups, bottleneck):
    params = {'radix': radix, 'groups': groups, 'bottleneck': bottleneck}
    layers = []

    for i, depth in enumerate(depths):
        layers.append((round(channels * bottleneck), 1 if i == 0 else 2, params))
        layers.extend((round(channels * bottleneck), 1, params) for _ in range(depth - 1))
        channels *= 2

    return layers


imagenet_params = dict(
    stem=DeepLargeStem,
    block=BasicBlock,
    operation=SplitAttentionOperation,
    downsample=AverageLinearDownsample,
    junction=AddJunction,
    head=NoneHead,
    classifier=LinearClassifier,
)

imagenet_models = {
    'ResNeSt-50-2s1x64d': clone_params(
        imagenet_params,
        layers=make_resnest_layers([3, 4, 6, 3], 64, 2, 1, 256 / 64),
        stem_channels=64, head_channels=2048,
        stem=DeepLargeStem, downsample=AverageLinearDownsample,
        operation=SplitAttentionOperation, avg_first=False,
        timm_name='resnest50d',
        timm_loader=load_resnest_parameters),

    'ResNeSt-50-1s4x24d': clone_params(
        imagenet_params,
        layers=make_resnest_layers([3, 4, 6, 3], 96, 1, 4, 256 / 96),
        stem_channels=64, head_channels=2048,
        stem=DeepLargeStem, downsample=AverageLinearDownsample,
        operation=SplitAttentionOperation, avg_first=True,
        timm_name='resnest50d_1s4x24d',
        timm_loader=load_resnest_parameters),

    'ResNeSt-50-4s2x40d': clone_params(
        imagenet_params,
        layers=make_resnest_layers([3, 4, 6, 3], 80, 4, 2, 256 / 80),
        stem_channels=64, head_channels=2048,
        stem=DeepLargeStem, downsample=AverageLinearDownsample,
        operation=SplitAttentionOperation, avg_first=True,
        timm_name='resnest50d_4s2x40d',
        timm_loader=load_resnest_parameters),

    'ResNeSt-101-2s1x64d': clone_params(
        imagenet_params,
        layers=make_resnest_layers([3, 4, 23, 3], 64, 2, 1, 256 / 64),
        stem_channels=128, head_channels=2048,
        stem=DeepLargeStem, downsample=AverageLinearDownsample,
        operation=SplitAttentionOperation, avg_first=False,
        timm_name='resnest101e',
        timm_loader=load_resnest_parameters),

    'ResNeSt-200-2s1x64d': clone_params(
        imagenet_params,
        layers=make_resnest_layers([3, 24, 36, 3], 64, 2, 1, 256 / 64),
        stem_channels=128, head_channels=2048,
        stem=DeepLargeStem, downsample=AverageLinearDownsample,
        operation=SplitAttentionOperation, avg_first=False,
        timm_name='resnest200e',
        timm_loader=load_resnest_parameters),
}

cifar_params = dict(
    stem=SmallStem,
    block=BasicBlock,
    operation=SplitAttentionOperation,
    downsample=AverageLinearDownsample,
    junction=AddJunction,
    head=NoneHead,
    classifier=LinearClassifier,
)

cifar_models = {
    'ResNeSt-47-2s1x64d': clone_params(
        cifar_params,
        layers=make_resnest_layers([5, 5, 5], 64, 2, 1, 256 / 64),
        stem_channels=16, head_channels=1024,
        downsample=AverageLinearDownsample,
        operation=SplitAttentionOperation),

    'ResNeSt-47-2s1x128d': clone_params(
        cifar_params,
        layers=make_resnest_layers([5, 5, 5], 128, 2, 1, 256 / 128),
        stem_channels=16, head_channels=1024,
        downsample=AverageLinearDownsample,
        operation=SplitAttentionOperation),
}
