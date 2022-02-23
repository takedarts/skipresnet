from ..blocks import BasicBlock, PreActivationBlock
from ..classifiers import LinearClassifier
from ..downsamples import AverageLinearDownsample, LinearDownsample
from ..heads import NoneHead, PreActivationHead
from ..junctions import AddJunction
from ..loaders import load_resnet_parameters, load_resnetd_parameters
from ..operations import (BasicOperation, BottleneckOperation,
                          PreActivationBasicOperation)
from ..stems import DeepLargeStem, LargeStem, PreActSmallStem, SmallStem


def clone_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def make_resnet_layers(depths, channels, groups, bottleneck):
    params = {'groups': groups, 'bottleneck': bottleneck}
    layers = []

    for i, depth in enumerate(depths):
        layers.append((round(channels * bottleneck), 1 if i == 0 else 2, params))
        layers.extend((round(channels * bottleneck), 1, params) for _ in range(depth - 1))
        channels *= 2

    return layers


imagenet_params = dict(
    stem=LargeStem,
    block=BasicBlock,
    operation=BasicOperation,
    downsample=LinearDownsample,
    junction=AddJunction,
    head=NoneHead,
    classifier=LinearClassifier,
)

imagenet_models = {
    'ResNet-18': clone_params(
        imagenet_params,
        layers=make_resnet_layers([2, 2, 2, 2], 64, 1, 1),
        stem_channels=64, head_channels=512,
        timm_name='resnet18',
        timm_loader=load_resnet_parameters),

    'ResNet-34': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 1),
        stem_channels=64, head_channels=512,
        timm_name='resnet34',
        timm_loader=load_resnet_parameters),

    'ResNet-50': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnet50',
        timm_loader=load_resnet_parameters),

    'ResNet-101': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 23, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnet101',
        timm_loader=load_resnet_parameters),

    'ResNet-152': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 8, 36, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnet152',
        timm_loader=load_resnet_parameters),

    'SE-ResNet-34': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 1),
        stem_channels=64, head_channels=512, semodule=True,
        semodule_reduction=16, semodule_divisor=8,
        timm_loader=load_resnet_parameters),

    'SE-ResNet-50': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048, semodule=True,
        semodule_reduction=16, semodule_divisor=8,
        operation=BottleneckOperation,
        timm_name='seresnet50',
        timm_loader=load_resnet_parameters),

    'ResNeXt-50-32x4d': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 6, 3], 4, 32, 64),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnext50_32x4d',
        timm_loader=load_resnet_parameters),

    'ResNeXt-101-32x4d': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 23, 3], 4, 32, 64),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnext101_32x4d',
        timm_loader=load_resnet_parameters),

    'ResNeXt-101-32x8d': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 23, 3], 8, 32, 32),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnext101_32x8d',
        timm_loader=load_resnet_parameters),

    'ResNetD-50': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=DeepLargeStem, downsample=AverageLinearDownsample,
        operation=BottleneckOperation,
        timm_name='resnet50d',
        timm_loader=load_resnetd_parameters),

    'ResNetD-101': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 4, 23, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=DeepLargeStem, downsample=AverageLinearDownsample,
        operation=BottleneckOperation,
        timm_name='resnet101d',
        timm_loader=load_resnetd_parameters),

    'ResNetD-152': clone_params(
        imagenet_params,
        layers=make_resnet_layers([3, 8, 36, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=DeepLargeStem, downsample=AverageLinearDownsample,
        operation=BottleneckOperation,
        timm_name='resnet152d',
        timm_loader=load_resnetd_parameters),
}


cifar_params = dict(
    stem=SmallStem,
    block=BasicBlock,
    operation=BasicOperation,
    downsample=LinearDownsample,
    junction=AddJunction,
    head=NoneHead,
    classifier=LinearClassifier,
)

cifar_models = {
    'ResNet-110': clone_params(
        cifar_params,
        layers=make_resnet_layers([18, 18, 18], 16, 1, 1),
        stem_channels=16, head_channels=64,
        gate_reduction=2),

    'ResNet-200': clone_params(
        cifar_params,
        layers=make_resnet_layers([33, 33, 33], 16, 1, 1),
        stem_channels=16, head_channels=64,
        gate_reduction=2),

    'SE-ResNet-110': clone_params(
        cifar_params,
        layers=make_resnet_layers([18, 18, 18], 16, 1, 1),
        stem_channels=16, head_channels=64, semodule=True,
        gate_reduction=2),

    'WideResNet-28-k10': clone_params(
        cifar_params,
        layers=make_resnet_layers([4, 4, 4], 160, 1, 1),
        stem_channels=16, head_channels=640,
        stem=PreActSmallStem,
        head=PreActivationHead,
        block=PreActivationBlock,
        operation=PreActivationBasicOperation),

    'WideResNet-40-k4': clone_params(
        cifar_params,
        layers=make_resnet_layers([6, 6, 6], 64, 1, 1),
        stem_channels=16, head_channels=256,
        stem=PreActSmallStem,
        head=PreActivationHead,
        block=PreActivationBlock,
        operation=PreActivationBasicOperation),

    'WideResNet-40-k10': clone_params(
        cifar_params,
        layers=make_resnet_layers([6, 6, 6], 160, 1, 1),
        stem_channels=16, head_channels=640,
        stem=PreActSmallStem,
        head=PreActivationHead,
        block=PreActivationBlock,
        operation=PreActivationBasicOperation),

    'ResNeXt-29-8x64d': clone_params(
        cifar_params,
        layers=make_resnet_layers([3, 3, 3], 64, 8, 4),
        stem_channels=64, head_channels=1024,
        operation=BottleneckOperation),

    'ResNeXt-47-32x4d': clone_params(
        cifar_params,
        layers=make_resnet_layers([5, 5, 5], 4, 32, 64),
        stem_channels=16, head_channels=1024,
        operation=BottleneckOperation),
}
