import functools
import itertools
import math
from typing import Any, Callable, Dict, List, Tuple

import torch.nn as nn

from .block import (BasicBlock, DenseNetBlock, MobileNetBlock, PreActBlock,
                    ViTBlock)
from .classifier import BasicClassifier
from .downsample import (AverageDownsample, BasicDownsample, NFDownsample,
                         NoneDownsample, TweakedDownsample)
from .head import (BasicHead, MobileNetV2Head, MobileNetV3Head, NFHead,
                   PreActHead, ViTHead)
from .junction import (BasicJunction, ConcatJunction, DenseJunction,
                       DynamicJunction, MeanJunction, NoneJunction,
                       SkipJunction, StaticJunction, SumJunction)
from .modules import HSigmoid, HSwish
from .operation import (BasicOperation, BottleneckOperation, DenseNetOperation,
                        MobileNetOperation, NFOperation, PreActBasicOperation,
                        SelectedKernelOperation, SingleActBasicOperation,
                        SingleActBottleneckOperation, SplitAttentionOperation,
                        TweakedBottleneckOperation,
                        TweakedSlectedKernelOperation, ViTOperation)
from .stem import (BasicLargeStem, BasicSmallStem, MobileNetStem, NFNetStem,
                   PreActSmallStem, TweakedLargeStem, ViTPatchStem)


def make_resnet_layers(depths, channels, groups, bottleneck):
    params = {'groups': groups, 'bottleneck': bottleneck}
    layers = []

    for i, depth in enumerate(depths):
        layers.append((round(channels * bottleneck), 1 if i == 0 else 2, params))
        layers.extend((round(channels * bottleneck), 1, params) for _ in range(depth - 1))
        channels *= 2

    return layers


def make_resnest_layers(depths, channels, radix, groups, bottleneck):
    params = {'radix': radix, 'groups': groups, 'bottleneck': bottleneck}
    layers = []

    for i, depth in enumerate(depths):
        layers.append((round(channels * bottleneck), 1 if i == 0 else 2, params))
        layers.extend((round(channels * bottleneck), 1, params) for _ in range(depth - 1))
        channels *= 2

    return layers


def make_skresnet_layers(depths, channels, radix, groups, bottleneck):
    params = {'radix': radix, 'groups': groups, 'bottleneck': bottleneck}
    layers = []

    for i, depth in enumerate(depths):
        layers.append((round(channels * bottleneck), 1 if i == 0 else 2, params))
        layers.extend((round(channels * bottleneck), 1, params) for _ in range(depth - 1))
        channels *= 2

    return layers


def make_pyramid_layers(depths, base, alpha, groups, bottleneck):
    params = {'groups': groups, 'bottleneck': bottleneck}
    depths = list(itertools.accumulate(depths))
    layers = []

    for i in range(depths[-1]):
        channels = round(base + alpha * (i + 1) / depths[-1])
        stride = 2 if i in depths[:-1] else 1
        layers.append((round(channels * bottleneck), stride, params))

    return layers


def make_mobilenet_layer(
    kernel: int,
    channels: int,
    stride: int,
    expansion: int,
    seoperation: bool,
    sereduction: float,
    activation: Callable[..., nn.Module],
    divisor: int = 8,
) -> Tuple[int, int, Dict[str, Any]]:
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)

    if new_channels < 0.9 * channels:
        new_channels += divisor

    params = {
        'kernel': kernel,
        'expansion': expansion,
        'seoperation': seoperation,
        'seoperation_reduction': sereduction,
        'activation': activation}

    return (new_channels, stride, params)


def make_mobilenet_layers(
    settings: List,
    width: int,
    depth: int
) -> List[Tuple[int, int, Dict[str, Any]]]:
    layers = []

    for (kernel, channels, stride, expansion,
         seoperation, sereduction, activation, repeats) in settings:
        repeats = math.ceil(repeats * depth)
        params = {
            'kernel': kernel,
            'channels': channels * width,
            'expansion': expansion,
            'seoperation': seoperation,
            'sereduction': sereduction,
            'activation': activation}
        layers.append(make_mobilenet_layer(stride=stride, **params))
        layers.extend(make_mobilenet_layer(stride=1, **params) for _ in range(repeats - 1))

    return layers


def make_mobilenetv2_layers(width):
    settings = [
        # kernel, channels, stride, expansion,
        # se-module, se-reduction, activation, repeats
        [3, 16, 1, 1, False, 0, nn.ReLU6, 1],
        [3, 24, 2, 6, False, 0, nn.ReLU6, 2],
        [3, 32, 2, 6, False, 0, nn.ReLU6, 3],
        [3, 64, 2, 6, False, 0, nn.ReLU6, 4],
        [3, 96, 1, 6, False, 0, nn.ReLU6, 3],
        [3, 160, 2, 6, False, 0, nn.ReLU6, 3],
        [3, 320, 1, 6, False, 0, nn.ReLU6, 1]]

    return make_mobilenet_layers(settings, width, 1.0)


def make_mobilenetv3_large_layers(width):
    settings = [
        # kernel, channels, stride, expansion,
        # se-module, se-reduction, activation, repeats
        [3, 16, 1, 1, False, 0, nn.ReLU, 1],
        [3, 24, 2, 4, False, 0, nn.ReLU, 1],
        [3, 24, 1, 3, False, 0, nn.ReLU, 1],
        [5, 40, 2, 3, True, 0, nn.ReLU, 3],
        [3, 80, 2, 6, False, 0, HSwish, 1],
        [3, 80, 1, 2.5, False, 0, HSwish, 1],
        [3, 80, 1, 2.3, False, 0, HSwish, 2],
        [3, 112, 1, 6, True, 4, HSwish, 2],
        [5, 160, 2, 6, True, 4, HSwish, 3]]

    return make_mobilenet_layers(settings, width, 1.0)


def make_efficientnet_layers(width, depth):
    settings = [
        # kernel, channels, stride, expansion,
        # se-module, se-reduction, activation, repeats
        [3, 16, 1, 1, True, 4, nn.SiLU, 1],
        [3, 24, 2, 6, True, 24, nn.SiLU, 2],
        [5, 40, 2, 6, True, 24, nn.SiLU, 2],
        [3, 80, 2, 6, True, 24, nn.SiLU, 3],
        [5, 112, 1, 6, True, 24, nn.SiLU, 3],
        [5, 192, 2, 6, True, 24, nn.SiLU, 4],
        [3, 320, 1, 6, True, 24, nn.SiLU, 1]]

    return make_mobilenet_layers(settings, width, depth)


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


def make_vit_layers(
    depths: List[int],
    channels: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    layers = []

    for i, depth in enumerate(depths):
        for d in range(depth):
            stride = 2 if i != 0 and d == 0 else 1
            layers.append((channels, stride, {'operation_type': 'attn'}))
            layers.append((channels, stride, {'operation_type': 'mlp'}))
        channels *= 2

    return layers


def update_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def update_models(models, **kwargs):
    new_models = {}

    for name, params in models.items():
        new_models[name] = update_params(params, **kwargs)

    return new_models


def dense_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()

        if new_params['junction'] == BasicJunction:
            new_params['junction'] = DenseJunction
        else:
            continue

        if new_params['downsample'] == NoneDownsample:
            new_params['downsample'] = AverageDownsample

        new_models[f'Dense-{name}'] = new_params

    return new_models


def skip_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()
        new_params['semodule'] = False

        if (new_params['junction'] == BasicJunction
                or new_params['junction'] == NoneJunction):
            new_params['junction'] = SkipJunction
        else:
            continue

        if new_params['downsample'] == NoneDownsample:
            new_params['downsample'] = AverageDownsample

        new_models[f'Skip-{name}'] = new_params

    return new_models


def dynamic_dense_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()
        new_params['semodule'] = False

        if (new_params['junction'] == BasicJunction
                or new_params['junction'] == NoneJunction):
            new_params['junction'] = DynamicJunction
        else:
            continue

        if new_params['downsample'] == NoneDownsample:
            new_params['downsample'] = AverageDownsample

        new_models[f'Dynamic-{name}'] = new_params

    return new_models


def static_dense_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()

        if new_params['junction'] == BasicJunction:
            new_params['junction'] = StaticJunction
        else:
            continue

        if new_params['downsample'] == NoneDownsample:
            new_params['downsample'] = AverageDownsample

        new_models[f'Static-{name}'] = new_params

    return new_models


def mean_dense_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()
        new_params['semodule'] = False

        if new_params['junction'] == BasicJunction:
            new_params['junction'] = MeanJunction
        else:
            continue

        if new_params['downsample'] == NoneDownsample:
            new_params['downsample'] = AverageDownsample

        new_models[f'Mean-{name}'] = new_params

    return new_models


def sum_dense_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()
        new_params['semodule'] = False

        if new_params['junction'] == BasicJunction:
            new_params['junction'] = SumJunction
        else:
            continue

        if new_params['downsample'] == NoneDownsample:
            new_params['downsample'] = AverageDownsample

        new_models[f'Sum-{name}'] = new_params

    return new_models


large_basic_params = {
    'stem': BasicLargeStem,
    'head': BasicHead,
    'classifier': BasicClassifier,
    'block': BasicBlock,
    'operation': BasicOperation,
    'downsample': BasicDownsample,
    'junction': BasicJunction}

large_models = {
    'ResNet-18': update_params(
        large_basic_params,
        layers=make_resnet_layers([2, 2, 2, 2], 64, 1, 1),
        stem_channels=64, head_channels=512,
        timm_name='resnet18', timm_loader='resnet'),

    'ResNet-34': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 1),
        stem_channels=64, head_channels=512,
        timm_name='resnet34', timm_loader='resnet'),

    'ResNet-50': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnet50', timm_loader='resnet'),

    'ResNet-101': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 23, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnet101', timm_loader='resnet'),

    'ResNet-152': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 8, 36, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnet152', timm_loader='resnet'),

    'SE-ResNet-34': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 1),
        stem_channels=64, head_channels=512, semodule=True,
        timm_name='seresnet50', timm_loader='resnet'),

    'SE-ResNet-50': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation, semodule=True,
        timm_name='seresnet50', timm_loader='resnet'),

    'SK-ResNet-50': update_params(
        large_basic_params,
        layers=make_skresnet_layers([3, 4, 6, 3], 64, 2, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=SelectedKernelOperation),

    'ResNetD-50': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=TweakedLargeStem, downsample=TweakedDownsample,
        operation=TweakedBottleneckOperation),

    'SK-ResNetD-50': update_params(
        large_basic_params,
        layers=make_skresnet_layers([3, 4, 6, 3], 64, 2, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=TweakedLargeStem, downsample=TweakedDownsample,
        operation=TweakedSlectedKernelOperation),

    'ResNeXt-50-32x4d': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 4, 32, 64),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnext50_32x4d', timm_loader='resnet'),

    'ResNeXt-101-32x4d': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 23, 3], 4, 32, 64),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnext101_32x4d', timm_loader='resnet'),

    'ResNeXt-101-32x8d': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 23, 3], 8, 32, 32),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation,
        timm_name='resnext101_32x8d', timm_loader='resnet'),

    'ResNeSt-50-2s1x64d': update_params(
        large_basic_params,
        layers=make_resnest_layers([3, 4, 6, 3], 64, 2, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=TweakedLargeStem, downsample=TweakedDownsample,
        operation=SplitAttentionOperation),

    'MobileNetV2-1.0': update_params(
        large_basic_params,
        layers=make_mobilenetv2_layers(1.0),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.ReLU6),

    'MobileNetV2-0.5': update_params(
        large_basic_params,
        layers=make_mobilenetv2_layers(0.5),
        stem_channels=16, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.ReLU6),

    'MobileNetV3-large': update_params(
        large_basic_params,
        layers=make_mobilenetv3_large_layers(1.0),
        stem_channels=16, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV3Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=HSwish,
        seoperation=False, seoperation_reduction=4,
        seoperation_sigmoid=lambda: HSigmoid(inplace=True)),

    'EfficientNet-B0': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.0, 1.0),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.SiLU,
        timm_name='efficientnet_b0', timm_loader='efficientnet'),

    'EfficientNet-B1': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.0, 1.1),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.SiLU,
        timm_name='efficientnet_b1', timm_loader='efficientnet'),

    'EfficientNet-B2': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.1, 1.2),
        stem_channels=32, head_channels=1408,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.SiLU,
        timm_name='efficientnet_b2', timm_loader='efficientnet'),

    'EfficientNet-B3': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.2, 1.4),
        stem_channels=40, head_channels=1536,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.SiLU,
        timm_name='efficientnet_b3', timm_loader='efficientnet'),

    'DenseNet-121': update_params(
        large_basic_params,
        layers=make_densenet_layers([6, 12, 24, 16], 64, 32, 4),
        stem_channels=64, head_channels=1024,
        head=PreActHead, block=DenseNetBlock, operation=DenseNetOperation,
        downsample=NoneDownsample, junction=ConcatJunction),

    'DenseNet-169': update_params(
        large_basic_params,
        layers=make_densenet_layers([6, 12, 32, 32], 64, 32, 4),
        stem_channels=64, head_channels=1664,
        head=PreActHead, block=DenseNetBlock, operation=DenseNetOperation,
        downsample=NoneDownsample, junction=ConcatJunction),

    # In the paper, semodule_gain = 2.0, alpha = 0.2.
    # But in this implementation, due to same procedure,
    # semodule_gain = 0.4, alpha = 1.0 in this implementation
    # because alpha is multiplied to features before se-module.
    'NFNet-F0': update_params(
        large_basic_params,
        layers=make_nfnet_layers([1, 2, 6, 3], 128, 1, 2, alpha=0.2),
        stem_channels=128, head_channels=3072,
        stem=NFNetStem, block=PreActBlock, operation=NFOperation,
        downsample=NFDownsample, head=NFHead,
        semodule=True, semodule_reduction=2, semodule_gain=0.4,
        normalization=lambda *args, **kwargs: nn.Identity(),
        activation=lambda *args, **kwargs: nn.GELU(),
        alpha=1.0, gamma=1.7015043497085571),

    'NFNet-F1': update_params(
        large_basic_params,
        layers=make_nfnet_layers([2, 4, 12, 6], 128, 1, 2, alpha=0.2),
        stem_channels=128, head_channels=3072,
        stem=NFNetStem, block=PreActBlock, operation=NFOperation,
        downsample=NFDownsample, head=NFHead,
        semodule=True, semodule_reduction=2, semodule_gain=0.4,
        normalization=lambda *args, **kwargs: nn.Identity(),
        activation=lambda *args, **kwargs: nn.GELU(),
        alpha=1.0, gamma=1.7015043497085571),

    'ViTSmallPatch16-224': update_params(
        large_basic_params,
        layers=make_vit_layers([12], 384),
        stem_channels=384, head_channels=384,
        patch_size=16, num_patches=196, attn_heads=6, mlp_ratio=4.0,
        stem=ViTPatchStem, block=ViTBlock, operation=ViTOperation,
        junction=BasicJunction, head=ViTHead,
        normalization=functools.partial(nn.LayerNorm, eps=1e-6),
        activation=lambda *args, **kwargs: nn.GELU(),
        timm_name='vit_small_patch16_224', timm_loader='vit'),

    'ViTBasePatch16-224': update_params(
        large_basic_params,
        layers=make_vit_layers([12], 768),
        stem_channels=768, head_channels=768,
        patch_size=16, num_patches=196, attn_heads=12, mlp_ratio=4.0,
        stem=ViTPatchStem, block=ViTBlock, operation=ViTOperation,
        junction=BasicJunction, head=ViTHead,
        normalization=functools.partial(nn.LayerNorm, eps=1e-6),
        activation=lambda *args, **kwargs: nn.GELU(),
        timm_name='vit_base_patch16_224', timm_loader='vit'),
}

small_basic_params = update_params(large_basic_params, stem=BasicSmallStem)

small_models = {
    'ResNet-20': update_params(
        small_basic_params,
        layers=make_resnet_layers([3, 3, 3], 16, 1, 1),
        stem_channels=16, head_channels=64),

    'ResNet-110': update_params(
        small_basic_params,
        layers=make_resnet_layers([18, 18, 18], 16, 1, 1),
        stem_channels=16, head_channels=64),

    'ResNet-200': update_params(
        small_basic_params,
        layers=make_resnet_layers([33, 33, 33], 16, 1, 1),
        stem_channels=16, head_channels=64),

    'SE-ResNet-110': update_params(
        small_basic_params,
        layers=make_resnet_layers([18, 18, 18], 16, 1, 1),
        stem_channels=16, head_channels=64, semodule=True),

    'WideResNet-28-k10': update_params(
        small_basic_params,
        layers=make_resnet_layers([4, 4, 4], 160, 1, 1),
        stem_channels=16, head_channels=640,
        stem=PreActSmallStem, head=PreActHead, block=PreActBlock,
        operation=PreActBasicOperation),

    'WideResNet-40-k4': update_params(
        small_basic_params,
        layers=make_resnet_layers([6, 6, 6], 64, 1, 1),
        stem_channels=16, head_channels=256,
        stem=PreActSmallStem, head=PreActHead, block=PreActBlock,
        operation=PreActBasicOperation),

    'WideResNet-40-k10': update_params(
        small_basic_params,
        layers=make_resnet_layers([6, 6, 6], 160, 1, 1),
        stem_channels=16, head_channels=640,
        stem=PreActSmallStem, head=PreActHead, block=PreActBlock,
        operation=PreActBasicOperation),

    'ResNeXt-29-8x64d': update_params(
        small_basic_params,
        layers=make_resnet_layers([3, 3, 3], 64, 8, 4),
        stem_channels=64, head_channels=1024,
        operation=BottleneckOperation),

    'ResNeXt-47-32x4d': update_params(
        small_basic_params,
        layers=make_resnet_layers([5, 5, 5], 4, 32, 64),
        stem_channels=16, head_channels=1024,
        operation=BottleneckOperation),

    'ResNeSt-47-2s1x64d': update_params(
        small_basic_params,
        layers=make_resnest_layers([5, 5, 5], 64, 2, 1, 4),
        stem_channels=16, head_channels=1024,
        downsample=TweakedDownsample, operation=SplitAttentionOperation),

    'ResNeSt-47-2s1x128d': update_params(
        small_basic_params,
        layers=make_resnest_layers([5, 5, 5], 128, 2, 1, 2),
        stem_channels=16, head_channels=1024,
        downsample=TweakedDownsample, operation=SplitAttentionOperation),

    'PyramidNet-110-a48': update_params(
        small_basic_params,
        layers=make_pyramid_layers([18, 18, 18], 16, 48, 1, 1),
        stem_channels=16, head_channels=64,
        stem=PreActSmallStem, head=PreActHead,
        block=PreActBlock, downsample=AverageDownsample,
        operation=SingleActBasicOperation),

    'PyramidNet-110-a270': update_params(
        small_basic_params,
        layers=make_pyramid_layers([18, 18, 18], 16, 270, 1, 1),
        stem_channels=16, head_channels=286,
        stem=PreActSmallStem, head=PreActHead,
        block=PreActBlock, downsample=AverageDownsample,
        operation=SingleActBasicOperation),

    'PyramidNet-200-a240': update_params(
        small_basic_params,
        layers=make_pyramid_layers([22, 22, 22], 16, 240, 1, 4),
        stem_channels=16, head_channels=1024,
        stem=PreActSmallStem, head=PreActHead,
        block=PreActBlock, downsample=AverageDownsample,
        operation=SingleActBottleneckOperation),

    'PyramidNet-272-a200': update_params(
        small_basic_params,
        layers=make_pyramid_layers([30, 30, 30], 16, 200, 1, 4),
        stem_channels=16, head_channels=864,
        stem=PreActSmallStem, head=PreActHead,
        block=PreActBlock, downsample=AverageDownsample,
        operation=SingleActBottleneckOperation),
}

size256_models = {}
size256_models.update(large_models)
size256_models.update(dense_models(size256_models))
size256_models.update(skip_models(size256_models))
size256_models.update(dynamic_dense_models(size256_models))
size256_models.update(static_dense_models(size256_models))
size256_models.update(mean_dense_models(size256_models))
size256_models.update(sum_dense_models(size256_models))

size32_models = {}
size32_models.update(small_models)
size32_models.update(dense_models(size32_models))
size32_models.update(skip_models(size32_models))
size32_models.update(dynamic_dense_models(size32_models))
size32_models.update(static_dense_models(size32_models))
size32_models.update(mean_dense_models(size32_models))
size32_models.update(sum_dense_models(size32_models))

PARAMETERS: Dict[str, Dict[str, Any]] = {
    'imagenet': update_models(size256_models, num_classes=1000),
    'cifar100': update_models(size32_models, num_classes=100),
    'cifar10': update_models(size32_models, num_classes=10),
    'dummy': update_models(size32_models, num_classes=10),
}
