from typing import List, Tuple

import models
import pytest
import timm
import torch
import torch.nn as nn

# ConvNeXt models are loaded.
# This code will be removed when timm supoprts ConvNeXt.
from models.reference import convnext
convnext.load()


pretrained_model_names: List[Tuple[str, str, int]] = [
    # ResNets
    ('ResNet-18', 'resnet18', 224),
    ('ResNet-34', 'resnet34', 224),
    ('ResNet-50', 'resnet50', 224),
    ('SE-ResNet-50', 'seresnet50', 224),
    ('ResNeXt-50-32x4d', 'resnext50_32x4d', 224),
    ('ResNeXt-101-32x8d', 'resnext101_32x8d', 224),
    ('ResNetD-50', 'resnet50d', 224),
    ('ResNetD-101', 'resnet101d', 224),

    # ResNeSt
    ('ResNeSt-50-2s1x64d', 'resnest50d', 224),
    ('ResNeSt-50-1s4x24d', 'resnest50d_1s4x24d', 224),
    ('ResNeSt-50-4s2x40d', 'resnest50d_4s2x40d', 224),
    ('ResNeSt-101-2s1x64d', 'resnest101e', 224),
    ('ResNeSt-200-2s1x64d', 'resnest200e', 224),

    # MobileNets
    ('MobileNetV3-large', 'mobilenetv3_large_100', 224),
    ('MobileNetV2-1.0', 'mobilenetv2_100', 224),
    ('MobileNetV2-1.4', 'mobilenetv2_140', 224),

    # EfficientNets
    ('EfficientNet-B0', 'tf_efficientnet_b0', 224),
    ('EfficientNet-B1', 'tf_efficientnet_b1', 224),
    ('EfficientNet-B2', 'tf_efficientnet_b2', 224),
    ('EfficientNet-B3', 'tf_efficientnet_b3', 224),

    # EfficientNetV2s
    ('EfficientNetV2-S', 'tf_efficientnetv2_s', 224),
    ('EfficientNetV2-M', 'tf_efficientnetv2_m', 224),
    ('EfficientNetV2-L', 'tf_efficientnetv2_l', 224),

    # RegNets
    ('RegNetX-0.8', 'regnetx_008', 224),
    ('RegNetX-1.6', 'regnetx_016', 224),
    ('RegNetX-3.2', 'regnetx_032', 224),
    ('RegNetY-0.8', 'regnety_008', 224),
    ('RegNetY-1.6', 'regnety_016', 224),
    ('RegNetY-3.2', 'regnety_032', 224),

    # DenseNets
    ('DenseNet-121', 'densenet121', 224),
    ('DenseNet-169', 'densenet169', 224),

    # NFNets
    ('NFNet-F0', 'dm_nfnet_f0', 224),
    ('NFNet-F1', 'dm_nfnet_f1', 224),

    # ViTs
    ('ViTSmallPatch16-224', 'vit_small_patch16_224', 224),
    ('ViTBasePatch16-224', 'vit_base_patch16_224', 224),
    ('ViTSmallPatch16-384', 'vit_small_patch16_384', 384),
    ('ViTBasePatch16-384', 'vit_base_patch16_384', 384),

    # SwinTransformers
    ('SwinTinyPatch4-224', 'swin_tiny_patch4_window7_224', 224),
    ('SwinSmallPatch4-224', 'swin_small_patch4_window7_224', 224),
    ('SwinBasePatch4-224', 'swin_base_patch4_window7_224', 224),

    # ConvNeXt
    ('ConvNeXt-T', 'convnext_tiny', 224),
    ('ConvNeXt-S', 'convnext_small', 224),
    ('ConvNeXt-B', 'convnext_base', 224),
]

random_model_names = [
    # ResNets
    ('ResNet-101', 'resnet101', 224),
    ('ResNet-152', 'resnet152', 224),
    ('SE-ResNet-34', 'seresnet34', 224),
    ('ResNeXt-101-32x4d', 'resnext101_32x4d', 224),
]


@pytest.mark.parametrize(
    'model_name,timm_model_name,image_size',
    pretrained_model_names,
    ids=[n for n, _, _ in pretrained_model_names])
def test_pretrained_models(model_name: str, timm_model_name: str, image_size: int) -> None:
    timm_model = timm.create_model(timm_model_name, pretrained=True)
    model = models.create_model(model_name, 'imagenet', pretrained=True)

    _test_model(model, timm_model, image_size=image_size)


@pytest.mark.parametrize(
    'model_name,timm_model_name,image_size',
    random_model_names,
    ids=[n for n, _, _ in random_model_names])
def test_random_models(model_name: str, timm_model_name: str, image_size: int) -> None:
    timm_model = timm.create_model(timm_model_name)
    for parameter in timm_model.parameters():
        nn.init.normal_(parameter, 0.0, 0.1)

    model = models.create_model(model_name, 'imagenet')
    model.load_model_parameters(timm_model)

    _test_model(model, timm_model, image_size=image_size)


@torch.no_grad()
def _test_model(
    model: nn.Module,
    timm_model: nn.Module,
    image_size: int,
) -> None:
    timm_model.eval()
    model.eval()

    x = torch.randn([1, 3, image_size, image_size], dtype=torch.float32)

    assert (model(x) - timm_model(x)).abs().sum() < 1e-3
