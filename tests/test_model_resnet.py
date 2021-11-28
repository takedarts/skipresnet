import models
import timm
import torch


def test_resnet18():
    model = models.create_model('imagenet', 'ResNet-18', pretrained=True)
    timm_model = timm.create_model('resnet18', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def test_resnet34():
    model = models.create_model('imagenet', 'ResNet-34', pretrained=True)
    timm_model = timm.create_model('resnet34', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def test_resnet50():
    model = models.create_model('imagenet', 'ResNet-50', pretrained=True)
    timm_model = timm.create_model('resnet50', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def test_resnext50_32x4d():
    model = models.create_model('imagenet', 'ResNeXt-50-32x4d', pretrained=True)
    timm_model = timm.create_model('resnext50_32x4d', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def test_resnext101_32x8d():
    model = models.create_model('imagenet', 'ResNeXt-101-32x8d', pretrained=True)
    timm_model = timm.create_model('resnext101_32x8d', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def test_seresnet50():
    model = models.create_model('imagenet', 'SE-ResNet-50', pretrained=True)
    timm_model = timm.create_model('seresnet50', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def _test_model(model, timm_model, image_size):
    model.eval()
    timm_model.eval()

    x = torch.randn([1, 3, image_size, image_size], dtype=torch.float32)

    assert (timm_model(x) - model(x)).abs().sum() < 1e-3
