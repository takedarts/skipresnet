import models
import timm
import torch


def test_efficient_net_b0():
    model = models.create_model('imagenet', 'EfficientNet-B0', pretrained=True)
    timm_model = timm.create_model('efficientnet_b0', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def test_efficient_net_b1():
    model = models.create_model('imagenet', 'EfficientNet-B1', pretrained=True)
    timm_model = timm.create_model('efficientnet_b1', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def test_efficient_net_b2():
    model = models.create_model('imagenet', 'EfficientNet-B2', pretrained=True)
    timm_model = timm.create_model('efficientnet_b2', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def test_efficient_net_b3():
    model = models.create_model('imagenet', 'EfficientNet-B3', pretrained=True)
    timm_model = timm.create_model('efficientnet_b3', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def _test_model(model, timm_model, image_size):
    model.eval()
    timm_model.eval()

    x = torch.randn([1, 3, image_size, image_size], dtype=torch.float32)

    assert (timm_model(x) - model(x)).abs().sum() < 1e-3
