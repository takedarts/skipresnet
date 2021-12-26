import utils
import timm
import torch
import models
import warnings


def test_resnet34() -> None:
    assert (
        utils.count_operations(
            timm.create_model('resnet34'),
            torch.randn([1, 3, 224, 224], dtype=torch.float32))
        == 3_681_374_184)


def test_skipresnet34() -> None:
    warnings.filterwarnings(
        'ignore', category=UserWarning,
        message='This model contains a squeeze operation on dimension 1.')
    assert (
        utils.count_operations(
            models.create_model('Skip-ResNet-34'),
            torch.randn([1, 3, 224, 224], dtype=torch.float32))
        == 3_694_372_680)
