import models
import timm
import torch


def test_vit_small_patch16_224() -> None:
    model = models.create_model('imagenet', 'ViTSmallPatch16-224', pretrained=True)
    timm_model = timm.create_model('vit_small_patch16_224', pretrained=True)
    _test_model(model, timm_model, image_size=224)


def test_vit_base_patch16_224() -> None:
    model = models.create_model('imagenet', 'ViTBasePatch16-224', pretrained=True)
    timm_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    _test_model(model, timm_model, image_size=224)


@torch.no_grad()
def _test_model(model, timm_model, image_size) -> None:
    timm_model.eval()
    model.eval()

    x = torch.randn([1, 3, image_size, image_size], dtype=torch.float32)

    assert (timm_model(x) - model(x)).abs().sum() < 1e-3
