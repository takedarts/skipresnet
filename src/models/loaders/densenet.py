from typing import Any


def load_densenet_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.features.conv0.state_dict())
    model.stem.norm.load_state_dict(timm_model.features.norm0.state_dict())

    index = 0

    for block, trans in (
            (timm_model.features.denseblock1, timm_model.features.transition1),
            (timm_model.features.denseblock2, timm_model.features.transition2),
            (timm_model.features.denseblock3, timm_model.features.transition3),
            (timm_model.features.denseblock4, None),
    ):
        for _, layer in block.items():
            block = model.blocks[index]
            block.operation.norm1.load_state_dict(layer.norm1.state_dict())
            block.operation.conv1.load_state_dict(layer.conv1.state_dict())
            block.operation.norm2.load_state_dict(layer.norm2.state_dict())
            block.operation.conv2.load_state_dict(layer.conv2.state_dict())
            index += 1

        if trans is not None:
            block = model.blocks[index]
            block.operation.norm1.load_state_dict(trans.norm.state_dict())
            block.operation.conv1.load_state_dict(trans.conv.state_dict())
            index += 1

    model.head.norm.load_state_dict(timm_model.features.norm5.state_dict())

    if model.classifier.conv.weight.shape[:2] == timm_model.classifier.weight.shape:
        model.classifier.conv.weight.data[:] = timm_model.classifier.weight[:, :, None, None].data
        model.classifier.conv.bias.data[:] = timm_model.classifier.bias.data
