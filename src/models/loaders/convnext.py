from typing import Any


def load_convnext_parameters(model: Any, timm_model: Any) -> None:
    timm_stem = timm_model.downsample_layers[0]
    timm_downsample_layers = timm_model.downsample_layers[1:]

    model.stem.conv.load_state_dict(timm_stem[0].state_dict())
    model.stem.norm.load_state_dict(timm_stem[1].state_dict())

    index = 0

    for timm_block in timm_model.stages[0]:
        _load_convnext_block_parameters(model.blocks[index], timm_block)
        index += 1

    for timm_down, timm_stages in zip(timm_downsample_layers, timm_model.stages[1:]):
        _load_convnext_downsample_parameters(model.blocks[index], timm_down)
        for timm_block in timm_stages:
            _load_convnext_block_parameters(model.blocks[index], timm_block)
            index += 1

    model.classifier.norm.load_state_dict(timm_model.norm.state_dict())
    model.classifier.conv.weight.data[:] = timm_model.head.weight[:, :, None, None].data
    model.classifier.conv.bias.data[:] = timm_model.head.bias.data


def _load_convnext_block_parameters(block: Any, timm_block: Any) -> None:
    block.operation.dwconv.load_state_dict(timm_block.dwconv.state_dict())
    block.operation.norm.load_state_dict(timm_block.norm.state_dict())
    block.operation.pwconv1.load_state_dict(timm_block.pwconv1.state_dict())
    block.operation.pwconv2.load_state_dict(timm_block.pwconv2.state_dict())

    if timm_block.gamma is not None:
        block.operation.gamma.data[:] = timm_block.gamma.data


def _load_convnext_downsample_parameters(block: Any, timm_down: Any) -> None:
    block.preprocess.norm.load_state_dict(timm_down[0].state_dict())
    block.preprocess.conv.load_state_dict(timm_down[1].state_dict())
