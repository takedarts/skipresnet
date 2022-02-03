from typing import Any


def load_convnext_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.stem[0].state_dict())
    model.stem.norm.load_state_dict(timm_model.stem[1].state_dict())

    index = 0

    for i, timm_stage in enumerate(timm_model.stages):
        if i != 0:
            _load_convnext_downsample_parameters(
                model.blocks[index], timm_stage.downsample)

        for timm_block in timm_stage.blocks:
            _load_convnext_block_parameters(model.blocks[index], timm_block)
            index += 1

    model.classifier.norm.load_state_dict(timm_model.head.norm.state_dict())
    _load_linear_to_conv2d_parameters(model.classifier.conv, timm_model.head.fc)


def _load_convnext_block_parameters(block: Any, timm_block: Any) -> None:
    block.operation.dwconv.load_state_dict(timm_block.conv_dw.state_dict())
    block.operation.norm.load_state_dict(timm_block.norm.state_dict())
    _load_linear_to_conv2d_parameters(block.operation.pwconv1, timm_block.mlp.fc1)
    _load_linear_to_conv2d_parameters(block.operation.pwconv2, timm_block.mlp.fc2)

    if timm_block.gamma is not None:
        block.operation.gamma.data[:] = timm_block.gamma.data


def _load_convnext_downsample_parameters(block: Any, timm_down: Any) -> None:
    block.downsample.norm.load_state_dict(timm_down[0].state_dict())
    block.downsample.conv.load_state_dict(timm_down[1].state_dict())


def _load_linear_to_conv2d_parameters(conv2d: Any, linear: Any) -> None:
    conv2d.weight.data[:] = linear.weight[:, :, None, None].data

    if linear.bias is not None:
        conv2d.bias.data[:] = linear.bias.data
    else:
        assert conv2d.bias is None
