from typing import Any
import torch.nn as nn


def load_tresnet_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.body.conv1[0].state_dict())
    model.stem.norm.load_state_dict(timm_model.body.conv1[1].state_dict())

    index = 0
    for timm_layer in (timm_model.body.layer1,
                       timm_model.body.layer2,
                       timm_model.body.layer3,
                       timm_model.body.layer4):
        for timm_block in timm_layer:
            if hasattr(timm_block, 'conv3'):
                _load_bottleneck_block_parameters(model.blocks[index], timm_block)
            else:
                _load_basic_block_parameters(model.blocks[index], timm_block)

            index += 1

    _load_fc2conv_parameters(model.classifier.conv, timm_model.head.fc)


def _load_basic_block_parameters(block: Any, timm_block: Any) -> None:
    if isinstance(timm_block.conv1[0], nn.Sequential):
        block.operation.conv1.load_state_dict(timm_block.conv1[0][0].state_dict())
        block.operation.norm1.load_state_dict(timm_block.conv1[0][1].state_dict())
    else:
        block.operation.conv1.load_state_dict(timm_block.conv1[0].state_dict())
        block.operation.norm1.load_state_dict(timm_block.conv1[1].state_dict())

    block.operation.conv2.load_state_dict(timm_block.conv2[0].state_dict())
    block.operation.norm2.load_state_dict(timm_block.conv2[1].state_dict())

    if hasattr(timm_block, 'se') and timm_block.se is not None:
        _load_semodule_parameters(block.operation.semodule, timm_block.se)

    if hasattr(timm_block, 'downsample') and timm_block.downsample is not None:
        _load_downsample_parameters(block.downsample, timm_block.downsample)


def _load_bottleneck_block_parameters(block: Any, timm_block: Any) -> None:
    block.operation.conv1.load_state_dict(timm_block.conv1[0].state_dict())
    block.operation.norm1.load_state_dict(timm_block.conv1[1].state_dict())

    if isinstance(timm_block.conv2[0], nn.Sequential):
        block.operation.conv2.load_state_dict(timm_block.conv2[0][0].state_dict())
        block.operation.norm2.load_state_dict(timm_block.conv2[0][1].state_dict())
    else:
        block.operation.conv2.load_state_dict(timm_block.conv2[0].state_dict())
        block.operation.norm2.load_state_dict(timm_block.conv2[1].state_dict())

    block.operation.conv3.load_state_dict(timm_block.conv3[0].state_dict())
    block.operation.norm3.load_state_dict(timm_block.conv3[1].state_dict())

    if hasattr(timm_block, 'se') and timm_block.se is not None:
        _load_semodule_parameters(block.operation.semodule, timm_block.se)

    if hasattr(timm_block, 'downsample') and timm_block.downsample is not None:
        _load_downsample_parameters(block.downsample, timm_block.downsample)


def _load_semodule_parameters(semodule: Any, timm_semodule: Any) -> None:
    semodule.op.conv1.load_state_dict(timm_semodule.fc1.state_dict())
    semodule.op.conv2.load_state_dict(timm_semodule.fc2.state_dict())


def _load_downsample_parameters(downsample: Any, timm_downsample: Any) -> None:
    downsample.conv.load_state_dict(timm_downsample[1][0].state_dict())
    downsample.norm.load_state_dict(timm_downsample[1][1].state_dict())


def _load_fc2conv_parameters(conv: Any, timm_fc: Any) -> None:
    conv.weight.data[:, :, 0, 0] = timm_fc.weight.data
    if conv.bias is not None:
        conv.bias.data[:] = timm_fc.bias.data
