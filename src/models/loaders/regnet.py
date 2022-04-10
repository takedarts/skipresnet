from typing import Any
import torch.nn as nn


def load_regnet_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.stem.conv.state_dict())
    model.stem.norm.load_state_dict(timm_model.stem.bn.state_dict())

    index = 0

    for timm_stage in list(timm_model.children())[1:-1]:
        for timm_block in timm_stage.children():
            block = model.blocks[index]

            if (timm_block.downsample is not None
                    and not isinstance(timm_block.downsample, nn.Identity)):
                block.downsample.conv.load_state_dict(timm_block.downsample.conv.state_dict())
                block.downsample.norm.load_state_dict(timm_block.downsample.bn.state_dict())

            block.operation.conv1.load_state_dict(timm_block.conv1.conv.state_dict())
            block.operation.norm1.load_state_dict(timm_block.conv1.bn.state_dict())
            block.operation.conv2.load_state_dict(timm_block.conv2.conv.state_dict())
            block.operation.norm2.load_state_dict(timm_block.conv2.bn.state_dict())

            if timm_block.se is not None and not isinstance(timm_block.se, nn.Identity):
                block.operation.semodule.op.conv1.load_state_dict(timm_block.se.fc1.state_dict())
                block.operation.semodule.op.conv2.load_state_dict(timm_block.se.fc2.state_dict())

            block.operation.conv3.load_state_dict(timm_block.conv3.conv.state_dict())
            block.operation.norm3.load_state_dict(timm_block.conv3.bn.state_dict())

            index += 1

    if model.classifier.conv.weight.shape[:2] == timm_model.head.fc.weight.shape:
        model.classifier.conv.weight.data[:] = timm_model.head.fc.weight[:, :, None, None].data
        model.classifier.conv.bias.data[:] = timm_model.head.fc.bias.data
