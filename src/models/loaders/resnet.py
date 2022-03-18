from typing import Any


def load_resnet_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.conv1.state_dict())
    model.stem.norm.load_state_dict(timm_model.bn1.state_dict())

    index = 0

    for timm_layer in (timm_model.layer1, timm_model.layer2,
                       timm_model.layer3, timm_model.layer4):
        for timm_block in timm_layer:
            block = model.blocks[index]

            if timm_block.downsample is not None:
                block.downsample.conv.load_state_dict(timm_block.downsample[0].state_dict())
                block.downsample.norm.load_state_dict(timm_block.downsample[1].state_dict())

            if timm_block.se is not None:
                block.operation.semodule.op.conv1.load_state_dict(timm_block.se.fc1.state_dict())
                block.operation.semodule.op.conv2.load_state_dict(timm_block.se.fc2.state_dict())

            block.operation.conv1.load_state_dict(timm_block.conv1.state_dict())
            block.operation.norm1.load_state_dict(timm_block.bn1.state_dict())
            block.operation.conv2.load_state_dict(timm_block.conv2.state_dict())
            block.operation.norm2.load_state_dict(timm_block.bn2.state_dict())

            if hasattr(timm_block, 'conv3'):
                block.operation.conv3.load_state_dict(timm_block.conv3.state_dict())
                block.operation.norm3.load_state_dict(timm_block.bn3.state_dict())

            index += 1

    if model.classifier.conv.weight.shape[:2] == timm_model.fc.weight.shape:
        model.classifier.conv.weight.data[:] = timm_model.fc.weight[:, :, None, None].data
        model.classifier.conv.bias.data[:] = timm_model.fc.bias.data


def load_resnetd_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv1.load_state_dict(timm_model.conv1[0].state_dict())
    model.stem.norm1.load_state_dict(timm_model.conv1[1].state_dict())
    model.stem.conv2.load_state_dict(timm_model.conv1[3].state_dict())
    model.stem.norm2.load_state_dict(timm_model.conv1[4].state_dict())
    model.stem.conv3.load_state_dict(timm_model.conv1[6].state_dict())
    model.stem.norm3.load_state_dict(timm_model.bn1.state_dict())

    index = 0

    for timm_layer in (timm_model.layer1, timm_model.layer2,
                       timm_model.layer3, timm_model.layer4):
        for timm_block in timm_layer:
            block = model.blocks[index]

            if timm_block.downsample is not None:
                block.downsample.conv.load_state_dict(
                    timm_block.downsample[1].state_dict())
                block.downsample.norm.load_state_dict(
                    timm_block.downsample[2].state_dict())

            block.operation.conv1.load_state_dict(
                timm_block.conv1.state_dict())
            block.operation.norm1.load_state_dict(
                timm_block.bn1.state_dict())
            block.operation.conv2.load_state_dict(
                timm_block.conv2.state_dict())
            block.operation.norm2.load_state_dict(
                timm_block.bn2.state_dict())
            block.operation.conv3.load_state_dict(
                timm_block.conv3.state_dict())
            block.operation.norm3.load_state_dict(
                timm_block.bn3.state_dict())

            index += 1

    model.classifier.conv.weight.data[:] = timm_model.fc.weight[:, :, None, None].data
    model.classifier.conv.bias.data[:] = timm_model.fc.bias.data
