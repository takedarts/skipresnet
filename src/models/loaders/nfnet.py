from typing import Any


def load_nfnet_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv1.load_state_dict(timm_model.stem.conv1.state_dict())
    model.stem.conv2.load_state_dict(timm_model.stem.conv2.state_dict())
    model.stem.conv3.load_state_dict(timm_model.stem.conv3.state_dict())
    model.stem.conv4.load_state_dict(timm_model.stem.conv4.state_dict())

    index = 0

    for timm_stage in timm_model.stages:
        for timm_block in timm_stage:
            block = model.blocks[index]

            if timm_block.downsample is not None:
                block.downsample.conv.load_state_dict(
                    timm_block.downsample.conv.state_dict())

            block.operation.op.conv1.load_state_dict(timm_block.conv1.state_dict())
            block.operation.op.conv2.load_state_dict(timm_block.conv2.state_dict())
            block.operation.op.conv3.load_state_dict(timm_block.conv2b.state_dict())
            block.operation.op.conv4.load_state_dict(timm_block.conv3.state_dict())

            block.operation.op.semodule.op.conv1.load_state_dict(
                timm_block.attn_last.fc1.state_dict())
            block.operation.op.semodule.op.conv2.load_state_dict(
                timm_block.attn_last.fc2.state_dict())

            block.operation.gain.data = timm_block.skipinit_gain.data.clone()

            index += 1

    model.head.conv.load_state_dict(timm_model.final_conv.state_dict())

    if model.classifier.conv.weight.shape[:2] == timm_model.head.fc.weight.shape:
        model.classifier.conv.weight.data[:] = timm_model.head.fc.weight[:, :, None, None].data
        model.classifier.conv.bias.data[:] = timm_model.head.fc.bias.data
