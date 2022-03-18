from typing import Any


def load_mobilenetv2_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.conv_stem.state_dict())
    model.stem.norm.load_state_dict(timm_model.bn1.state_dict())

    index = 0
    for stage in timm_model.blocks:
        for ref_block in stage:
            block = model.blocks[index]
            if hasattr(ref_block, 'conv_pwl'):
                block.operation.conv1.load_state_dict(ref_block.conv_pw.state_dict())
                block.operation.norm1.load_state_dict(ref_block.bn1.state_dict())
                block.operation.conv2.load_state_dict(ref_block.conv_dw.state_dict())
                block.operation.norm2.load_state_dict(ref_block.bn2.state_dict())
                block.operation.conv3.load_state_dict(ref_block.conv_pwl.state_dict())
                block.operation.norm3.load_state_dict(ref_block.bn3.state_dict())
            else:
                block.operation.conv2.load_state_dict(ref_block.conv_dw.state_dict())
                block.operation.norm2.load_state_dict(ref_block.bn1.state_dict())
                block.operation.conv3.load_state_dict(ref_block.conv_pw.state_dict())
                block.operation.norm3.load_state_dict(ref_block.bn2.state_dict())
            index += 1

    model.head.conv.load_state_dict(timm_model.conv_head.state_dict())
    model.head.norm.load_state_dict(timm_model.bn2.state_dict())

    if model.classifier.conv.weight.shape[:2] == timm_model.classifier.weight.shape:
        model.classifier.conv.weight.data[:] = timm_model.classifier.weight[:, :, None, None].data
        model.classifier.conv.bias.data[:] = timm_model.classifier.bias.data


def load_mobilenetv3_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.conv_stem.state_dict())
    model.stem.norm.load_state_dict(timm_model.bn1.state_dict())

    index = 0
    for stage in timm_model.blocks[:-1]:
        for ref_block in stage:
            block = model.blocks[index]
            if hasattr(ref_block, 'conv_pwl'):
                block.operation.conv1.load_state_dict(ref_block.conv_pw.state_dict())
                block.operation.norm1.load_state_dict(ref_block.bn1.state_dict())
                block.operation.conv2.load_state_dict(ref_block.conv_dw.state_dict())
                block.operation.norm2.load_state_dict(ref_block.bn2.state_dict())
                if hasattr(block.operation, 'semodule'):
                    block.operation.semodule.op.conv1.load_state_dict(
                        ref_block.se.conv_reduce.state_dict())
                    block.operation.semodule.op.conv2.load_state_dict(
                        ref_block.se.conv_expand.state_dict())
                block.operation.conv3.load_state_dict(ref_block.conv_pwl.state_dict())
                block.operation.norm3.load_state_dict(ref_block.bn3.state_dict())
            else:
                block.operation.conv2.load_state_dict(ref_block.conv_dw.state_dict())
                block.operation.norm2.load_state_dict(ref_block.bn1.state_dict())
                block.operation.conv3.load_state_dict(ref_block.conv_pw.state_dict())
                block.operation.norm3.load_state_dict(ref_block.bn2.state_dict())
            index += 1

    model.head.conv1.load_state_dict(timm_model.blocks[-1][0].conv.state_dict())
    model.head.norm1.load_state_dict(timm_model.blocks[-1][0].bn1.state_dict())
    model.head.conv2.load_state_dict(timm_model.conv_head.state_dict())

    if model.classifier.conv.weight.shape[:2] == timm_model.classifier.weight.shape:
        model.classifier.conv.weight.data[:] = timm_model.classifier.weight[:, :, None, None].data
        model.classifier.conv.bias.data[:] = timm_model.classifier.bias.data
