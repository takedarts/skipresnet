from typing import Any


def load_efficientnetv2_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.conv_stem.state_dict())
    model.stem.norm.load_state_dict(timm_model.bn1.state_dict())

    index = 0

    for timm_blocks in timm_model.blocks:
        for timm_block in timm_blocks:
            block = model.blocks[index]
            index += 1

            if hasattr(timm_block, 'conv_pw'):
                _load_inverted_block_parameters(block, timm_block)
            elif hasattr(timm_block, 'conv_exp'):
                _load_edge_block_parameters(block, timm_block)
            else:
                _load_conv_block_parameters(block, timm_block)

    model.head.conv.load_state_dict(timm_model.conv_head.state_dict())
    model.head.norm.load_state_dict(timm_model.bn2.state_dict())
    model.classifier.conv.weight.data[:] = timm_model.classifier.weight[:, :, None, None].data
    model.classifier.conv.bias.data[:] = timm_model.classifier.bias.data


def _load_inverted_block_parameters(block: Any, timm_block: Any) -> None:
    block.operation.conv1.load_state_dict(timm_block.conv_pw.state_dict())
    block.operation.norm1.load_state_dict(timm_block.bn1.state_dict())
    block.operation.conv2.load_state_dict(timm_block.conv_dw.state_dict())
    block.operation.norm2.load_state_dict(timm_block.bn2.state_dict())
    block.operation.conv3.load_state_dict(timm_block.conv_pwl.state_dict())
    block.operation.norm3.load_state_dict(timm_block.bn3.state_dict())

    if hasattr(block.operation, 'semodule'):
        block.operation.semodule.op.conv1.load_state_dict(
            timm_block.se.conv_reduce.state_dict())
        block.operation.semodule.op.conv2.load_state_dict(
            timm_block.se.conv_expand.state_dict())


def _load_edge_block_parameters(block: Any, timm_block: Any) -> None:
    block.operation.conv1.load_state_dict(timm_block.conv_exp.state_dict())
    block.operation.norm1.load_state_dict(timm_block.bn1.state_dict())
    block.operation.conv2.load_state_dict(timm_block.conv_pwl.state_dict())
    block.operation.norm2.load_state_dict(timm_block.bn2.state_dict())

    if hasattr(block.operation, 'semodule'):
        block.operation.semodule.op.conv1.load_state_dict(
            timm_block.se.conv_reduce.state_dict())
        block.operation.semodule.op.conv2.load_state_dict(
            timm_block.se.conv_expand.state_dict())


def _load_conv_block_parameters(block: Any, timm_block: Any) -> None:
    block.operation.conv.load_state_dict(timm_block.conv.state_dict())
    block.operation.norm.load_state_dict(timm_block.bn1.state_dict())
