from typing import Any


def load_swin_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.patch_embed.proj.state_dict())
    model.stem.norm.load_state_dict(timm_model.patch_embed.norm.state_dict())

    index = 0

    for layer in timm_model.layers:
        for block in layer.blocks:
            load_swin_attn_parameters(model.blocks[index + 0].operation.attn_attn, block.attn)
            model.blocks[index + 0].operation.attn_norm.load_state_dict(block.norm1.state_dict())
            model.blocks[index + 1].operation.mlp_norm.load_state_dict(block.norm2.state_dict())
            model.blocks[index + 1].operation.mlp_fc1.load_state_dict(block.mlp.fc1.state_dict())
            model.blocks[index + 1].operation.mlp_fc2.load_state_dict(block.mlp.fc2.state_dict())
            index += 2

        if layer.downsample is not None:
            model.blocks[index + 0].downsample.norm.load_state_dict(
                layer.downsample.norm.state_dict())
            model.blocks[index + 0].downsample.reduction.load_state_dict(
                layer.downsample.reduction.state_dict())

    model.head.norm.load_state_dict(timm_model.norm.state_dict())
    model.classifier.conv.weight.data[:] = timm_model.head.weight[:, :, None, None].data
    model.classifier.conv.bias.data[:] = timm_model.head.bias.data


def load_swin_attn_parameters(attn: Any, timm_attn: Any) -> None:
    attn.qkv.load_state_dict(timm_attn.qkv.state_dict())
    attn.proj.load_state_dict(timm_attn.proj.state_dict())
    attn.relative_position_bias_table.data[:] = timm_attn.relative_position_bias_table.data
