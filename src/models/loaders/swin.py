from typing import Any
import torch


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
            model.blocks[index + 0].preprocess.norm.load_state_dict(
                layer.downsample.norm.state_dict())
            model.blocks[index + 0].preprocess.reduction.load_state_dict(
                layer.downsample.reduction.state_dict())

    model.head.norm.load_state_dict(timm_model.norm.state_dict())
    model.classifier.conv.weight.data[:] = timm_model.head.weight[:, :, None, None].data
    model.classifier.conv.bias.data[:] = timm_model.head.bias.data


def load_swin_attn_parameters(attn: Any, timm_attn: Any) -> None:
    attn.qkv.load_state_dict(timm_attn.qkv.state_dict())
    attn.proj.load_state_dict(timm_attn.proj.state_dict())

    # -- load `relative_position_bias_table` ---
    window_size = timm_attn.window_size
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += attn.window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += attn.window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * attn.window_size[1] - 1
    relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww, Wh*Ww
    relative_position_bias = timm_attn.relative_position_bias_table[relative_position_index].view(
        window_size[0] * window_size[1], window_size[0] * window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    relative_position_bias = relative_position_bias.unsqueeze(0)

    attn.relative_position_bias.data[:] = relative_position_bias.data
