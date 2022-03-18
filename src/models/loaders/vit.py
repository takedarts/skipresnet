from typing import Any


def load_vit_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.patch_embed.proj.state_dict())
    model.stem.cls_token.data[0, :, 0, 0] = timm_model.cls_token.data[0, 0]
    model.stem.pos_embed.data[0, :, :, 0] = timm_model.pos_embed[0].permute(1, 0).data

    for i in range(len(model.blocks) // 2):
        model.blocks[i * 2 + 0].operation.attn_norm.load_state_dict(
            timm_model.blocks[i].norm1.state_dict())
        model.blocks[i * 2 + 0].operation.attn_qkv.load_state_dict(
            timm_model.blocks[i].attn.qkv.state_dict())
        model.blocks[i * 2 + 0].operation.attn_proj.load_state_dict(
            timm_model.blocks[i].attn.proj.state_dict())

        model.blocks[i * 2 + 1].operation.mlp_norm.load_state_dict(
            timm_model.blocks[i].norm2.state_dict())
        model.blocks[i * 2 + 1].operation.mlp_fc1.load_state_dict(
            timm_model.blocks[i].mlp.fc1.state_dict())
        model.blocks[i * 2 + 1].operation.mlp_fc2.load_state_dict(
            timm_model.blocks[i].mlp.fc2.state_dict())

    model.head.norm.load_state_dict(timm_model.norm.state_dict())

    if model.classifier.conv.weight.shape[:2] == timm_model.head.weight.shape:
        model.classifier.conv.weight.data[:] = timm_model.head.weight[:, :, None, None].data
        model.classifier.conv.bias.data[:] = timm_model.head.bias.data
