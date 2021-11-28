'''
This is a module for loading pre-trainied parameters from timm models.
Copyright 2021 Atsushi Takeda
'''
from typing import Any
import torch.nn as nn


def load_parameters(
    model: nn.Module, timm_model: nn.Module, load_name: str
) -> None:
    if load_name.lower() == 'resnet':
        _load_resnet_parameters(model, timm_model)
    elif load_name.lower() == 'efficientnet':
        _load_efficientnet_parameters(model, timm_model)
    elif load_name.lower() == 'vit':
        _load_vit_parameters(model, timm_model)
    else:
        raise Exception(f'Unsupported weight loader: {load_name}')


def _load_resnet_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.conv1.state_dict())
    model.stem.norm.load_state_dict(timm_model.bn1.state_dict())

    def copy_block_state_dict(block, timm_block):
        if timm_block.downsample is not None:
            block.downsample.conv.load_state_dict(timm_block.downsample[0].state_dict())
            block.downsample.norm.load_state_dict(timm_block.downsample[1].state_dict())

        if timm_block.se is not None:
            block.semodule.op.conv1.load_state_dict(timm_block.se.fc1.state_dict())
            block.semodule.op.conv2.load_state_dict(timm_block.se.fc2.state_dict())

        block.operation.conv1.load_state_dict(timm_block.conv1.state_dict())
        block.operation.norm1.load_state_dict(timm_block.bn1.state_dict())
        block.operation.conv2.load_state_dict(timm_block.conv2.state_dict())
        block.operation.norm2.load_state_dict(timm_block.bn2.state_dict())

        if hasattr(timm_block, 'conv3'):
            block.operation.conv3.load_state_dict(timm_block.conv3.state_dict())
            block.operation.norm3.load_state_dict(timm_block.bn3.state_dict())

    index = 0

    for timm_block in timm_model.layer1:
        copy_block_state_dict(model.blocks[index], timm_block)
        index += 1

    for timm_block in timm_model.layer2:
        copy_block_state_dict(model.blocks[index], timm_block)
        index += 1

    for ref_block in timm_model.layer3:
        copy_block_state_dict(model.blocks[index], ref_block)
        index += 1

    for ref_block in timm_model.layer4:
        copy_block_state_dict(model.blocks[index], ref_block)
        index += 1

    model.classifier.conv.weight.data[:] = timm_model.fc.weight[:, :, None, None].data
    model.classifier.conv.bias.data[:] = timm_model.fc.bias.data


def _load_efficientnet_parameters(model: Any, timm_model: Any) -> None:
    model.stem.conv.load_state_dict(timm_model.conv_stem.state_dict())
    model.stem.norm.load_state_dict(timm_model.bn1.state_dict())

    index = 0
    for stage in timm_model.blocks:
        for ref_block in stage:
            test_block = model.blocks[index]
            if hasattr(ref_block, 'conv_pwl'):
                test_block.operation.conv1.load_state_dict(ref_block.conv_pw.state_dict())
                test_block.operation.norm1.load_state_dict(ref_block.bn1.state_dict())
                test_block.operation.conv2.load_state_dict(ref_block.conv_dw.state_dict())
                test_block.operation.norm2.load_state_dict(ref_block.bn2.state_dict())
                test_block.operation.semodule.op.conv1.load_state_dict(
                    ref_block.se.conv_reduce.state_dict())
                test_block.operation.semodule.op.conv2.load_state_dict(
                    ref_block.se.conv_expand.state_dict())
                test_block.operation.conv3.load_state_dict(ref_block.conv_pwl.state_dict())
                test_block.operation.norm3.load_state_dict(ref_block.bn3.state_dict())
            else:
                test_block.operation.conv2.load_state_dict(ref_block.conv_dw.state_dict())
                test_block.operation.norm2.load_state_dict(ref_block.bn1.state_dict())
                test_block.operation.semodule.op.conv1.load_state_dict(
                    ref_block.se.conv_reduce.state_dict())
                test_block.operation.semodule.op.conv2.load_state_dict(
                    ref_block.se.conv_expand.state_dict())
                test_block.operation.conv3.load_state_dict(ref_block.conv_pw.state_dict())
                test_block.operation.norm3.load_state_dict(ref_block.bn2.state_dict())
            index += 1

    model.head.conv.load_state_dict(timm_model.conv_head.state_dict())
    model.head.norm.load_state_dict(timm_model.bn2.state_dict())
    model.classifier.conv.weight.data[:] = timm_model.classifier.weight[:, :, None, None].data
    model.classifier.conv.bias.data[:] = timm_model.classifier.bias.data


def _load_vit_parameters(model: Any, timm_model: Any) -> None:
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
    model.classifier.conv.weight.data[:] = timm_model.head.weight[:, :, None, None].data
    model.classifier.conv.bias.data[:] = timm_model.head.bias.data
