'''
This is a module for building an image classification model.
Copyright 2021 Atsushi TAKEDA
'''
import itertools
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn

from .modules import FrozenBatchNorm2d
from .parameter import PARAMETERS
from .loader import load_parameters

try:
    import timm
except BaseException:
    timm = None


def _freeze_parameters(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(m)


class Model(nn.Module):
    def __init__(
        self,
        stem: Callable[..., nn.Module],
        block: Callable[..., nn.Module],
        head: Callable[..., nn.Module],
        classifier: Callable[..., nn.Module],
        layers: List[Tuple[int, int, Dict[str, Any]]],
        stem_channels: int,
        head_channels: int,
        dropout: float,
        pretrained: bool = False,
        timm_name: str = None,
        timm_loader: str = None,
        **kwargs: Dict[str, Any]
    ):
        super().__init__()

        # make blocks
        channels = [stem_channels] + [c for c, _, _ in layers]
        settings = [(ic, oc, s) for ic, oc, (_, s, _) in zip(channels[:-1], channels[1:], layers)]
        dropblocks = list(itertools.accumulate(s - 1 for _, s, _ in layers))
        dropblocks = [v >= dropblocks[-1] - 1 for v in dropblocks]
        blocks = []

        for i, ((_, _, params), dropblock) in enumerate(zip(layers, dropblocks)):
            block_kwargs = kwargs.copy()
            block_kwargs.update(params)
            blocks.append(block(i, settings, dropblock=dropblock, ** block_kwargs))

        # modules
        self.stem = stem(stem_channels, **kwargs)
        self.blocks = nn.ModuleList(blocks)
        self.head = head(channels[-1], head_channels, **kwargs)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.classifier = classifier(head_channels, **kwargs)

        # load pretrained weights
        if pretrained:
            if timm_name is None or timm_loader is None:
                raise Exception(
                    'Required parameters for loading pretrained weights are not set.')
            self.load_pretrained_parameters(timm_name, timm_loader)

    def freeze_blocks(self) -> None:
        _freeze_parameters(self.stem)
        _freeze_parameters(self.blocks)
        _freeze_parameters(self.head)

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        y: List[torch.Tensor] = [self.stem(x)]
        f: List[torch.Tensor] = []

        for block in self.blocks:
            y = block(y)
            f.append(y[-1])

        z = self.head(y[-1])

        return z, f

    def get_prediction(self, x: torch.Tensor, aggregation: bool = True) -> torch.Tensor:
        if aggregation:
            x = nn.functional.adaptive_avg_pool2d(x, [1, 1])
            x = self.dropout(x)
            x = self.classifier(x)
            x = x.reshape(x.shape[0], -1)
        else:
            x = self.classifier(x)

        return x

    def forward(self, x: torch.Tensor, aggregation: bool = True) -> torch.Tensor:
        x = self.get_features(x)[0]
        x = self.get_prediction(x, aggregation=aggregation)

        return x

    def load_pretrained_parameters(self, timm_name: str, timm_loader: str) -> None:
        if timm is None:
            raise Exception('Module `timm` is required: try `pip install timm`')

        if timm_name not in timm.list_models(pretrained=True):
            raise Exception(f'Pretrained weights of `{timm_name}` do not exist.')

        timm_model = timm.create_model(timm_name, pretrained=True)
        load_parameters(self, timm_model, timm_loader)
        self.timm_model = timm_model


def create_model(dataset_name, model_name, **kwargs):
    model_params = {
        'normalization': nn.BatchNorm2d,
        'activation': nn.ReLU,
        'semodule': False,
        'semodule_reduction': 16,
        'semodule_activation': nn.ReLU,
        'semodule_sigmoid': nn.Sigmoid,
        'seoperation': False,
        'seoperation_reduction': 4,
        'seoperation_sigmoid': nn.Sigmoid,
        'gate_normalization': nn.BatchNorm2d,
        'gate_activation': nn.ReLU,
        'dropout': 0.0,
        'shakedrop': 0.0,
        'stochdepth': 0.0,
        'signalaugment': 0.0,
        'gate_reduction': 8,
        'dense_connections': 4,
        'skip_connections': 4,
    }

    model_params.update(PARAMETERS[dataset_name][model_name])
    model_params.update(kwargs)

    return Model(**model_params)
