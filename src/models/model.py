'''
This is a module for building an image classification model.
Copyright 2021 Atsushi TAKEDA
'''
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parameter import PARAMETERS

try:
    import timm
except BaseException:
    timm = None


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
        pretrained: bool = False,
        timm_name: Optional[str] = None,
        timm_loader: Optional[Callable[[nn.Module, nn.Module], None]] = None,
        **kwargs: Dict[str, Any]
    ):
        super().__init__()

        # make blocks
        channels = [stem_channels] + [c for c, _, _ in layers]
        strides = [s for _, s, _ in layers]
        settings = [s for s in zip(channels[:-1], channels[1:], strides)]
        dropblocks = list(itertools.accumulate(s - 1 for _, s, _ in layers))
        dropblocks = [v >= dropblocks[-1] - 1 for v in dropblocks]
        blocks = []

        for i, ((_, _, params), dropblock) in enumerate(zip(layers, dropblocks)):
            block_kwargs = kwargs.copy()
            block_kwargs.update(params)
            blocks.append(block(
                index=i,
                settings=settings,
                dropblock=dropblock,
                **block_kwargs,
            ))

        # modules
        self.stem = stem(stem_channels, **kwargs)
        self.blocks = nn.ModuleList(blocks)
        self.head = head(channels[-1], head_channels, **kwargs)
        self.classifier = classifier(head_channels, **kwargs)

        # loader
        self.timm_name = timm_name
        self.timm_loader = timm_loader

        if pretrained:
            self.load_pretrained_parameters()

    def get_features(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        y: List[torch.Tensor] = [self.stem(x)]
        f: List[torch.Tensor] = []

        for block in self.blocks:
            y = block(y)
            f.append(y[-1])

        return f

    def get_prediction(
        self,
        x: torch.Tensor,
        aggregation: bool = True,
    ) -> torch.Tensor:
        x = self.head(x)

        if aggregation:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = self.classifier(x)
            x = x.reshape(x.shape[0], -1)
        else:
            x = self.classifier(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        aggregation: bool = True,
    ) -> torch.Tensor:
        x = self.get_features(x)[-1]
        x = self.get_prediction(x, aggregation=aggregation)

        return x

    def load_model_parameters(self, model: nn.Module) -> None:
        if self.timm_loader is None:
            raise Exception('Function for loading weights is not set.')

        self.timm_loader(self, model)

    def load_pretrained_parameters(self) -> None:
        if timm is None:
            raise Exception(
                'Module `timm` is required: try `pip install timm`')

        if self.timm_name is None:
            raise Exception('Name of a pretrained model is not set.')

        if self.timm_name not in timm.list_models(pretrained=True):
            raise Exception(
                f'Pretrained weights of `{self.timm_name}` is not found.')

        model = timm.create_model(self.timm_name, pretrained=True)
        self.load_model_parameters(model)


def create_model(
    model_name: str,
    dataset_name_or_num_classes: Union[str, int],
    **kwargs,
) -> 'Model':
    '''Build the specified model.
    Args:
        model_name: Model name.
        dataset_name_or_num_classes: Dataset name or the number of classses.
        kwargs: Additional parameters.
    Returns:
        A model object.
    '''

    # Set the number of classes
    if isinstance(dataset_name_or_num_classes, int):
        num_classes = dataset_name_or_num_classes
    elif dataset_name_or_num_classes in ('imagenet',):
        num_classes = 1000
    elif dataset_name_or_num_classes in ('cifar100',):
        num_classes = 100
    elif dataset_name_or_num_classes in ('cifar10', 'dummy'):
        num_classes = 10
    else:
        raise Exception(f'Unsupported dataset: {dataset_name_or_num_classes}')

    # Make a default parameters for building the model.
    model_params = dict(
        normalization=nn.BatchNorm2d,
        activation=nn.ReLU,
        semodule=False,
        semodule_reduction=8,
        semodule_divisor=1,
        semodule_activation=nn.ReLU,
        semodule_sigmoid=nn.Sigmoid,
        gate_reduction=8,
        gate_normalization=nn.BatchNorm2d,
        gate_activation=nn.ReLU,
        gate_connections=4,
        dropout_prob=0.0,
        shakedrop_prob=0.0,
        stochdepth_prob=0.0,
        signalaugment=0.0,
        pretrained=False,
    )

    # Apply the model speficied parameters.
    model_params.update(PARAMETERS[model_name])
    model_params.update(num_classes=num_classes)

    # Apply parameters specified as `kwargs`.
    # The value will be converted to a valid type if the value is a string.
    # Unknown parameters in `kwargs` are ignored.
    for key, value in kwargs.items():
        if key not in model_params:
            continue

        if not isinstance(value, str):
            model_params[key] = value
            continue

        old_value = model_params[key]

        if isinstance(old_value, bool):
            model_params[key] = (value.lower() in ('yes', 'true', 'on', '1'))
        elif isinstance(old_value, (str, int, float)):
            model_params[key] = type(old_value)(value)
        elif isinstance(old_value, type):
            model_params[key] = eval(value)
        else:
            raise Exception(
                f'Parameter `{key}` can not be converted from a string')

    # Build a model.
    return Model(**model_params)  # type:ignore


def create_model_from_checkpoint(checkpoint: Dict[str, Any]) -> 'Model':
    '''Build a model with the parameters loaded from a checkpoint file.
    Args:
        checkpoint: Parameters loaded from a checkpoint file.
    Returns:
        A model object.
    '''
    model_name = checkpoint['config']['model']
    dataset_name = checkpoint['config']['dataset']
    parameters = checkpoint['config']['parameters']
    parameters['pretrained'] = False

    state_dict = checkpoint['state_dict']
    state_dict = {k[6:]: v for k, v in state_dict.items() if k[:6] == 'model.'}

    model = create_model(model_name, dataset_name, **parameters)
    model.load_state_dict(state_dict)

    return model
