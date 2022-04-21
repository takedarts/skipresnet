from typing import Any, Dict, List, Tuple

PARAMETERS: List[Tuple[str, Any, str]] = [
    ('train_batch', 64, 'Batch size.'),
    ('train_crop', 224, 'Input image size.'),
    ('train_epoch', 300, 'Number of epochs.'),
    ('train_warmup', 5, 'Number of epochs for warmup.'),
    ('train_optim', 'sgd', 'Optimizer name (sgd/adamw/rmsprop/rmsproptf/sam).'),
    ('train_lr', 0.025, 'Initial learning rate.'),
    ('train_layerlr_ratio', 1.0, 'Input/Output LR ratio of layer-wise learning rate decay.'),
    ('train_momentum', 0.9, 'Momentum factor of optimizer.'),
    ('train_eps', 1e-08, 'Machine epsilon for optimizer stabilization.'),
    ('train_alpha', 0.99, 'Smoothing constant of optimizer.'),
    ('train_wdecay', 0.0001, 'Weight decay (L2 penalty).'),
    ('train_bdecay', False, 'Weight decay (L2 penalty) is adapted to bias parameters.'),
    ('train_schedule', 'cosine', 'Learning rate schedule (cosine/exponential).'),
    ('train_lastlr', 0.0, 'Learning rate at last epoch (default is used if 0.0).'),
    ('valid_crop', 224, 'Input image size at validation.'),
    ('cutmix_alpha', 1.0, 'Distribution parameter Alpha of CutMix.'),
    ('cutmix_prob', 0.0, 'Probability of CutMix.'),
    ('mixup_alpha', 1.0, 'Distribution parameter Alpha of Mixup.'),
    ('mixup_prob', 0.0, 'Probability of Mixup.'),
    ('randomerasing_prob', 0.0, 'Probability of RandomErasing.'),
    ('randomerasing_type', 'random', 'Type of RandomErasing (random/zero).'),
    ('randaugment_prob', 0.0, 'Probability of RandAugment.'),
    ('randaugment_num', 0, 'Number of RandAugment transformations.'),
    ('randaugment_mag', 0, 'Magnitude of RandAugment transformations.'),
    ('randaugment_std', 0.0, 'Standard deviation of RandAugment noise.'),
    ('autoaugment', False, 'Auto augumentation is used.'),
    ('labelsmooth', 0.0, 'Factor "k" of label smoothing.'),
    ('gradclip_value', 0.0, 'Threshold of gradient value clipping.'),
    ('gradclip_norm', 0.0, 'Threshold of gradient norm clipping.'),
    ('dropout_prob', 0.0, 'Probability of dropout.'),
    ('shakedrop_prob', 0.0, 'Probability of shake-drop.'),
    ('dropblock_prob', 0.0, 'Drop probability of DropBlock.'),
    ('dropblock_size', 7, 'Drop block size of DropBlock.'),
    ('stochdepth_prob', 0.0, 'Drop probability of stochastic depth.'),
    ('signalaugment', 0.0, 'Standard deviation of signal augmentation.'),
    ('pretrained', False, 'Pretrained weights from pytorch-image-models (timm) is loaded.'),
]


class Config(object):
    @staticmethod
    def create_from_checkpoint(checkpoint: Dict[str, Any]) -> 'Config':
        config = Config()
        config.model = checkpoint['config']['model']
        config.dataset = checkpoint['config']['dataset']
        config.parameters.update(checkpoint['config']['parameters'])
        return config

    def __init__(self, file_or_stream: Any = None) -> None:
        self.model = ''
        self.dataset = ''
        self.parameters = {k: v for k, v, _ in PARAMETERS}

        if file_or_stream is None:
            return
        elif hasattr(file_or_stream, 'read'):
            self._read(file_or_stream)
        else:
            with open(str(file_or_stream), 'r') as reader:
                self._read(reader)

    def _read(self, reader: Any) -> None:
        for line in reader:
            tokens = line.split('#', maxsplit=1)[0].split(':', maxsplit=1)
            if len(tokens) == 2:
                self._update(*[v.strip() for v in tokens])

    def _update(self, key: str, value: str) -> None:
        if key == 'model':
            self.model = value
        elif key == 'dataset':
            self.dataset = value
        elif key in self.parameters:
            self._update_parameter(key, value)
        else:
            raise Exception(f'unknown parameter name: {key}')

    def _update_parameter(self, key: str, value: str) -> None:
        old_value = self.parameters[key]

        if isinstance(old_value, bool):
            self.parameters[key] = (value.lower() in ('yes', 'true', 'on', '1'))
        else:
            self.parameters[key] = type(old_value)(value)

    def update(self, key: str, value: Any) -> None:
        self._update(key, str(value))

    def __str__(self) -> str:
        helps = {k: d for k, _, d in PARAMETERS}
        texts = [
            f'model={self.model}  # Model name.',
            f'dataset={self.dataset}  # Dataset name.']
        texts.extend(
            f'{k}={v}  # {helps[k]}' for k, v in self.parameters.items())
        return '\n'.join(texts)
