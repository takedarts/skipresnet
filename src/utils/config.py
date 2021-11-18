from typing import Any, Dict, List, Tuple

PARAMETERS: List[Tuple[str, Any, str]] = [
    ('train_batch', 64, 'Batch size at training.'),
    ('train_crop', 224, 'Input image size at training.'),
    ('train_epoch', 300, 'Number of epochs at training.'),
    ('train_warmup', 5, 'Number of epochs for warmup at training.'),
    ('train_optim', 'sgd', 'Optimizer at training (sgd/rmsprop/adamw/sam).'),
    ('train_lr', 0.025, 'Initial learning rate at training'),
    ('train_wdecay', 0.0001, 'Weight decay (L2 penalty) at training.'),
    ('train_bdecay', False, 'Adaptation of decay (L2 penalty) to bias parameters.'),
    ('train_schedule', 'cosine', 'Learning rate schedule at training (cosine/exponential).'),
    ('valid_crop', 224, 'Input image size at validation.'),
    ('cutmix_alpha', 1.0, 'Distribution parameter Alpha of CutMix at training.'),
    ('cutmix_prob', 0.0, 'Probability of CutMix at training.'),
    ('mixup_alpha', 1.0, 'Distribution parameter Alpha of Mixup at training.'),
    ('mixup_prob', 0.0, 'Probability of Mixup at training.'),
    ('randomerasing_prob', 0.0, 'Probability of RandomErasing.'),
    ('randomerasing_type', 'random', 'Type of RandomErasing (random/zero).'),
    ('randaugment_num', 0, 'Number of transformations in RandAugment.'),
    ('randaugment_mag', 0, 'Magnitude of transformations in RandAugment.'),
    ('autoaugment', False, 'Use auto augumentation.'),
    ('labelsmooth', 0.0, 'Factor "k" of label smoothing.'),
    ('gradclip_value', 0.0, 'Threshold of gradient value clipping.'),
    ('gradclip_norm', 0.0, 'Threshold of gradient norm clipping.'),
    ('dropout_prob', 0.0, 'Probability of dropout.'),
    ('shakedrop_prob', 0.0, 'Probability of shake-drop.'),
    ('dropblock_prob', 0.0, 'Drop probability of DropBlock.'),
    ('dropblock_size', 7, 'Drop block size of DropBlock.'),
    ('stochdepth_prob', 0.0, 'Drop probability of stochastic depth.'),
    ('signalaugment', 0.0, 'Standard deviation of signal augmentation.'),
    ('semodule_reduction', 16, 'Reduction ratio of "Squeeze and Excitation" modules.'),
    ('gate_reduction', 8, 'reduction rate of gate modules in DenseResNets or SkipResNets.'),
    ('dense_connections', 4, 'number of connections of gate modules in DenseResNets.'),
    ('skip_connections', 16, 'number of connections of gate modules in SkipResNets.'),
]


class Config(object):
    def __init__(self, file_or_stream: Any = None) -> None:
        self.model: str = ''
        self.dataset: str = ''
        self.parameters: Dict[str, Any] = {k: v for k, v, _ in PARAMETERS}

        if file_or_stream is None:
            return
        elif hasattr(file_or_stream, 'read'):
            self._load(file_or_stream)
        else:
            with open(str(file_or_stream), 'r') as reader:
                self._load(reader)

    def _load(self, reader: Any) -> None:
        for line in reader:
            tokens = line.split('#', maxsplit=1)[0].split(':', maxsplit=1)
            if len(tokens) != 2:
                continue

            key, value = [v.strip() for v in tokens]
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

    def __str__(self) -> str:
        helps = {k: d for k, _, d in PARAMETERS}
        texts = [
            f'model={self.model}  # Model name.',
            f'dataset={self.dataset}  # Dataset name.']
        texts.extend(
            f'{k}={v}  # {helps[k]}' for k, v in self.parameters.items())
        return '\n'.join(texts)
