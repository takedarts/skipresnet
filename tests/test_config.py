import io
import pytest
from utils import Config

CONFIG1 = [
    ('model', 'resnet-110'),
    ('dataset', 'cifar10'),
    ('train_batch', 64),
    ('train_crop', 32),
    ('train_epoch', 600),
    ('train_warmup', 5),
    ('train_optim', 'sgd'),
    ('train_lr', 0.05),
    ('train_wdecay', 0.0005),
    ('train_bdecay', False),
    ('train_schedule', 'cosine'),
    ('autoaugment', True),
]

ERROR1 = [
    ('model', 'resnet-110'),
    ('dataset', 'cifar10'),
    ('unknown', 0.0),
]


def test_config() -> None:
    text = '\n'.join(f'{k}: {str(v).lower()}' for k, v in CONFIG1)
    config = Config(io.StringIO(text))

    assert config.model == CONFIG1[0][1]
    assert config.dataset == CONFIG1[1][1]
    for k, v in CONFIG1[2:]:
        assert config.parameters[k] == v


def test_error() -> None:
    text = '\n'.join(f'{k}: {str(v).lower()}' for k, v in ERROR1)
    with pytest.raises(Exception):
        Config(io.StringIO(text))
