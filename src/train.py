import argparse
import functools
import gc
import json
import logging
import os
import pathlib
import re
import subprocess
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from weakref import ReferenceType

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as pll
import pytorch_lightning.plugins as plp
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from datasets import (create_train_dataloader, create_valid_dataloader,
                      prepare_dataset)
from models import create_model
from models.modules import DropBlock
from utils import (Config, create_optimizer, create_scheduler, setup_logging,
                   setup_progressbar, setup_random_seed)

try:
    import torch_xla.core.xla_model as xm  # type:ignore
except BaseException:
    xm = None

LOGGER = logging.getLogger(__name__)


def _parse_params_arg(arg: str) -> Dict[str, str]:
    params = [p.split(':') for p in arg.split(',')]
    return {k.strip(): v.strip() for k, v in params}


def _parse_gpus_arg(arg: str) -> List[int]:
    return list(map(int, arg.split(',')))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a model',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', type=str, help='Configuration file.')
    parser.add_argument('output', type=str, help='Output directory.')
    parser.add_argument(
        '--parameters', type=_parse_params_arg, default={},
        help='Additional parameter for training the model.')
    parser.add_argument(
        '--data', type=str, default=None,
        help='Data directory.')
    parser.add_argument(
        '--precision', type=int, default=32, choices=[16, 32],
        help='Precision of training.')
    parser.add_argument(
        '--gpus', type=_parse_gpus_arg, default=None,
        help='GPU IDs used for training.')
    parser.add_argument(
        '--tpus', type=int, default=None,
        help='Number of TPU cores used for training.')
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of workers for data loaders.')
    parser.add_argument(
        '--accum', type=int, default=1,
        help='Number of accumlated batches.')
    parser.add_argument(
        '--timestamp', action='store_true', default=False,
        help='Timestamp file is created.')
    parser.add_argument(
        '--wandb', action='store_true', default=False,
        help='W&B logger is used.')
    parser.add_argument(
        '--neptune', action='store_true', default=False,
        help='Neptune logger is used.')
    parser.add_argument(
        '--gsbacket', type=str, default=None,
        help='Backet URL of google cloud strage.')
    parser.add_argument(
        '--progress-step', type=int, default=10,
        help='Number of steps among updating progress bar.')
    parser.add_argument(
        '--log-step', type=int, default=10,
        help='Number of steps among updating logs.')
    parser.add_argument(
        '--seed', type=int, default=2021,
        help='Random seed.')
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Run with debug mode.')

    return parser.parse_args()


def _run_gsutil(args: List[str]) -> None:
    command = ['gsutil'] + args
    LOGGER.debug('run gsutil: cmd=[%s]', ' '.join(command))

    pipe = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = pipe.communicate()
    retval = pipe.wait()
    LOGGER.debug('run gsutil: out=[%s]', stdout.decode('utf-8'))
    LOGGER.debug('run gsutil: err=[%s]', stderr.decode('utf-8'))

    if retval != 0:
        raise RuntimeError(stderr.decode('utf-8'))


def prepare_gsbacket(gsbacket: str, output_path: pathlib.Path) -> None:
    os.makedirs(output_path, exist_ok=True)

    timestamp_file = output_path / 'timestamp.txt'
    timestamp_file.write_text('Time: {}'.format(
        time.strftime('%Y/%m/%d %H:%M:%S %Z')))

    _run_gsutil(['cp', str(timestamp_file), f'{gsbacket}/timestamp.txt'])
    _run_gsutil(['rsync', '-r', '-d', gsbacket, str(output_path)])


def update_gsbacket(gsbacket: str, output_path: pathlib.Path) -> None:
    _run_gsutil(['rsync', '-r', '-d', str(output_path), gsbacket])


class GSBacketUpdater(plc.Callback):
    def __init__(
        self,
        output_path: pathlib.Path,
        gsbacket: str,
    ) -> None:
        self.output_path = output_path
        self.gsbacket = gsbacket

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.is_global_zero:
            update_gsbacket(self.gsbacket, self.output_path)


class TimestampWriter(plc.Callback):
    def __init__(self, output_path: pathlib.Path) -> None:
        self.output_path = output_path

    def on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if not trainer.is_global_zero:
            return

        try:
            from psutil import virtual_memory
            mem = virtual_memory()
            memory_usage = '{:,d} ({:.1f}%)'.format(mem.used, mem.percent)
        except BaseException:
            memory_usage = ''

        timestamp = time.strftime('%Y/%m/%d %H:%M:%S %Z')

        if (isinstance(pl_module, Model)
                and len(pl_module.train_logs) != 0
                and len(pl_module.valid_logs) != 0):
            train_log = pl_module.train_logs[-1]
            valid_log = pl_module.valid_logs[-1]

            learning_rate = '{:.6f}'.format(train_log['learning_rate'])
            train_result = 'loss={:.4f}, accuracy1={:.4f}'.format(
                train_log['loss'], train_log['accuracy1'])
            valid_result = 'loss={:.4f}, accuracy1={:.4f}'.format(
                valid_log['loss'], valid_log['accuracy1'])
        else:
            learning_rate = ''
            train_result = ''
            valid_result = ''

        texts = [
            f'Time: {timestamp}',
            f'Memory: {memory_usage} used',
            f'Epoch: {trainer.current_epoch}',
            f'LearningRate: {learning_rate}',
            f'Train: {train_result}',
            f'Valid: {valid_result}']
        self.output_path.write_text('\n'.join(texts))


class Model(pl.LightningModule):
    def __init__(self, config: Config, parameters: Dict[str, str]) -> None:
        super().__init__()
        self.config = config
        self.params = config.parameters.copy()
        self.params.update(parameters)

        self.model = create_model(
            config.model, config.dataset, **self.params)
        self.optimizer_states_dict: Optional[Dict] = None
        self.scheduler_states_dict: Optional[Dict] = None
        self.train_logs: List[Dict[str, Any]] = []
        self.valid_logs: List[Dict[str, Any]] = []
        LOGGER.debug(self.model)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Any:
        optimizer = create_optimizer(self.model, **self.config.parameters)
        scheduler = create_scheduler(optimizer, **self.config.parameters)

        if self.optimizer_states_dict is not None:
            optimizer.load_state_dict(self.optimizer_states_dict)

        if self.scheduler_states_dict is not None:
            scheduler.load_state_dict(self.scheduler_states_dict)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, probs = batch

        logits = self(images)
        targets = probs.argmax(dim=1)

        loss = (-1 * F.log_softmax(logits, dim=1) * probs).sum(dim=1).mean()
        accuracy1 = torchmetrics.functional.accuracy(logits, targets, top_k=1)
        accuracy5 = torchmetrics.functional.accuracy(logits, targets, top_k=5)
        learning_rate = self.optimizers().param_groups[0]['lr']  # type:ignore

        return {
            'loss': loss,
            'accuracy1': float(accuracy1),
            'accuracy5': float(accuracy5),
            'learning_rate': float(learning_rate),
            'size': int(images.shape[0]),
        }

    def training_epoch_end(self, training_step_outputs):
        loss = 0.0
        accuracy1 = 0.0
        accuracy5 = 0.0
        learning_rate = 0.0
        size = 0

        for outputs in training_step_outputs:
            loss += float(outputs['loss'] * outputs['size'])
            accuracy1 += float(outputs['accuracy1'] * outputs['size'])
            accuracy5 += float(outputs['accuracy5'] * outputs['size'])
            learning_rate += float(outputs['learning_rate'] * outputs['size'])
            size += int(outputs['size'])

        loss = loss / size
        accuracy1 = accuracy1 / size
        accuracy5 = accuracy5 / size
        learning_rate = learning_rate / size

        self.log('train/loss', loss)
        self.log('train/accuracy1', accuracy1)
        self.log('train/accuracy5', accuracy5)
        self.log('train/learning_rate', learning_rate, prog_bar=True)

        self.train_logs.append({
            'loss': loss,
            'accuracy1': accuracy1,
            'accuracy5': accuracy5,
            'learning_rate': learning_rate,
            'datetime': time.strftime('%Y/%m/%d %H:%M:%S %Z'),
        })

        gc.collect()

    def validation_step(self, batch, batch_idx):
        images, probs = batch

        logits = self(images)
        targets = probs.argmax(dim=1)

        loss = (-1 * F.log_softmax(logits, dim=1) * probs).sum(dim=1).mean()
        accuracy1 = torchmetrics.functional.accuracy(logits, targets, top_k=1)
        accuracy5 = torchmetrics.functional.accuracy(logits, targets, top_k=5)

        return {
            'loss': float(loss),
            'accuracy1': float(accuracy1),
            'accuracy5': float(accuracy5),
            'size': int(images.shape[0]),
        }

    def validation_epoch_end(self, validation_step_outputs):
        loss = 0.0
        accuracy1 = 0.0
        accuracy5 = 0.0
        size = 0

        for outputs in validation_step_outputs:
            loss += float(outputs['loss'] * outputs['size'])
            accuracy1 += float(outputs['accuracy1'] * outputs['size'])
            accuracy5 += float(outputs['accuracy5'] * outputs['size'])
            size += int(outputs['size'])

        loss = loss / size
        accuracy1 = accuracy1 / size
        accuracy5 = accuracy5 / size

        self.log('valid/loss', loss)
        self.log('valid/accuracy1', accuracy1, prog_bar=True)
        self.log('valid/accuracy5', accuracy5)

        self.valid_logs.append({
            'loss': loss,
            'accuracy1': accuracy1,
            'accuracy5': accuracy5,
            'datetime': time.strftime('%Y/%m/%d %H:%M:%S %Z'),
        })
        gc.collect()

    def on_train_epoch_start(self) -> None:
        if (self.config.parameters['dropblock_prob'] > 0.0
                and self.config.parameters['dropblock_size'] > 0):
            progress = (self.current_epoch + 1) / self.config.parameters['train_epoch']
            dropblock_prob = self.config.parameters['dropblock_prob'] * progress

            for module in self.model.modules():
                if isinstance(module, DropBlock):
                    module.drop_prob = dropblock_prob
                    module.block_size = self.config.parameters['dropblock_size']

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.optimizer_states_dict = {
            k: {} if k == 'state' else v
            for k, v in checkpoint['optimizer_states'][0].items()}
        self.scheduler_states_dict = checkpoint['lr_schedulers'][0]
        self.train_logs = checkpoint['train_logs']
        self.valid_logs = checkpoint['valid_logs']

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['config'] = {
            'model': self.config.model,
            'dataset': self.config.dataset,
            'parameters': self.params,
        }
        checkpoint['train_logs'] = self.train_logs
        checkpoint['valid_logs'] = self.valid_logs


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Config,
        data_path: str,
        num_workers: int,
        num_accum: int = 1,
        gpus: Union[int, List[int], None] = None,
        tpus: Union[int, List[int], None] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.data_path = data_path
        self.num_workers = num_workers
        self.num_accum = num_accum
        self.gpus = gpus

        if gpus is not None:
            self.num_cores = len(gpus) if isinstance(gpus, list) else gpus
        elif tpus is not None:
            self.num_cores = len(tpus) if isinstance(tpus, list) else tpus
        else:
            self.num_cores = 1

    def train_dataloader(self) -> Iterable:  # type:ignore
        batch = self.config.parameters['train_batch']
        return create_train_dataloader(
            dataset_name=self.config.dataset,
            data_path=self.data_path,
            batch_size=batch // self.num_accum,
            num_workers=self.num_workers,
            num_cores=self.num_cores,
            pin_memory=self.gpus is not None,
            **self.config.parameters)

    def val_dataloader(self) -> Iterable:  # type:ignore
        batch = self.config.parameters['train_batch']
        return create_valid_dataloader(
            dataset_name=self.config.dataset,
            data_path=self.data_path,
            batch_size=batch // self.num_accum,
            num_workers=self.num_workers,
            num_cores=self.num_cores,
            pin_memory=self.gpus is not None,
            **self.config.parameters)


class WandbLogger(pll.WandbLogger):
    def __init__(
        self,
        save_path: pathlib.Path,
        wandb_name: str,
        **kwargs,
    ) -> None:
        project_name = wandb_name.split('/', maxsplit=1)
        kwargs['project'] = project_name[0]
        kwargs['name'] = project_name[-1]
        kwargs['save_dir'] = str(save_path)

        if (save_path / 'wandb.json').is_file():
            with open(save_path / 'wandb.json', 'r') as reader:
                kwargs['id'] = json.load(reader)['id']

        super().__init__(**kwargs)
        self._save_path = save_path

    def after_save_checkpoint(
        self,
        checkpoint_callback: 'ReferenceType[plc.ModelCheckpoint]',
    ) -> None:
        with open(self._save_path / 'wandb.json', 'w') as writer:
            json.dump({'id': self.version}, writer)
        super().after_save_checkpoint(checkpoint_callback)


class NeptuneLogger(pll.NeptuneLogger):
    def __init__(
        self,
        save_path: pathlib.Path,
        neptune_name: str,
        **kwargs,
    ) -> None:
        project_name = neptune_name.split('/', maxsplit=1)
        user_name, api_token = self._load_config()
        kwargs['api_key'] = re.sub(r'\s+', '', api_token)
        kwargs['project'] = f'{user_name}/{project_name[0]}'
        kwargs['name'] = project_name[-1]
        kwargs['fail_on_exception'] = False

        super().__init__(**kwargs)
        self._save_path = save_path

        if (save_path / 'neptune.json').is_file():
            with open(save_path / 'neptune.json', 'r') as reader:
                self._run_short_id = json.load(reader)['id']

    def _load_config(self):
        config_path = pathlib.Path(os.environ['HOME']) / '.neptune.json'

        if config_path.is_file():
            with open(config_path, 'r') as reader:
                config = json.load(reader)
        else:
            print('Neptune config file is not found.')
            user_name = input('Neptune user name: ')
            api_token = input('Neptune API token: ')
            config = {'user_name': user_name, 'api_token': api_token}
            with open(config_path, 'w') as writer:
                json.dump(config, writer)

        return config['user_name'], config['api_token']

    def after_save_checkpoint(
        self,
        checkpoint_callback: 'ReferenceType[plc.ModelCheckpoint]',
    ) -> None:
        with open(self._save_path / 'neptune.json', 'w') as writer:
            json.dump({'id': self.version}, writer)
        super().after_save_checkpoint(checkpoint_callback)


class TPUSpawnPlugin(plp.TPUSpawnPlugin):
    def new_process(self, *args, **kwargs):
        os.environ['XLA_USE_BF16'] = str(1)
        super().new_process(*args, **kwargs)


class TPUPrecisionPlugin(plp.TPUPrecisionPlugin):
    def optimizer_step(
        self,
        model: Union[pl.LightningModule, nn.Module],
        optimizer: optim.Optimizer,
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if isinstance(model, pl.LightningModule):
            closure = functools.partial(
                self._wrap_closure, model, optimizer, optimizer_idx, closure)

        def optimizer_closure():
            result = closure()
            xm.reduce_gradients(optimizer)
            return result

        optimizer.step(closure=optimizer_closure)


class TPUBf16PrecisionPlugin(TPUPrecisionPlugin):
    precision: str = "bf16"

    def connect(
        self,
        model: nn.Module,
        optimizers: List[optim.Optimizer],
        lr_schedulers: List[Any],
    ) -> Tuple[nn.Module, List[optim.Optimizer], List[Any]]:
        os.environ["XLA_USE_BF16"] = "1"
        return super().connect(model=model, optimizers=optimizers, lr_schedulers=lr_schedulers)

    def teardown(self) -> None:
        os.environ.pop("XLA_USE_BF16", None)


def create_trainer(
    config: Config,
    output_path: pathlib.Path,
    num_accum: int,
    precision: int,
    gpus: Union[int, List[int], None],
    tpus: Union[int, List[int], None],
    timestamp: Optional[str],
    wandb: Optional[str],
    neptune: Optional[str],
    gsbacket: Optional[str],
    progress_step: int,
    log_step: int,
) -> pl.Trainer:
    # loggers
    loggers: List[pll.LightningLoggerBase] = [
        pll.TensorBoardLogger(
            save_dir=str(output_path / 'tensorboard'), name='', version=0)
    ]

    if wandb is not None:
        loggers.append(
            WandbLogger(save_path=output_path, wandb_name=wandb))

    if neptune is not None:
        loggers.append(
            NeptuneLogger(save_path=output_path, neptune_name=neptune))

    # checkpoint
    callbacks: List[plc.Callback] = [
        plc.ModelCheckpoint(
            dirpath=str(output_path / 'checkpoint'),
            filename='epoch_{epoch:03d}', auto_insert_metric_name=False,
            save_last=True, save_top_k=1, monitor='valid/accuracy1', mode='max'),
        plc.TQDMProgressBar(refresh_rate=progress_step),
    ]

    if timestamp is not None:
        callbacks.append(TimestampWriter(output_path / timestamp))

    if gsbacket is not None:
        callbacks.append(GSBacketUpdater(output_path, gsbacket))

    # parameters
    parameters = {
        'default_root_dir': str(output_path),
        'max_epochs': config.parameters['train_epoch'],
        'accumulate_grad_batches': num_accum,
        'logger': loggers,
        'callbacks': callbacks,
        'log_every_n_steps': log_step,
        'num_sanity_val_steps': 0,
    }

    # gpu/tpu
    if gpus is not None:
        gpu_cores = len(gpus) if isinstance(gpus, list) else gpus
        parameters['gpus'] = gpus
        parameters['deterministic'] = True
        parameters['benchmark'] = True
        parameters['precision'] = precision
        parameters['strategy'] = plp.DDPPlugin(
            find_unused_parameters=gpu_cores > 1)
    elif tpus is not None:
        tpu_cores = len(tpus) if isinstance(tpus, list) else tpus
        parameters['tpu_cores'] = tpus
        parameters['strategy'] = plp.TPUSpawnPlugin(
            find_unused_parameters=tpu_cores > 1)
        if precision == 16:
            parameters['precision'] = 'bf16'
            parameters['plugins'] = [TPUBf16PrecisionPlugin()]
        else:
            parameters['precision'] = precision
            parameters['plugins'] = [TPUPrecisionPlugin()]

    # gradient clipping
    if config.parameters['gradclip_value'] > 0:
        parameters['gradient_clip_algorithm'] = 'value'
        parameters['gradient_clip_val'] = config.parameters['gradclip_value']
    elif config.parameters['gradclip_norm'] > 0:
        parameters['gradient_clip_algorithm'] = 'norm'
        parameters['gradient_clip_val'] = config.parameters['gradclip_value']

    return pl.Trainer(**parameters)


def train_model(
    config: Config,
    parameters: Dict[str, str],
    output_path: pathlib.Path,
    data_path: str,
    num_workers: int = 1,
    num_accum: int = 1,
    precision: int = 32,
    gpus: Union[int, List[int], None] = None,
    tpus: Union[int, List[int], None] = None,
    timestamp: Optional[str] = None,
    wandb: Optional[str] = None,
    neptune: Optional[str] = None,
    gsbacket: Optional[str] = None,
    progress_step: int = 10,
    log_step: int = 10,
) -> None:
    # prepare the dataset
    prepare_dataset(dataset_name=config.dataset, data_path=data_path)

    # make an output directory
    os.makedirs(output_path / 'checkpoint', exist_ok=True)

    # make training modules
    model = Model(config, parameters)
    datamodule = DataModule(
        config, data_path, num_workers, num_accum, gpus, tpus)

    # prepare an output directory in the google strage backet
    if gsbacket is not None:
        prepare_gsbacket(gsbacket, output_path)

    # resume setting
    output_file = output_path / 'checkpoint' / 'last.ckpt'
    if output_file.is_file():
        checkpoint_path = str(output_file)
    else:
        checkpoint_path = None

    # training
    create_trainer(
        config, output_path, num_accum, precision, gpus, tpus,
        timestamp, wandb, neptune, gsbacket, progress_step, log_step,
    ).fit(
        model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

    # update the google strage backset
    if gsbacket is not None:
        update_gsbacket(gsbacket, output_path)


def main() -> None:
    args = parse_args()
    setup_logging(debug=args.debug)
    setup_random_seed(seed=args.seed)
    setup_progressbar(enabled=args.progress_step != 0)

    config = Config(args.config)
    log_name = f'{config.dataset}/{config.model}'

    if args.data is None:
        root_path = pathlib.Path(__file__).parent.parent
        data_path = str(root_path / 'data' / config.dataset)
    else:
        data_path = args.data

    LOGGER.info('Output: %s', args.output)
    LOGGER.info('Config:\n%s', config)

    if not args.debug:
        torch.autograd.profiler.profile(enabled=False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
        torch.autograd.set_detect_anomaly(mode=False)

    train_model(
        config=config,
        parameters=args.parameters,
        output_path=pathlib.Path(args.output),
        data_path=data_path,
        num_workers=args.workers,
        num_accum=args.accum,
        precision=args.precision,
        gpus=args.gpus,
        tpus=args.tpus,
        timestamp='timestamp.txt' if args.timestamp else None,
        wandb=log_name if args.wandb else None,
        neptune=log_name if args.neptune else None,
        gsbacket=args.gsbacket,
        progress_step=args.progress_step,
        log_step=args.log_step)


if __name__ == '__main__':
    main()
