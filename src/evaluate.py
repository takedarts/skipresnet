import argparse
import pathlib
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import tqdm

from datasets import create_valid_dataloader, setup_dataloader
from models import create_model_from_checkpoint
from utils import Config, setup_logging

parser = argparse.ArgumentParser(
    description='Evaluate the model performance.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('checkpoint', type=str, help='Checkpoint file.')
parser.add_argument('--dataset', type=str, default=None, help='Dataset name for the evaluation.')
parser.add_argument('--image-size', type=int, default=224, help='Image size.')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
parser.add_argument('--data', type=str, default=None, help='Data directory.')
parser.add_argument('--gpu', type=int, default=None, help='GPU ID.')
parser.add_argument('--no-progress', action='store_true', default=False, help='Without progress bar.')
parser.add_argument('--debug', action='store_true', default=False, help='Run with debug mode.')


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: str,
    progress_bar: bool,
) -> Tuple[float, float, float]:
    loss_total = 0.0
    acc1_total = 0.0
    acc5_total = 0.0
    count = 0

    model = model.to(device)
    model.eval()

    if progress_bar:
        pbar = tqdm.tqdm
    else:
        def pbar(x):
            return x

    for images, probs in pbar(loader):
        images = images.to(device)
        probs = probs.to(device)

        logits = model(images)
        targets = probs.argmax(dim=1)

        loss = (-1 * F.log_softmax(logits, dim=1) * probs).sum(dim=1).mean()
        acc1 = torchmetrics.functional.accuracy(logits, targets, top_k=1)
        acc5 = torchmetrics.functional.accuracy(logits, targets, top_k=5)

        loss_total += float(loss) * len(images)
        acc1_total += float(acc1) * len(images)
        acc5_total += float(acc5) * len(images)
        count += len(images)

    return loss_total / count, acc1_total / count, acc5_total / count


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.debug)

    # check parameters
    checkpoint = torch.load(args.checkpoint, map_location=lambda s, _: s)
    config = Config.create_from_checkpoint(checkpoint)

    if args.dataset is None:
        args.dataset = config.dataset

    if args.data is None:
        root_path = pathlib.Path(__file__).parent.parent
        args.data = str(root_path / 'data' / args.dataset)

    if args.image_size is not None:
        config.parameters['valid_crop'] = args.image_size

    if args.gpu is not None:
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    # prepare dataset
    setup_dataloader(dataset_name=args.dataset, data_path=args.data)
    loader = create_valid_dataloader(
        dataset_name=args.dataset, data_path=args.data, batch_size=args.batch_size,
        num_workers=0, num_cores=1, pin_memory=False, **config.parameters)

    # create model
    model = create_model_from_checkpoint(checkpoint)

    # evaluate
    loss, accuracy1, accuracy5 = evaluate(
        model, loader, device=device, progress_bar=not args.no_progress)

    print(f'dataset = {config.dataset}')
    print(f'model = {config.model}')
    print(f'loss = {loss:.6f}')
    print(f'accuracy1 = {accuracy1:.6f}')
    print(f'accuracy5 = {accuracy5:.6f}')


if __name__ == '__main__':
    main()
