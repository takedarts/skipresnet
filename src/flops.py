import argparse
import logging
import warnings
import torch
from models import create_model
from utils import count_operations, setup_logging
import os
os.environ['PYTORCH_JIT'] = '0'


LOGGER = logging.getLogger(__name__)

warnings.filterwarnings(
    'ignore', category=UserWarning,
    message='This model contains a squeeze operation on dimension 1.')

parser = argparse.ArgumentParser(
    description='Show FLOPs of the specified model.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('model', type=str, help='Model name.')
parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes.')
parser.add_argument('--image-size', type=int, default=224, help='Size of input image.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')


def format_with_scale(num: int) -> str:
    scale_chars = ('', 'K', 'M', 'G', 'T', 'P')
    scale_index = (len(str(num)) - 1) // 3
    after_point = 3 - ((len(str(num)) - 1) % 3)
    return f'{{:.{after_point}f}}{scale_chars[scale_index]}'.format(
        num / (1_000 ** scale_index))


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.debug)

    # check jit setting
    if 'PYTORCH_JIT' not in os.environ or int(os.environ['PYTORCH_JIT']) != 0:
        LOGGER.warning(
            'This process might fail because the torch script is enabled.'
            + ' Set an environment variable `PYTORCH_JIT=0` to disable the torch script.')

    # model
    model = create_model(args.model, args.num_classes)

    # flops
    image = torch.randn([1, 3, args.image_size, args.image_size], dtype=torch.float32)
    flops = count_operations(model, (image,))

    # parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'parameters: {params:,d} ({format_with_scale(params)})')
    print(f'flops: {flops:,d} ({format_with_scale(flops)})')


if __name__ == '__main__':
    main()
