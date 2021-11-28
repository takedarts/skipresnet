import logging
import os
import random
import sys
import warnings

import numpy as np
import torch.cuda

LOGGING_HANDLER = logging.StreamHandler(stream=sys.stdout)
LOGGING_HANDLER.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)-5.5s] '
    '%(message)s (%(module)s.%(funcName)s:%(lineno)s)',
    '%Y-%m-%d %H:%M:%S'))


def setup_logging(debug: bool) -> None:
    if LOGGING_HANDLER not in logging.getLogger().handlers:
        logging.getLogger().addHandler(LOGGING_HANDLER)

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger('PIL').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    warnings.filterwarnings('ignore', message='Corrupt EXIF data.')


def setup_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
