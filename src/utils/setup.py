import logging
import os
import random
import sys
import warnings

import numpy as np
import torch.cuda


def setup_logging(debug: bool) -> None:
    if len(logging.getLogger().handlers) != 0:
        return

    formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] '
                                  '%(message)s (%(module)s.%(funcName)s:%(lineno)s)',
                                  '%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

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
