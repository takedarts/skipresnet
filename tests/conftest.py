import pathlib
import sys

SRC_PATH = str((pathlib.Path(__file__).parent.parent / 'src').absolute())
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils

utils.setup_logging(True)
