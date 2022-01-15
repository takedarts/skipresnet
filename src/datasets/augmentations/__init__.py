from .autoaugment import (AutoAugmentCIFAR10, AutoAugmentImageNet,
                          AutoAugmentSVHN)
from .cutmixup import (apply_cutmix, apply_cutmixup_to_dataset,
                       apply_cutmixup_to_stream, apply_labelsmooth,
                       apply_mixup)
from .lighting import Lighting
from .randaugment import RandAugment
