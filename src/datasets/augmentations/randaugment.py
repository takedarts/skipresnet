import timm.data.auto_augment


class RandAugment(timm.data.auto_augment.RandAugment):
    def __init__(
        self,
        num_layers: int,
        magnitude: int,
        prob: float,
        mstd: float,
    ) -> None:
        hparams = timm.data.auto_augment._HPARAMS_DEFAULT.copy()
        hparams['magnitude_std'] = mstd

        ops = [
            timm.data.auto_augment.AugmentOp(
                name, prob=prob, magnitude=magnitude, hparams=hparams)
            for name in timm.data.auto_augment._RAND_TRANSFORMS]

        super().__init__(ops, num_layers)
        self.count = 0
