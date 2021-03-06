from .base import BaseBlock


class SwinBlock(BaseBlock):
    '''
    Block class for Swin Transformers.
    Downsample is performed before the main block.
    '''

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            downsample_before_block=True,
            **kwargs)
