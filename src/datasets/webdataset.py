import gc
import io
import logging
import os
import subprocess
import time
import urllib.parse
from typing import Any, Dict, Generator, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

import webdataset as wds
import webdataset.gopen as wdsg

from . import augmentations

try:
    import torch_xla.core.xla_model as xm  # type:ignore
except Exception:
    xm = None


LOGGER = logging.getLogger(__name__)


@ torch.no_grad()
def apply_class_to_tensor(
    data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    for image, target in data:
        target = F.one_hot(
            torch.tensor(target), num_classes=num_classes).float()
        yield image, target


def apply_augmentations(
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    mixup_prob: float,
    mixup_alpha: float,
    cutmix_prob: float,
    cutmix_alpha: float,
    labelsmooth: float,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if ((mixup_prob > 0.0 and mixup_alpha > 0.0)
            or (cutmix_prob > 0.0 and cutmix_alpha > 0.0)):
        dataset = apply_mixup_cutmix(
            dataset=dataset,
            mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
            cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha)

    if labelsmooth > 0.0:
        dataset = apply_labelsmooth(dataset=dataset, labelsmooth=labelsmooth)

    return dataset


@ torch.no_grad()
def apply_mixup_cutmix(
    dataset: Any,
    mixup_prob: float,
    mixup_alpha: float,
    cutmix_prob: float,
    cutmix_alpha: float,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    cutmix_prob /= max(1.0 - mixup_prob, 1e-6)
    image1, target1 = None, None
    image2, target2 = None, None

    for values in dataset:
        image1, target1 = image2, target2
        image2, target2 = values

        if image1 is None:
            continue

        if mixup_prob + cutmix_prob > 1.0:
            mixup_prob /= mixup_prob + cutmix_prob
            cutmix_prob /= mixup_prob + cutmix_prob

        prob = np.random.rand()

        if prob < mixup_prob:
            yield augmentations.apply_mixup(
                image1, target1, image2, target2, mixup_alpha)
        elif prob < mixup_prob + cutmix_prob:
            yield augmentations.apply_cutmix(
                image1, target1, image2, target2, cutmix_alpha)
        else:
            yield image1, target1

    if image2 is not None and target2 is not None:
        yield image2, target2


@torch.no_grad()
def apply_labelsmooth(
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    labelsmooth: float,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    for image, target in dataset:
        yield augmentations.apply_labelsmooth(image, target, labelsmooth)


class PytorchShardList(wds.PytorchShardList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_env(self):
        if self.rank is None and xm is not None:
            world = xm.xrt_world_size(defval=0)
            if world != 0:
                self.rank = xm.get_ordinal(), world

        super().update_env()


class Processor(wds.Processor):
    def __init__(self, loader, length, sampler):
        self.length = length
        self.sampler = sampler
        super().__init__(loader, wds.iterators.identity)

    def __len__(self):
        return self.length


def url_opener(data: Sequence[Dict[str, Any]], **kwargs) -> Any:
    for sample in data:
        try:
            with gopen(sample['url'], **kwargs) as stream:
                sample.update(stream=stream)
                yield sample
        except Exception as exn:
            exn.args = exn.args + (sample['url'],)
            raise exn


def gopen_pipe(url, mode="rb", bufsize=8192):
    assert url.startswith("pipe:")
    cmd = url[5:]
    if mode[0] == "r":
        return Pipe(
            cmd, mode=mode, shell=True, bufsize=bufsize, ignore_status=[141])
    elif mode[0] == "w":
        return Pipe(
            cmd, mode=mode, shell=True, bufsize=bufsize, ignore_status=[141])
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_curl(url, mode="rb", bufsize=8192):
    if mode[0] == "r":
        cmd = f"curl -s -L '{url}'"
        return Pipe(
            cmd, mode=mode, shell=True, bufsize=bufsize, ignore_status=[141, 23])
    elif mode[0] == "w":
        cmd = f"curl -s -L -T - '{url}'"
        return Pipe(
            cmd, mode=mode, shell=True, bufsize=bufsize, ignore_status=[141, 26])
    else:
        raise ValueError(f"{mode}: unknown mode")


class Pipe(wdsg.Pipe):
    def close(self):
        self.stream.close()
        self.proc.terminate()
        self.status = self.proc.wait(self.timeout)


class GSReader(io.BytesIO):
    def __init__(
        self,
        url: str,
        mode: str = 'rb',
        bufsize: int = 8192,
        timeout: float = 600.0,
        retry: int = 10,
        interval: float = 3.0,
    ) -> None:
        assert url.startswith("gs://")
        assert mode == 'rb'

        for _ in range(retry):
            binary, error = self._read_from_storage(url, bufsize, timeout)
            if binary is not None:
                break
            LOGGER.debug('GSReader: %s [ERROR] %s', url, error)
            time.sleep(interval)

        if binary is None:
            raise Exception(
                f'Error occured at reading data from {url}: {error}')

        LOGGER.debug('GSReader: %s [%d bytes]', url, len(binary))
        super().__init__(binary)

    def _read_from_storage(
        self,
        url: str,
        bufsize: int,
        timeout: float,
    ) -> Tuple[Optional[bytes], Optional[str]]:
        pipe = subprocess.Popen(
            ['gsutil', 'cat', url], bufsize=bufsize,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = None
        error = None

        try:
            stdout, stderr = pipe.communicate(timeout=timeout)
            pipe.wait(timeout=timeout)
            output = stdout
            error = stderr.decode('utf-8')
        except Exception as e:
            pipe.kill()
            error = str(e)

        if output is None or pipe.returncode != 0:
            return None, error

        return output, None

    def close(self):
        super().close()
        gc.collect()


def gopen(url, mode="rb", bufsize=8192, **kw):
    gopen_schemes = dict(
        __default__=wdsg.gopen_error,
        pipe=gopen_pipe,
        http=gopen_curl,
        https=gopen_curl,
        sftp=gopen_curl,
        ftps=gopen_curl,
        scp=gopen_curl,
        gs=GSReader,
    )

    pr = urllib.parse.urlparse(url)
    if pr.scheme == "":
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(url, mode, buffering=bufsize)
    if pr.scheme == "file":
        bufsize = int(os.environ.get("GOPEN_BUFFER", -1))
        return open(pr.path, mode, buffering=bufsize)
    handler = gopen_schemes["__default__"]
    handler = gopen_schemes.get(pr.scheme, handler)
    return handler(url, mode, bufsize, **kw)
