import gc
import io
import logging
import os
import subprocess
import time
import urllib.parse
from typing import Any, Dict, Generator, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.utils.data

import webdataset as wds
import webdataset.gopen as wdsg

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
        # Attribute `dataset` is a mimic attribute which is required for
        # pytorch_lightning.plugins.training_type.TPUSpawnPlugin
        self.dataset = None
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
