import hashlib
import logging
import os
import pathlib
from typing import Union

import requests

from .progressbar import ProgressBar

LOGGER = logging.getLogger(__name__)


def download_and_verify(
    path: Union[pathlib.Path, str],
    url: str,
    digest: str,
    size: int = None,
) -> None:
    if isinstance(path, str):
        path = pathlib.Path(path)

    download_path = path.parent / f'{path.name}.download'

    LOGGER.info('Downloading: %s', url)
    _download(download_path, url, size)

    LOGGER.info('Verifying: %s', download_path)
    _verify(download_path, digest)

    os.rename(download_path, path)
    LOGGER.info('Downloaded and Verified: %s', path)


def _download(path: pathlib.Path, url: str, size: int = None) -> None:
    response = requests.get(url, stream=True)

    if size is None:
        size = int(response.headers['content-length'])

    progbar = ProgressBar(
        total=size, unit='B', unit_scale=True, desc='Downloading')
    with open(path, 'wb') as writer:
        for chunk in response.iter_content(chunk_size=1024):
            writer.write(chunk)
            progbar.update(len(chunk))


def _verify(path: pathlib.Path, digest: str) -> None:
    digest_name, digest_value = digest.split(':', maxsplit=1)
    hashobj = hashlib.new(digest_name)
    size = path.stat().st_size

    progbar = ProgressBar(
        total=size, unit='B', unit_scale=True, desc='Verifying')
    with open(path, 'rb') as reader:
        while True:
            binary = reader.read(102400)
            if len(binary) == 0:
                break
            hashobj.update(binary)
            progbar.update(len(binary))

    if hashobj.hexdigest() != digest_value:
        os.remove(path)
        raise VerifyException(
            f'{path}:'
            f' Hash digest is {hashobj.hexdigest()},'
            f' but expected is {digest_value}')


class VerifyException(Exception):
    pass
