from typing import Any, Callable, Iterable, Optional, Type

import tqdm

_ENABLED = True
_MODULE = tqdm.tqdm


def setup_progressbar(enabled: bool, module: Type = tqdm.tqdm):
    global _ENABLED, _MODULE
    _ENABLED = enabled
    _MODULE = module


class ProgressBar(object):
    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        hook: Optional[Callable[['ProgressBar', Any], None]] = None,
        **kwargs,
    ) -> None:
        self.iterable = iterable
        self.hook = hook

        if _ENABLED:
            self.module = _MODULE(iterable=iterable, **kwargs)
        else:
            self.module = None

    def __iter__(self):
        if self.module is not None:
            iterator = self.module
        else:
            iterator = self.iterable

        for value in iterator:
            if self.hook is not None:
                self.hook(self, value)
            yield value

    def __getattr__(self, name):
        if name not in ['update', 'close', 'clear', 'refresh', 'unpause',
                        'reset', 'set_description', 'set_postfix', 'display']:
            raise AttributeError(name)

        if self.module is not None:
            return getattr(self.module, name)
        else:
            return self.do_nothing

    def do_nothing(self, *args, **kwargs):
        pass

    # def update(self, n=1):
    #     if self.module is not None:
    #         self.module.update(n=n)

    # def close(self):
    #     if self.module is not None:
    #         self.module.close()

    # def clear(self):
    #     if self.module is not None:
    #         self.module.clear()

    # def refresh(self):
    #     if self.module is not None:
    #         self.module.refresh()

    # def set_description(self, desc=None, refresh=True):
    #     if self.module is not None:
    #         self.module.set_description(desc=desc, refresh=refresh)

    # def set_postfix(self, ordered_dict=None, refresh=True, **tqdm_kwargs):
    #     if self.module is not None:
    #         self.module.set_postfix(
    #             ordered_dict=ordered_dict, refresh=refresh, **tqdm_kwargs)
