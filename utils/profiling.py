import os
from contextlib import contextmanager
from collections.abc import Callable, Generator
from functools import wraps

import torch

@contextmanager
def nvtx_range(msg: str) -> Generator[None, None, None]:
    if os.environ.get("ENABLE_NVTX") != "1":
        yield
        return

    with torch.cuda.nvtx.range(msg):
        yield
        
def nvtx_annotate(msg: str):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with nvtx_range(msg):
                return func(*args, **kwargs)

        return wrapper
    return decorator