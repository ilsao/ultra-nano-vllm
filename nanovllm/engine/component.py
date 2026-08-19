from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from hashlib import sha256
import importlib
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BlockManagerProtocol(Protocol):
    def can_allocate(self, seq: Any) -> int: ...
    def allocate(self, seq: Any, num_cached_blocks: int) -> None: ...
    def deallocate(self, seq: Any) -> None: ...
    def can_append(self, seq: Any) -> bool: ...
    def may_append(self, seq: Any) -> None: ...
    def hash_blocks(self, seq: Any) -> None: ...


@runtime_checkable
class SchedulerProtocol(Protocol):
    def is_finished(self) -> bool: ...
    def add(self, seq: Any) -> None: ...
    def schedule(self) -> tuple[list[Any], bool]: ...
    def postprocess(
        self,
        seqs: list[Any],
        token_ids: list[int],
        is_prefill: bool,
    ) -> None: ...


@runtime_checkable
class AttentionProtocol(Protocol):
    k_cache: Any
    v_cache: Any

    def __call__(self, q: Any, k: Any, v: Any) -> Any: ...


@runtime_checkable
class SamplerProtocol(Protocol):
    def __call__(self, logits: Any, temperatures: Any) -> Any: ...


@runtime_checkable
class StoreKVCacheProtocol(Protocol):
    def __call__(
        self,
        key: Any,
        value: Any,
        k_cache: Any,
        v_cache: Any,
        slot_mapping: Any,
    ) -> None: ...


_ROOT = Path(__file__).resolve().parent.parent
_IMPLEMENTATIONS = {
    "scheduler": (
        _ROOT / "engine" / "scheduler",
        "nanovllm.engine.scheduler",
        "create_component",
    ),
    "block_manager": (
        _ROOT / "engine" / "block_manager",
        "nanovllm.engine.block_manager",
        "create_component",
    ),
    "attention": (
        _ROOT / "layers" / "attention",
        "nanovllm.layers.attention",
        "create_component",
    ),
    "sampler": (
        _ROOT / "layers" / "sampler",
        "nanovllm.layers.sampler",
        "create_component",
    ),
    "store_kvcache": (
        _ROOT / "kernels" / "store_kvcache",
        "nanovllm.kernels.store_kvcache",
        "create_kernel",
    ),
}
_SELECTOR_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class EngineComponent:
    """File-name selectors for replaceable engine components and kernels."""

    scheduler: str = "scheduler"
    block_manager: str = "block_manager"
    attention: str = "attention"
    sampler: str = "sampler"
    store_kvcache: str = "store_kvcache"

    def __post_init__(self) -> None:
        for field in fields(self):
            _load_factory(field.name, getattr(self, field.name))

    def create_block_manager(
        self,
        num_blocks: int,
        block_size: int,
    ) -> BlockManagerProtocol:
        return _create(
            "block_manager",
            self.block_manager,
            BlockManagerProtocol,
            num_blocks=num_blocks,
            block_size=block_size,
        )

    def create_scheduler(self, config: Any) -> SchedulerProtocol:
        block_manager = self.create_block_manager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        return _create(
            "scheduler",
            self.scheduler,
            SchedulerProtocol,
            config=config,
            block_manager=block_manager,
        )

    def create_attention(
        self,
        *,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ) -> AttentionProtocol:
        return _create(
            "attention",
            self.attention,
            AttentionProtocol,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            store_kvcache=self.create_store_kvcache(),
        )

    def create_sampler(self) -> SamplerProtocol:
        return _create("sampler", self.sampler, SamplerProtocol)

    def create_store_kvcache(self) -> StoreKVCacheProtocol:
        return _create(
            "store_kvcache",
            self.store_kvcache,
            StoreKVCacheProtocol,
        )


ENGINE_COMPONENT_DIMENSIONS = tuple(field.name for field in fields(EngineComponent))


def validate_engine_component_selector(role: str, selector: str) -> str:
    """Validate one selector without constructing its implementation."""
    _load_factory(role, selector)
    return selector


def _create(
    role: str,
    selector: str,
    protocol: type,
    **kwargs: Any,
) -> Any:
    factory = _load_factory(role, selector)
    try:
        implementation = factory(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"could not create {role} implementation {selector!r}: {exc}"
        ) from exc
    if not isinstance(implementation, protocol):
        raise TypeError(
            f"{role} implementation {selector!r} returned "
            f"{type(implementation).__name__}, which does not satisfy "
            f"{protocol.__name__}"
        )
    return implementation


def _load_factory(role: str, selector: str):
    if role not in _IMPLEMENTATIONS:
        raise ValueError(f"unknown engine component role: {role}")
    if (
        not isinstance(selector, str)
        or ".." in selector
        or not _SELECTOR_PATTERN.fullmatch(selector)
    ):
        raise ValueError(
            f"invalid {role} selector {selector!r}; use a local file name without paths"
        )
    return _load_factory_cached(role, selector)


@lru_cache(maxsize=None)
def _load_factory_cached(role: str, selector: str):
    directory, package, export_name = _IMPLEMENTATIONS[role]
    path = directory / f"{selector}.py"
    if not path.is_file():
        raise ValueError(
            f"{role} implementation {selector!r} does not exist at {path}"
        )

    if selector.isidentifier():
        module_name = f"{package}.{selector}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ValueError(
                f"could not import {role} implementation {selector!r} "
                f"from {path}: {exc}"
            ) from exc
    else:
        digest = sha256(str(path).encode()).hexdigest()[:16]
        module_name = f"{package}._selected_{digest}"
        module = sys.modules.get(module_name)
    if not selector.isidentifier() and module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not load {role} implementation from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise ValueError(
                f"could not import {role} implementation {selector!r} "
                f"from {path}: {exc}"
            ) from exc

    factory = getattr(module, export_name, None)
    if not callable(factory):
        raise ValueError(
            f"{role} implementation {selector!r} at {path} must export "
            f"callable {export_name}"
        )
    return factory
