# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the device-type-driven dispatch in
``lmcache.v1.platform.cache_context.create_cache_context``.

The facade routes wrappers by non-consuming class-level ``device_type``
metadata and rejects legacy wrappers that would need a one-shot ``to_tensor``
probe. These tests exercise both paths without touching CUDA or real context
constructors.
"""

# Standard
from typing import Any, List

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import cache_context as cache_context_module
from lmcache.v1.platform.base_cache_context import BaseCacheContext
from lmcache.v1.platform.cache_context import create_cache_context


class _FakeCPUWrapper:
    device_type = "cpu"

    def to_tensor(self) -> torch.Tensor:
        raise AssertionError("device detection must be non-consuming")


class _FakeCUDAWrapper(_FakeCPUWrapper):
    device_type = "cuda"


class _LegacyWrapper:
    def to_tensor(self) -> torch.Tensor:
        raise AssertionError("legacy one-shot wrapper must not be consumed")


class _DeclaredWrapper:
    """Built-in-style wrapper whose device metadata avoids handle import."""

    device_type = "cpu"

    def to_tensor(self) -> torch.Tensor:
        raise AssertionError("device dispatch must not consume the IPC export")


class _FakeContext(BaseCacheContext):
    """Bare-bones ``BaseCacheContext`` subclass used as a registry stub.

    All abstract members are no-ops: the test never invokes them; it
    only checks that the right class is instantiated with the
    forwarded arguments.
    """

    device_type = "fake"

    def __init__(
        self,
        kv_caches: Any,
        lmcache_tokens_per_chunk: int,
        layout_hints: Any,
        engine_group_infos: Any,
        engine_type: Any,
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
    ) -> None:
        # Skip ``BaseCacheContext.__init__`` -- it requires real
        # KVLayerGroupsManager / shape descriptors that are out of
        # scope for the dispatch test.
        self.kv_caches = kv_caches
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self.layout_hints = layout_hints
        self.engine_group_infos = engine_group_infos
        self.engine_type = engine_type
        self.separate_object_groups = separate_object_groups
        self.full_sw_kv = full_sw_kv

    # ------------------------------------------------------------------
    # Abstract stubs -- never called from these tests.
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:  # pragma: no cover - never invoked
        return torch.float32

    @property
    def stream(self) -> Any:  # pragma: no cover - never invoked
        return None

    @property
    def cupy_stream(self) -> Any:  # pragma: no cover - never invoked
        return None

    @property
    def max_batch_size(self) -> int:  # pragma: no cover - never invoked
        return 0

    def close(self) -> None:  # pragma: no cover - never invoked
        return None

    def get_kernel_group_kv_pointers(
        self, kernel_group_idx: int
    ) -> torch.Tensor:  # pragma: no cover
        return torch.empty(0)

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:  # pragma: no cover
        return torch.empty(0)

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:  # pragma: no cover
        return torch.empty(0)

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> Any:  # pragma: no cover
        return torch.Size(()), torch.float32

    def cache_size_per_token(self) -> int:  # pragma: no cover
        return 0


class _FakeCPUContext(_FakeContext):
    device_type = "cpu"


class _FakeCUDAContext(_FakeContext):
    device_type = "cuda"


@pytest.fixture
def isolated_registry() -> Any:
    """Snapshot the backend table so each test can install fakes
    without polluting other tests / the production setup."""
    saved = cache_context_module.snapshot_backends()
    # Start each test from an empty backend table so we can assert
    # the "no class registered" branch deterministically.
    cache_context_module.restore_backends({})
    try:
        yield
    finally:
        cache_context_module.restore_backends(saved)


def _install(**backends: type) -> None:
    """Replace the live backend table with *backends*."""
    cache_context_module.restore_backends(dict(backends))


def test_dispatches_by_cpu_device_type(isolated_registry: None) -> None:
    """Wrappers reporting ``cpu`` tensors must yield the cpu-registered
    class."""
    _install(cpu=_FakeCPUContext)

    wrappers: List[_FakeCPUWrapper] = [_FakeCPUWrapper(), _FakeCPUWrapper()]
    ctx = create_cache_context(wrappers)  # type: ignore[arg-type]

    assert isinstance(ctx, _FakeCPUContext)
    assert ctx.kv_caches is wrappers


def test_declared_device_type_does_not_materialize_wrapper(
    isolated_registry: None,
) -> None:
    """Built-in wrapper metadata must preserve its one-shot import for backend use."""
    _install(cpu=_FakeCPUContext)
    wrappers = [_DeclaredWrapper()]

    ctx = create_cache_context(wrappers)  # type: ignore[arg-type]

    assert isinstance(ctx, _FakeCPUContext)
    assert ctx.kv_caches is wrappers


def test_dispatches_by_cuda_device_type(isolated_registry: None) -> None:
    """Wrappers reporting a non-cpu device type must route to the
    matching registered class -- no isinstance branching."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _install(cuda=_FakeCUDAContext)

    wrappers = [_FakeCUDAWrapper()]
    ctx = create_cache_context(wrappers, lmcache_tokens_per_chunk=128)  # type: ignore[arg-type]

    assert isinstance(ctx, _FakeCUDAContext)
    assert ctx.lmcache_tokens_per_chunk == 128


def test_empty_kv_caches_raises(isolated_registry: None) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        create_cache_context([])


def test_mixed_device_types_raises(isolated_registry: None) -> None:
    """Cross-device batches are unsupported and must fail loudly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _install(cpu=_FakeCPUContext, cuda=_FakeCUDAContext)

    wrappers = [_FakeCPUWrapper(), _FakeCUDAWrapper()]
    with pytest.raises(ValueError, match="share one"):
        create_cache_context(wrappers)  # type: ignore[arg-type]


def test_unregistered_device_type_raises(isolated_registry: None) -> None:
    """An unknown device type is a hard failure with a clear hint."""
    wrappers = [_FakeCPUWrapper()]
    with pytest.raises(ValueError, match="No cache-context class"):
        create_cache_context(wrappers)  # type: ignore[arg-type]


def test_legacy_wrapper_without_device_metadata_is_rejected(
    isolated_registry: None,
) -> None:
    with pytest.raises(TypeError, match="to_tensor.*unsafe"):
        create_cache_context([_LegacyWrapper()])  # type: ignore[list-item]
