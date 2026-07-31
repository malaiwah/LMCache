# SPDX-License-Identifier: Apache-2.0
"""Partial-prefix allocation behavior of ``L1Manager.reserve_write``.

A batch whose full allocation does not fit must not fail every key
(all-or-nothing would collapse large L2->L1 restores to zero prefix hits):
the longest prefix that fits is allocated and only the tail reports
OUT_OF_MEMORY. Uses the CPU shared-memory allocator, no GPU required.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager

POOL_BYTES = 64 * 1024 * 1024

# 8MB per object: ten of them cannot fit in the 64MB pool, most can.
OBJECT_LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([2048, 1024])],
    dtypes=[torch.float32],
)

# A single 128MB object exceeds the whole pool.
OVERSIZED_LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([2048 * 16, 1024])],
    dtypes=[torch.float32],
)


@pytest.fixture
def manager() -> Iterator[L1Manager]:
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=POOL_BYTES,
            use_lazy=False,
            init_size_in_bytes=POOL_BYTES,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    manager = L1Manager(config)
    yield manager
    manager.close()


def _make_keys(count: int) -> list[ObjectKey]:
    return [
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(i),
            model_name="test_model",
            kv_rank=0,
        )
        for i in range(count)
    ]


def test_reserve_write_allocates_longest_prefix_on_oom(manager: L1Manager) -> None:
    """When the full batch does not fit, a non-empty prefix succeeds and
    only the tail reports OUT_OF_MEMORY; every requested key keeps an
    explicit per-key result."""
    keys = _make_keys(10)
    result = manager.reserve_write(keys, [False] * 10, OBJECT_LAYOUT)

    assert set(result) == set(keys)
    statuses = [result[key][0] for key in keys]
    num_success = statuses.count(L1Error.SUCCESS)
    assert 0 < num_success < 10

    # The successful keys form a prefix in request order.
    assert statuses == [L1Error.SUCCESS] * num_success + [L1Error.OUT_OF_MEMORY] * (
        10 - num_success
    )
    for key in keys[:num_success]:
        assert result[key][1] is not None
    for key in keys[num_success:]:
        assert result[key][1] is None


def test_partial_prefix_is_committable(manager: L1Manager) -> None:
    """The allocated prefix stays usable: finish_write succeeds for it."""
    keys = _make_keys(10)
    result = manager.reserve_write(keys, [False] * 10, OBJECT_LAYOUT)
    successful = [key for key in keys if result[key][0] == L1Error.SUCCESS]
    assert successful

    finish_result = manager.finish_write(successful)
    for key in successful:
        assert finish_result[key] == L1Error.SUCCESS


def test_reserve_write_all_oom_when_nothing_fits(manager: L1Manager) -> None:
    """An object larger than the pool still fails every key."""
    keys = _make_keys(3)
    result = manager.reserve_write(keys, [False] * 3, OVERSIZED_LAYOUT)

    for key in keys:
        assert result[key][0] == L1Error.OUT_OF_MEMORY
        assert result[key][1] is None


def test_single_key_batch_keeps_all_or_nothing(manager: L1Manager) -> None:
    """A one-key batch has no prefix to fall back to; it simply fails."""
    keys = _make_keys(1)
    result = manager.reserve_write(keys, [False], OVERSIZED_LAYOUT)

    assert result[keys[0]] == (L1Error.OUT_OF_MEMORY, None)


def test_largest_prefix_search_does_not_stop_at_a_smaller_fit() -> None:
    """The OOM fallback finds the maximum, rather than accepting a midpoint."""

    class BoundedMemoryManager:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity
            self.allocate_counts: list[int] = []

        def allocate(self, _layout: MemoryLayoutDesc, count: int):
            self.allocate_counts.append(count)
            if count > self.capacity:
                return L1Error.OUT_OF_MEMORY, []
            return L1Error.SUCCESS, [object() for _ in range(count)]

        def free(self, _objects: list[object]) -> L1Error:
            return L1Error.SUCCESS

    memory_manager = BoundedMemoryManager(capacity=7)
    manager = SimpleNamespace(_memory_manager=memory_manager)

    err, objects = L1Manager._allocate_largest_prefix(
        manager,
        OBJECT_LAYOUT,
        10,  # type: ignore[arg-type]
    )

    assert err == L1Error.SUCCESS
    assert len(objects) == 7
    assert memory_manager.allocate_counts[-1] == 7
    assert 8 in memory_manager.allocate_counts
