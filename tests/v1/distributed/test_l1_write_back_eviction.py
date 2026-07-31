# SPDX-License-Identifier: Apache-2.0
"""Opt-in L1 write-back tier.

With ``EvictionConfig.write_back_on_evict`` enabled, L1 eviction flushes
readable objects to L2 synchronously and deletes them from L1 only once
every key in the batch is durable. ``periodic_flush_interval`` adds a
below-watermark backup flush that keeps the L1 copy.
``emergency_evict_for_prefetch`` lets the prefetch controller make room
for large restores through the same write-back path.

Uses the CPU shared-memory allocator; no GPU required.
"""

# Standard
from collections.abc import Iterator
import argparse
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    InFlightPrefetchRequest,
    PrefetchController,
    PrefetchPhase,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.mp_observability.config import add_observability_args

POOL_BYTES = 8 * 1024 * 1024

# 1MB per object so a handful of objects create real memory pressure.
OBJECT_LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([256, 1024])],
    dtypes=[torch.float32],
)


class FakeSyncStoreAdapter:
    """L2 adapter double exposing only the synchronous store path.

    ``result`` overrides the returned tuple; by default every key is
    reported durable.
    """

    def __init__(self) -> None:
        self.stored_batches: list[list[ObjectKey]] = []
        self.result: tuple[bool, int, int] | None = None

    def store_objects_sync(
        self,
        keys: list[ObjectKey],
        objects: list,
        timeout: float | None = None,
    ) -> tuple[bool, int, int]:
        self.stored_batches.append(list(keys))
        if self.result is not None:
            return self.result
        return True, len(keys), sum(obj.get_size() for obj in objects)


class BlockingSyncStoreAdapter(FakeSyncStoreAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def store_objects_sync(self, keys, objects, timeout=None):
        self.entered.set()
        if not self.release.wait(timeout=timeout or 5.0):
            return False, 0, 0
        return super().store_objects_sync(keys, objects, timeout=timeout)


@pytest.fixture
def l1_manager() -> Iterator[L1Manager]:
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


def _make_controller(
    l1_manager: L1Manager,
    adapter: FakeSyncStoreAdapter,
    **config_kwargs,
) -> L1EvictionController:
    config = EvictionConfig(eviction_policy="LRU", **config_kwargs)
    return L1EvictionController(
        l1_manager=l1_manager,
        eviction_config=config,
        l2_adapters={0: adapter},
    )


def _make_keys(count: int) -> list[ObjectKey]:
    return [
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(i),
            model_name="test_model",
            kv_rank=0,
        )
        for i in range(count)
    ]


def _store_keys(l1_manager: L1Manager, keys: list[ObjectKey]) -> None:
    result = l1_manager.reserve_write(keys, [False] * len(keys), OBJECT_LAYOUT)
    for key in keys:
        assert result[key][0] == L1Error.SUCCESS
    l1_manager.finish_write(keys)


def _readable_keys(l1_manager: L1Manager, keys: list[ObjectKey]) -> list[ObjectKey]:
    result = l1_manager.reserve_read(keys)
    readable = [
        key
        for key in keys
        if result.get(key) is not None and result[key][0] == L1Error.SUCCESS
    ]
    if readable:
        l1_manager.finish_read(readable)
    return readable


class TestWriteBackOnEvict:
    def test_flush_then_delete(self, l1_manager):
        """A durable flush deletes the batch from L1."""
        adapter = FakeSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter, write_back_on_evict=True)
        keys = _make_keys(3)
        _store_keys(l1_manager, keys)

        controller.execute_eviction_action(
            EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
        )

        assert adapter.stored_batches == [keys]
        assert _readable_keys(l1_manager, keys) == []

    def test_failed_flush_preserves_l1(self, l1_manager):
        """A failed flush keeps every readable key in L1 and opens the
        circuit breaker."""
        adapter = FakeSyncStoreAdapter()
        adapter.result = (False, 0, 0)
        controller = _make_controller(l1_manager, adapter, write_back_on_evict=True)
        keys = _make_keys(3)
        _store_keys(l1_manager, keys)

        controller.execute_eviction_action(
            EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
        )

        assert _readable_keys(l1_manager, keys) == keys
        status = controller.report_status()
        assert status["sync_flush_failures"] == 1
        assert status["sync_flush_backoff_seconds"] > 0.0

    def test_partial_flush_preserves_l1(self, l1_manager):
        """A partial persist is not durable; the batch stays in L1."""
        adapter = FakeSyncStoreAdapter()
        adapter.result = (False, 2, 1024)
        controller = _make_controller(l1_manager, adapter, write_back_on_evict=True)
        keys = _make_keys(3)
        _store_keys(l1_manager, keys)

        controller.execute_eviction_action(
            EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
        )

        assert _readable_keys(l1_manager, keys) == keys

    def test_missing_sync_adapter_fails_closed(self, l1_manager):
        config = EvictionConfig(eviction_policy="LRU", write_back_on_evict=True)
        controller = L1EvictionController(
            l1_manager=l1_manager,
            eviction_config=config,
            l2_adapters={0: object()},
        )
        keys = _make_keys(2)
        _store_keys(l1_manager, keys)

        controller.execute_eviction_action(
            EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
        )

        assert controller.has_l2_flush_adapter() is False
        assert _readable_keys(l1_manager, keys) == keys

    def test_reserve_read_failure_is_not_deleted(self, l1_manager, monkeypatch):
        adapter = FakeSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter, write_back_on_evict=True)
        keys = _make_keys(1)
        _store_keys(l1_manager, keys)
        monkeypatch.setattr(
            l1_manager,
            "reserve_read",
            lambda _keys: {keys[0]: (L1Error.KEY_IS_LOCKED, None)},
        )

        controller.execute_eviction_action(
            EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
        )

        monkeypatch.undo()
        assert adapter.stored_batches == []
        assert _readable_keys(l1_manager, keys) == keys

    def test_adapter_replacement_waits_for_active_flush(self, l1_manager):
        adapter = BlockingSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter, write_back_on_evict=True)
        keys = _make_keys(1)
        _store_keys(l1_manager, keys)
        flush = threading.Thread(
            target=lambda: controller.execute_eviction_action(
                EvictionAction(keys=keys, destination=EvictionDestination.L2_CACHE)
            )
        )
        replaced = threading.Event()
        replace = threading.Thread(
            target=lambda: (
                controller.set_l2_adapters({}),
                replaced.set(),
            )
        )

        flush.start()
        assert adapter.entered.wait(timeout=5.0)
        replace.start()
        time.sleep(0.05)
        assert not replaced.is_set()

        adapter.release.set()
        flush.join(timeout=5.0)
        replace.join(timeout=5.0)

        assert replaced.is_set()
        assert not flush.is_alive()
        assert not replace.is_alive()


class TestEmergencyEvict:
    def test_emergency_evict_frees_bytes_via_write_back(self, l1_manager):
        """emergency_evict_bytes flushes LRU victims to L2 and frees L1."""
        adapter = FakeSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter, write_back_on_evict=True)
        keys = _make_keys(6)
        _store_keys(l1_manager, keys)

        used_before, total = l1_manager.get_memory_usage()
        free_before = total - used_before

        free_after = controller.emergency_evict_bytes(
            free_before + 2 * 1024 * 1024, requester="test"
        )

        assert free_after > free_before
        assert adapter.stored_batches
        evicted = [k for batch in adapter.stored_batches for k in batch]
        assert set(evicted) <= set(keys)
        assert len(evicted) < len(keys)
        # Evicted keys are gone from L1.
        assert not set(evicted) & set(_readable_keys(l1_manager, keys))

    def test_emergency_evict_noop_when_enough_free(self, l1_manager):
        adapter = FakeSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter, write_back_on_evict=True)
        keys = _make_keys(2)
        _store_keys(l1_manager, keys)

        controller.emergency_evict_bytes(1024, requester="test")

        assert adapter.stored_batches == []
        assert _readable_keys(l1_manager, keys) == keys

    def test_emergency_evict_skips_scan_during_backoff(self, l1_manager, monkeypatch):
        adapter = FakeSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter, write_back_on_evict=True)
        keys = _make_keys(6)
        _store_keys(l1_manager, keys)
        controller._sync_flush_backoff_until = time.monotonic() + 60
        monkeypatch.setattr(
            controller._eviction_policy,
            "get_eviction_actions",
            lambda *args, **kwargs: pytest.fail("eviction scan should be skipped"),
        )

        used, total = l1_manager.get_memory_usage()
        free = total - used
        assert controller.emergency_evict_bytes(free + 1024) == free

    def test_emergency_evict_has_bounded_store_wait(self, l1_manager):
        adapter = BlockingSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter, write_back_on_evict=True)
        controller._EMERGENCY_FLUSH_TIMEOUT_SECONDS = 0.05
        keys = _make_keys(6)
        _store_keys(l1_manager, keys)
        used, total = l1_manager.get_memory_usage()

        start = time.monotonic()
        free_after = controller.emergency_evict_bytes(total - used + 1024)
        elapsed = time.monotonic() - start

        assert adapter.entered.is_set()
        assert elapsed < 0.5
        assert free_after == total - used
        assert _readable_keys(l1_manager, keys) == keys
        status = controller.report_status()
        assert status["sync_flush_failures"] == 0
        assert status["sync_flush_backoff_seconds"] == 0.0


class TestPeriodicBackupFlush:
    def test_backup_flush_keeps_l1_copy(self, l1_manager):
        """The periodic backup flush copies keys to L2 without deleting
        them from L1."""
        adapter = FakeSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter, periodic_flush_interval=0.1)
        keys = _make_keys(3)
        _store_keys(l1_manager, keys)

        controller.start()
        try:
            deadline = time.monotonic() + 5.0
            while not adapter.stored_batches and time.monotonic() < deadline:
                time.sleep(0.1)
        finally:
            controller.stop()

        assert adapter.stored_batches
        flushed = {k for batch in adapter.stored_batches for k in batch}
        assert flushed <= set(keys)
        assert _readable_keys(l1_manager, keys) == keys

    def test_backup_batches_rotate_without_full_snapshot(self, l1_manager):
        adapter = FakeSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter)
        keys = _make_keys(6)
        _store_keys(l1_manager, keys)

        for _ in range(3):
            controller._backup_to_l2_no_delete(batch_limit=2)

        assert [len(batch) for batch in adapter.stored_batches] == [2, 2, 2]
        assert {key for batch in adapter.stored_batches for key in batch} == set(keys)

    def test_backup_flush_has_bounded_store_wait(self, l1_manager):
        adapter = BlockingSyncStoreAdapter()
        controller = _make_controller(l1_manager, adapter)
        controller._PERIODIC_FLUSH_TIMEOUT_SECONDS = 0.05
        keys = _make_keys(2)
        _store_keys(l1_manager, keys)

        start = time.monotonic()
        controller._backup_to_l2_no_delete(batch_limit=2)
        elapsed = time.monotonic() - start

        assert adapter.entered.is_set()
        assert elapsed < 0.5
        assert _readable_keys(l1_manager, keys) == keys

    def test_evictable_batch_scan_wraps_and_is_bounded(self, l1_manager):
        keys = _make_keys(5)
        _store_keys(l1_manager, keys)
        assert l1_manager.reserve_read([keys[0]])[keys[0]][0] == L1Error.SUCCESS
        try:
            first, cursor = l1_manager.get_evictable_keys(
                limit=2,
                cursor=0,
                scan_limit=2,
            )
            second, cursor = l1_manager.get_evictable_keys(
                limit=2,
                cursor=cursor,
                scan_limit=2,
            )
            third, _ = l1_manager.get_evictable_keys(
                limit=2,
                cursor=cursor,
                scan_limit=2,
            )
        finally:
            l1_manager.finish_read([keys[0]])

        assert first == [keys[1]]
        assert second == keys[2:4]
        assert third == [keys[4]]


class _RecordingEvictor:
    """Eviction-controller double recording emergency_evict_bytes calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def emergency_evict_bytes(self, target_free_bytes: int, requester: str = "") -> int:
        self.calls.append((target_free_bytes, requester))
        return target_free_bytes


@pytest.fixture
def prefetch_controller(l1_manager: L1Manager) -> Iterator[PrefetchController]:
    controller = PrefetchController(
        l1_manager=l1_manager,
        l2_adapters=[],
        adapter_descriptors=[],
        policy=DefaultPrefetchPolicy(),
    )
    yield controller
    # The loop thread was never started, so stop() cannot join it.
    controller._submission_efd.close()
    controller._adapter_ctrl_efd.close()


def _make_request(keys: list[ObjectKey]) -> InFlightPrefetchRequest:
    return InFlightPrefetchRequest(
        request_id=7,
        keys=keys,
        layout_desc=OBJECT_LAYOUT,
        phase=PrefetchPhase.PLAN_AND_LOAD,
    )


class TestPrefetchMakeRoom:
    def test_make_room_asks_evictor_when_short(self, l1_manager, prefetch_controller):
        """A restore that does not fit in free L1 asks for eviction."""
        controller = prefetch_controller
        evictor = _RecordingEvictor()
        controller.set_l1_eviction_controller(evictor)
        resident = _make_keys(6)
        _store_keys(l1_manager, resident)

        restore_keys = [
            ObjectKey(
                chunk_hash=ObjectKey.IntHash2Bytes(100 + i),
                model_name="test_model",
                kv_rank=0,
            )
            for i in range(4)
        ]
        controller._make_room_for_restore(_make_request(restore_keys), restore_keys)

        assert len(evictor.calls) == 1
        target, requester = evictor.calls[0]
        assert target > 0
        assert "prefetch_request=7" in requester

    def test_make_room_noop_when_space_available(self, l1_manager, prefetch_controller):
        controller = prefetch_controller
        evictor = _RecordingEvictor()
        controller.set_l1_eviction_controller(evictor)

        keys = _make_keys(2)
        controller._make_room_for_restore(_make_request(keys), keys)

        assert evictor.calls == []

    def test_make_room_noop_without_evictor(self, l1_manager, prefetch_controller):
        controller = prefetch_controller
        keys = _make_keys(2)

        controller._make_room_for_restore(_make_request(keys), keys)

    def test_eviction_controller_can_be_disconnected(self, prefetch_controller):
        evictor = _RecordingEvictor()
        prefetch_controller.set_l1_eviction_controller(evictor)
        prefetch_controller.set_l1_eviction_controller(None)

        assert prefetch_controller._l1_eviction_controller is None

    def test_short_retention_policy_degrades_to_temporary(
        self, prefetch_controller, monkeypatch
    ):
        keys = _make_keys(3)
        monkeypatch.setattr(
            prefetch_controller._policy,
            "select_l1_retentions",
            lambda _keys: [True],
        )

        assert prefetch_controller._select_l1_retentions(7, keys) == [
            False,
            False,
            False,
        ]

    def test_retry_recovers_oom_reservations(self, l1_manager, prefetch_controller):
        """OOM reservations are retried once after emergency eviction."""
        controller = prefetch_controller
        evictor = _RecordingEvictor()
        controller.set_l1_eviction_controller(evictor)
        keys = _make_keys(2)
        write_results = {key: (L1Error.OUT_OF_MEMORY, None) for key in keys}

        merged = controller._retry_oom_reservations(
            _make_request(keys), keys, [True, True], write_results
        )

        assert len(evictor.calls) == 1
        for key in keys:
            err, mem_obj = merged[key]
            assert err == L1Error.SUCCESS
            assert mem_obj is not None
        l1_manager.finish_write(keys)

    def test_retry_noop_when_all_reserved(self, l1_manager, prefetch_controller):
        controller = prefetch_controller
        evictor = _RecordingEvictor()
        controller.set_l1_eviction_controller(evictor)
        keys = _make_keys(2)
        result = l1_manager.reserve_write(keys, [False, False], OBJECT_LAYOUT)

        merged = controller._retry_oom_reservations(
            _make_request(keys), keys, [True, True], result
        )

        assert evictor.calls == []
        assert merged is result
        l1_manager.finish_write(keys)


class TestConfigPlumbing:
    def test_parser_populates_write_back_fields(self):
        parser = argparse.ArgumentParser()
        add_storage_manager_args(parser)
        add_observability_args(parser)
        args = parser.parse_args(
            [
                "--l1-size-gb",
                "1",
                "--eviction-policy",
                "LRU",
                "--write-back-on-evict",
                "--periodic-flush-interval",
                "30.0",
                "--emergency-evict-for-prefetch",
            ]
        )
        config = parse_args_to_config(args)
        assert config.eviction_config.write_back_on_evict is True
        assert config.eviction_config.periodic_flush_interval == 30.0
        assert config.eviction_config.emergency_evict_for_prefetch is True

    def test_defaults_are_disabled(self):
        parser = argparse.ArgumentParser()
        add_storage_manager_args(parser)
        add_observability_args(parser)
        args = parser.parse_args(["--l1-size-gb", "1", "--eviction-policy", "LRU"])
        config = parse_args_to_config(args)
        assert config.eviction_config.write_back_on_evict is False
        assert config.eviction_config.periodic_flush_interval == 0.0
        assert config.eviction_config.emergency_evict_for_prefetch is False

    @pytest.mark.parametrize(
        "eviction_config",
        [
            EvictionConfig(
                eviction_policy="LRU",
                periodic_flush_interval=-1.0,
            ),
            EvictionConfig(
                eviction_policy="LRU",
                emergency_evict_for_prefetch=True,
            ),
        ],
    )
    def test_invalid_writeback_config_is_rejected(self, eviction_config):
        with pytest.raises(ValueError):
            StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=POOL_BYTES,
                        use_lazy=False,
                    )
                ),
                eviction_config=eviction_config,
            )
