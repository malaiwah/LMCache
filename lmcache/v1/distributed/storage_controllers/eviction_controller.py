# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from abc import abstractmethod
from collections import Counter
from enum import Enum, auto
from typing import TYPE_CHECKING
import inspect
import math
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import L1EvictionPolicy, L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.quota_manager import QuotaManager

logger = init_logger(__name__)


class _SyncFlushResult(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    DEADLINE_EXHAUSTED = auto()


class EvictionController(StorageControllerInterface):
    """
    Abstract base class for eviction controllers.

    Provides the shared eviction loop structure: background thread and stop
    flag. Subclasses implement eviction_loop and execute_eviction_action
    for their specific tier (L1 or L2).
    """

    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self.eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    @abstractmethod
    def report_status(self) -> dict:
        """Return a status dict for this controller.

        The child class needs to override this function to report
        controller-specific health and configuration information.
        """
        pass

    @abstractmethod
    def eviction_loop(self):
        """Run the eviction loop.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass

    @abstractmethod
    def execute_eviction_action(self, action: EvictionAction):
        """Execute a single eviction action.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass


class L1EvictionController(EvictionController):
    """
    Eviction controller for L1 cache.

    Uses an L1EvictionPolicy bridge to keep the eviction policy up-to-date
    with L1 manager events, and periodically triggers eviction based on
    L1 memory usage.

    With ``EvictionConfig.write_back_on_evict`` enabled and L2 adapters
    injected via ``set_l2_adapters``, eviction actions target
    ``EvictionDestination.L2_CACHE``: readable objects are synchronously
    persisted to L2 in bounded batches and deleted from L1 only once every
    key in the batch is durable. Repeated flush failures open a circuit
    breaker that pauses write-back instead of retrying every second.

    ``EvictionConfig.periodic_flush_interval`` additionally enables a
    below-watermark backup flush that copies evictable L1 keys to L2
    without deleting them from L1.
    """

    # Bounded batches keep each blocking sync-store call short.
    _SYNC_FLUSH_BATCH_SIZE = 128
    # Keys per periodic backup flush cycle.
    _BACKUP_FLUSH_BATCH_SIZE = 128
    # A prefetch request must not stall the controller indefinitely on L2.
    _EMERGENCY_FLUSH_TIMEOUT_SECONDS = 1.0
    # Periodic backup runs under the flush lock and must remain stoppable.
    _PERIODIC_FLUSH_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        l1_manager: L1Manager,
        eviction_config: EvictionConfig,
        l2_adapters: dict[int, L2AdapterInterface] | None = None,
    ):
        super().__init__()
        self._eviction_config = eviction_config
        self._eviction_policy = CreateEvictionPolicy(eviction_config)
        self._l1_manager = l1_manager
        self._listener = L1EvictionPolicy(self._eviction_policy)
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()
        self._last_extra_log = time.monotonic()

        self._write_back_enabled = bool(eviction_config.write_back_on_evict)
        self._l2_adapters: dict[int, L2AdapterInterface] = {}
        self._l2_adapter_supports_timeout: dict[int, bool] = {}
        self._l2_adapters_lock = threading.Lock()
        # Serializes all synchronous L2 flushes and adapter-set replacement.
        # StorageManager can therefore remove an adapter only after any L1
        # writeback currently using it has finished.
        self._flush_lock = threading.Lock()
        # Sync-flush circuit breaker: a dead L2 must not cause a
        # multi-thousand-key synchronous retry and warning storm every
        # second.
        self._sync_flush_failures: int = 0
        self._sync_flush_backoff_until: float = 0.0
        self._last_backup_flush = time.monotonic()
        self._backup_flush_cursor: int = 0
        # Serializes emergency evictions from concurrent callers.
        self._emergency_evict_lock = threading.Lock()
        if self._write_back_enabled:
            # Writeback must fail closed. Register the L2 destination even
            # before a compatible adapter exists so policy never falls back
            # to DISCARD when persistence is unavailable.
            self._eviction_policy.register_eviction_destination(
                EvictionDestination.L2_CACHE
            )
        if l2_adapters:
            self.set_l2_adapters(l2_adapters)

    def set_l2_adapters(self, l2_adapters: dict[int, L2AdapterInterface]) -> None:
        """Late-inject L2 adapters after StorageManager creates them.

        Only adapters with an explicit synchronous durability contract are
        eligible. The L2 eviction destination itself is registered at
        construction so enabled writeback always fails closed.
        """
        compatible: dict[int, L2AdapterInterface] = {}
        supports_timeout: dict[int, bool] = {}
        for adapter_id, adapter in l2_adapters.items():
            sync_store = getattr(adapter, "store_objects_sync", None)
            if not callable(sync_store):
                continue
            compatible[adapter_id] = adapter
            try:
                supports_timeout[adapter_id] = (
                    "timeout" in inspect.signature(sync_store).parameters
                )
            except (TypeError, ValueError):
                supports_timeout[adapter_id] = False
        ignored = sorted(set(l2_adapters) - set(compatible))
        if ignored:
            logger.warning(
                "L1 writeback ignored adapters without store_objects_sync: %s",
                ignored,
            )
        with self._flush_lock:
            with self._l2_adapters_lock:
                self._l2_adapters = compatible
                self._l2_adapter_supports_timeout = supports_timeout

    def has_l2_flush_adapter(self) -> bool:
        """Return whether a synchronous durability backend is available."""
        return bool(self._snapshot_l2_adapters())

    def _snapshot_l2_adapters(self) -> list[tuple[int, L2AdapterInterface]]:
        with self._l2_adapters_lock:
            return list(self._l2_adapters.items())

    def _snapshot_l2_flush_adapters(
        self,
    ) -> list[tuple[int, L2AdapterInterface, bool]]:
        with self._l2_adapters_lock:
            return [
                (
                    adapter_id,
                    adapter,
                    self._l2_adapter_supports_timeout.get(adapter_id, False),
                )
                for adapter_id, adapter in self._l2_adapters.items()
            ]

    def report_status(self) -> dict:
        adapters = self._snapshot_l2_adapters()
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "eviction_policy": self._eviction_config.eviction_policy,
            "trigger_watermark": self._eviction_config.trigger_watermark,
            "eviction_ratio": self._eviction_config.eviction_ratio,
            "write_back_enabled": self._write_back_enabled,
            "l2_flush_enabled": bool(adapters),
            "l2_flush_adapter_ids": [adapter_id for adapter_id, _ in adapters],
            "periodic_flush_interval": (self._eviction_config.periodic_flush_interval),
            "sync_flush_failures": self._sync_flush_failures,
            "sync_flush_backoff_seconds": max(
                0.0, self._sync_flush_backoff_until - time.monotonic()
            ),
        }

    def _publish_skipped(self, usage: float, watermark: float) -> None:
        """Publish a below-watermark loop tick (no eviction this cycle)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": False,
                },
            )
        )

    def _publish_triggered(self, usage: float, watermark: float) -> None:
        """Publish an above-watermark loop tick (eviction policy ran)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": True,
                },
            )
        )

    def _maybe_log_memory_usage(self, used_bytes: int, total_bytes: int) -> None:
        """Emit the opt-in L1 memory usage INFO line, throttled to the interval."""
        now = time.monotonic()
        if now - self._last_extra_log < self._eviction_config.extra_logging_interval:
            return
        self._last_extra_log = now
        pct = 0.0 if total_bytes == 0 else used_bytes / total_bytes * 100.0
        logger.info(
            "L1 memory usage: %.2f/%.2f GiB (%.1f%%)",
            used_bytes / (1 << 30),
            total_bytes / (1 << 30),
            pct,
        )

    def eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio
        backup_interval = self._eviction_config.periodic_flush_interval

        while not self._stop_flag.is_set():
            time.sleep(1)
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            if self._eviction_config.extra_logging_enabled:
                self._maybe_log_memory_usage(used_bytes, total_bytes)
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                if (
                    backup_interval > 0
                    and self._snapshot_l2_adapters()
                    and time.monotonic() - self._last_backup_flush >= backup_interval
                ):
                    self._last_backup_flush = time.monotonic()
                    self._backup_to_l2_no_delete(self._BACKUP_FLUSH_BATCH_SIZE)
                logger.debug(
                    "L1 memory usage %.2f below watermark %.2f; skipping eviction.",
                    usage,
                    watermark,
                )
                self._publish_skipped(usage, watermark)
                continue

            if (
                self._write_back_enabled
                and time.monotonic() < self._sync_flush_backoff_until
            ):
                self._publish_skipped(usage, watermark)
                continue

            logger.info(
                "L1 memory usage %.2f above watermark %.2f; triggering eviction.",
                usage,
                watermark,
            )
            actions = self._eviction_policy.get_eviction_actions(
                eviction_ratio,
                key_eligible_filter=self._l1_manager.is_key_evictable,
            )
            for action in actions:
                self.execute_eviction_action(action)
            self._publish_triggered(usage, watermark)

    def execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.L2_CACHE:
            self._flush_to_l2_then_delete(action.keys)
        elif action.destination == EvictionDestination.DISCARD:
            self._l1_manager.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            self._l1_manager.delete(action.keys)

    def emergency_evict_bytes(self, target_free_bytes: int, requester: str = "") -> int:
        """Synchronously evict LRU L1 keys until at least
        ``target_free_bytes`` bytes are free.

        Victims take the normal eviction path, so with write-back enabled
        they are flushed to L2 before deletion. Intended for the prefetch
        controller when a large L2-to-L1 restore cannot reserve L1 buffers.

        Args:
            target_free_bytes: The free-space goal in bytes.
            requester: Optional label for logging.

        Returns:
            The free byte count after eviction.
        """
        deadline = time.monotonic() + self._EMERGENCY_FLUSH_TIMEOUT_SECONDS
        if not self._emergency_evict_lock.acquire(
            timeout=self._EMERGENCY_FLUSH_TIMEOUT_SECONDS
        ):
            used, total = self._l1_manager.get_memory_usage()
            return max(0, total - used)
        try:
            used, total = self._l1_manager.get_memory_usage()
            free = max(0, total - used)
            if free >= target_free_bytes:
                return free
            deficit = target_free_bytes - free
            num_objects = self._l1_manager.num_objects()
            if num_objects <= 0:
                return free
            if self._write_back_enabled and (
                time.monotonic() < self._sync_flush_backoff_until
                or not self.has_l2_flush_adapter()
            ):
                return free
            per_key = max(1, used // num_objects)
            need_keys = math.ceil(deficit / per_key)
            need_keys += max(1, math.ceil(need_keys / 8))
            tracked = num_objects
            get_tracked = getattr(self._eviction_policy, "get_num_tracked_keys", None)
            if callable(get_tracked):
                tracked = max(1, int(get_tracked()))
            ratio = min(1.0, need_keys / max(tracked, 1))
            start = time.monotonic()
            actions = self._eviction_policy.get_eviction_actions(
                ratio,
                key_eligible_filter=self._l1_manager.is_key_evictable,
            )
            evicted_keys = 0
            for action in actions:
                evicted_keys += len(action.keys)
                if action.destination == EvictionDestination.L2_CACHE:
                    self._flush_to_l2_then_delete(action.keys, deadline=deadline)
                else:
                    self.execute_eviction_action(action)
                if time.monotonic() >= deadline:
                    break
            used_after, total_after = self._l1_manager.get_memory_usage()
            free_after = max(0, total_after - used_after)
            logger.info(
                "Emergency L1 eviction%s: wanted %.0f MB free, evicted %d "
                "keys in %.0f ms; free %.0f MB -> %.0f MB",
                (" for " + requester) if requester else "",
                target_free_bytes / 1e6,
                evicted_keys,
                (time.monotonic() - start) * 1000.0,
                free / 1e6,
                free_after / 1e6,
            )
            return free_after
        finally:
            self._emergency_evict_lock.release()

    def _flush_to_l2_then_delete(
        self,
        keys: list[ObjectKey],
        deadline: float | None = None,
    ) -> None:
        """Persist bounded batches to L2 and delete only fully durable
        batches from L1.

        The native adapter reports a store successful only when every key
        persisted, so this method conservatively retains the whole readable
        batch on any partial failure. Repeated failures open the circuit
        breaker with exponential backoff.
        """
        if deadline is None:
            self._flush_lock.acquire()
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._flush_lock.acquire(timeout=remaining):
                return
        try:
            if not keys:
                return
            if time.monotonic() < self._sync_flush_backoff_until:
                return

            all_ok = True
            deadline_exhausted = False
            for start in range(0, len(keys), self._SYNC_FLUSH_BATCH_SIZE):
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_exhausted = True
                    break
                batch = keys[start : start + self._SYNC_FLUSH_BATCH_SIZE]
                result = self._flush_one_l2_batch_then_delete(batch, deadline=deadline)
                if result == _SyncFlushResult.DEADLINE_EXHAUSTED:
                    deadline_exhausted = True
                    break
                if result == _SyncFlushResult.FAILURE:
                    all_ok = False
                    break

            if deadline_exhausted:
                logger.debug(
                    "L1-to-L2 emergency flush exhausted its time budget; "
                    "remaining L1 keys preserved"
                )
            elif all_ok:
                self._sync_flush_failures = 0
                self._sync_flush_backoff_until = 0.0
            else:
                self._sync_flush_failures += 1
                delay = min(60.0, float(2 ** min(self._sync_flush_failures, 6)))
                self._sync_flush_backoff_until = time.monotonic() + delay
                logger.warning(
                    "L1-to-L2 flush circuit breaker: failure=%d, retry in %.0fs; "
                    "remaining L1 keys preserved",
                    self._sync_flush_failures,
                    delay,
                )
        finally:
            self._flush_lock.release()

    def _flush_one_l2_batch_then_delete(
        self,
        keys: list[ObjectKey],
        deadline: float | None = None,
    ) -> _SyncFlushResult:
        """Flush one batch to L2; delete from L1 only when fully durable.

        Returns:
            SUCCESS when every readable key was persisted (or none were
            readable), FAILURE for an actual persistence failure, and
            DEADLINE_EXHAUSTED when an emergency flush consumed its budget.
        """
        if not keys:
            return _SyncFlushResult.SUCCESS

        read_result = self._l1_manager.reserve_read(keys)
        readable_keys: list[ObjectKey] = []
        readable_objs = []
        for key in keys:
            entry = read_result.get(key)
            if (
                entry is not None
                and entry[0] == L1Error.SUCCESS
                and entry[1] is not None
            ):
                readable_keys.append(key)
                readable_objs.append(entry[1])

        flushed = not readable_keys
        deadline_exhausted = False
        if readable_keys:
            try:
                for (
                    idx,
                    adapter,
                    supports_timeout,
                ) in self._snapshot_l2_flush_adapters():
                    sync_store = adapter.store_objects_sync
                    try:
                        if deadline is None:
                            result = sync_store(readable_keys, readable_objs)
                        else:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                deadline_exhausted = True
                                break
                            if not supports_timeout:
                                logger.warning(
                                    "Emergency L1 writeback skipped adapter %d: "
                                    "store_objects_sync has no timeout contract",
                                    idx,
                                )
                                continue
                            result = sync_store(
                                readable_keys,
                                readable_objs,
                                timeout=remaining,
                            )
                        ok, persisted_count, bytes_written = result
                    except Exception:
                        if deadline is not None and time.monotonic() >= deadline:
                            deadline_exhausted = True
                            break
                        logger.exception(
                            "L1-to-L2 flush: sync store failed on adapter %d", idx
                        )
                        continue
                    if not ok and deadline is not None and time.monotonic() >= deadline:
                        deadline_exhausted = True
                        break
                    # Never delete unless every readable key is durable.
                    if ok and persisted_count == len(readable_keys):
                        flushed = True
                        logger.info(
                            "L1-to-L2 flush: sync persisted %d/%d keys "
                            "(%d bytes) via adapter %d",
                            persisted_count,
                            len(readable_keys),
                            bytes_written,
                            idx,
                        )
                        break
                    logger.warning(
                        "L1-to-L2 flush: adapter %d partial/failed persist "
                        "(%d/%d keys, %d bytes)",
                        idx,
                        persisted_count,
                        len(readable_keys),
                        bytes_written,
                    )
            finally:
                self._l1_manager.finish_read(readable_keys)

        # Only keys we read and then fully persisted may be deleted. A key
        # that failed reserve_read can have become locked after candidate
        # selection and must never be treated as absent.
        keys_to_delete = list(readable_keys) if flushed else []
        if not flushed and readable_keys:
            logger.warning(
                "L1-to-L2 flush: preserving %d readable keys after failed persist",
                len(readable_keys),
            )

        if keys_to_delete:
            result = self._l1_manager.delete(keys_to_delete)
            not_deleted = [k for k, err in result.items() if err != L1Error.SUCCESS]
            if not_deleted:
                logger.debug(
                    "L1-to-L2 flush: %d keys not deleted, likely still locked",
                    len(not_deleted),
                )
        if deadline_exhausted:
            return _SyncFlushResult.DEADLINE_EXHAUSTED
        if flushed:
            return _SyncFlushResult.SUCCESS
        return _SyncFlushResult.FAILURE

    def _backup_to_l2_no_delete(self, batch_limit: int) -> None:
        """Flush evictable L1 keys to L2 without deleting them from L1.

        A backup, not an eviction: keys remain in L1 for fast access while
        L2 gains a copy, so losing L1 (session expiry, restart) no longer
        costs a cold prefill. A rotating cursor spreads repeated scans over
        the whole L1 keyspace instead of hammering the first ``batch_limit``
        insertion-ordered keys forever. Adapters skip keys they already
        hold, so the flush is idempotent.
        """
        with self._flush_lock:
            self._backup_to_l2_no_delete_locked(batch_limit)

    def _backup_to_l2_no_delete_locked(self, batch_limit: int) -> None:
        """Implementation of periodic backup under ``_flush_lock``."""
        batch, self._backup_flush_cursor = self._l1_manager.get_evictable_keys(
            limit=batch_limit,
            cursor=self._backup_flush_cursor,
        )
        if not batch:
            return

        read_result = self._l1_manager.reserve_read(batch)
        readable_keys: list[ObjectKey] = []
        readable_objs = []
        for key in batch:
            entry = read_result.get(key)
            if (
                entry is not None
                and entry[0] == L1Error.SUCCESS
                and entry[1] is not None
            ):
                readable_keys.append(key)
                readable_objs.append(entry[1])

        if not readable_keys:
            return

        persisted_keys = 0
        persisted_bytes = 0
        try:
            for idx, adapter, supports_timeout in self._snapshot_l2_flush_adapters():
                if not supports_timeout:
                    logger.warning(
                        "Periodic L2 backup skipped adapter %d: "
                        "store_objects_sync has no timeout contract",
                        idx,
                    )
                    continue
                sync_store = adapter.store_objects_sync
                try:
                    ok, persisted, written = sync_store(
                        readable_keys,
                        readable_objs,
                        timeout=self._PERIODIC_FLUSH_TIMEOUT_SECONDS,
                    )
                    if ok and persisted == len(readable_keys):
                        persisted_keys = persisted
                        persisted_bytes = written
                        break
                except Exception:
                    logger.exception("Periodic L2 backup: sync store failed on %d", idx)
        finally:
            self._l1_manager.finish_read(readable_keys)

        if persisted_bytes > 0:
            logger.info(
                "Periodic L2 backup: persisted %d keys (%d new bytes, "
                "%d evictable in L1, none deleted)",
                persisted_keys,
                persisted_bytes,
                len(batch),
            )


class L2AdapterEvictionState:
    """Per-adapter eviction state: its own policy, listener, and config."""

    def __init__(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        eviction_config: EvictionConfig,
    ):
        self.adapter_id = adapter_id
        self.adapter = adapter
        self.eviction_config = eviction_config
        self.eviction_policy = CreateEvictionPolicy(eviction_config)
        self.listener = L2EvictionPolicy(self.eviction_policy)
        adapter.register_listener(self.listener)


class L2EvictionController(StorageControllerInterface):
    """
    Unified eviction controller for all L2 adapters.

    Each adapter gets its own eviction policy and listener bridge, but a
    single background thread loops over all of them.

    When the adapter's policy sets ``support_isolation == True``
    (e.g. :class:`IsolatedLRUEvictionPolicy`), the controller consults
    the injected :class:`QuotaManager` to decide which ``cache_salt``
    buckets are over budget and evicts from each one in isolation.
    Otherwise it uses the adapter's aggregate ``usage_fraction``
    against the configured watermark — unchanged from the pre-PR5
    behavior.
    """

    def __init__(
        self,
        l2_adapter_states: list[L2AdapterEvictionState],
        quota_manager: QuotaManager | None = None,
    ):
        self._adapter_states = l2_adapter_states
        self._quota_manager = quota_manager
        # Guards _adapter_states against concurrent runtime add/remove.
        self._states_lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    def add_adapter_state(self, state: L2AdapterEvictionState) -> None:
        """Register a new adapter's eviction state at runtime."""
        with self._states_lock:
            self._adapter_states.append(state)

    def remove_adapter_state(self, adapter_id: int) -> None:
        """Drop the eviction state for ``adapter_id``.

        Blocks until any in-progress eviction pass finishes (it holds the
        same lock), so the adapter is guaranteed idle here before the
        caller closes it. A no-op if the adapter has no eviction state.
        """
        with self._states_lock:
            self._adapter_states = [
                s for s in self._adapter_states if s.adapter_id != adapter_id
            ]

    def report_status(self) -> dict:
        # NOTE: ``usage.bytes_by_cache_salt`` is intentionally NOT
        # surfaced here. A deployment can have 10k+ salts, so embedding
        # the full bucket map in the status response would blow up the
        # payload. Per-salt inspection goes through the dedicated HTTP
        # quota endpoints (which pull from ``QuotaManager`` +
        # ``StorageManager.get_usage_bytes_by_cache_salt``).
        adapter_statuses = []
        with self._states_lock:
            states = list(self._adapter_states)
        for state in states:
            usage = state.adapter.get_usage()
            adapter_statuses.append(
                {
                    "eviction_policy": state.eviction_config.eviction_policy,
                    "trigger_watermark": state.eviction_config.trigger_watermark,
                    "eviction_ratio": state.eviction_config.eviction_ratio,
                    "current_usage": usage.usage_fraction,
                    "total_bytes_used": usage.total_bytes_used,
                    "total_capacity_bytes": usage.total_capacity_bytes,
                    "num_cache_salt_buckets": len(usage.bytes_by_cache_salt),
                }
            )
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "adapters": adapter_statuses,
        }

    def _eviction_loop(self):
        while not self._stop_flag.is_set():
            time.sleep(1)
            # Hold the lock across the whole pass so remove_adapter_state
            # cannot detach (and the caller close) an adapter while we are
            # calling into it.
            with self._states_lock:
                for state in self._adapter_states:
                    self._check_and_evict(state)

    def _check_and_evict(self, state: L2AdapterEvictionState):
        if state.eviction_policy.support_isolation and self._quota_manager is not None:
            self._check_and_evict_by_cache_salt(state)
        else:
            self._check_and_evict_global(state)

    def _check_and_evict_global(self, state: L2AdapterEvictionState):
        """Aggregate-usage eviction (``LRU`` / ``noop``)."""
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio

        # ``usage_fraction == -1`` means the adapter doesn't support
        # usage-based eviction (no max_capacity_bytes declared), so we
        # do not trigger eviction. Adapters with ``supports_global_eviction ==
        # False`` should already have been filtered out at construction
        # time in ``StorageManager``; this check is a defensive belt.
        current_usage = state.adapter.get_usage().usage_fraction
        if current_usage < 0 or current_usage < watermark:
            logger.debug(
                "L2 usage %.2f below watermark %.2f; skipping eviction.",
                current_usage,
                watermark,
            )
            return

        logger.info(
            "L2 usage %.2f above watermark %.2f; triggering eviction.",
            current_usage,
            watermark,
        )
        actions = state.eviction_policy.get_eviction_actions(eviction_ratio)
        for action in actions:
            self._execute_eviction_action(state.adapter, action)

    def _check_and_evict_by_cache_salt(self, state: L2AdapterEvictionState):
        """Per-``cache_salt`` eviction driven by :class:`QuotaManager`.

        For every salt with non-zero bytes, compare its usage against
        ``watermark * quota``. Salts over threshold get eviction scoped
        to their own LRU list. Salts with no quota registered have an
        effective limit of ``0`` and are therefore always over budget,
        so they get a full eviction (``effective_ratio=1.0``) — this
        enforces the allowlist rule: only registered salts retain data.

        Per-destination keys are batched across all over-budget salts
        before invoking the adapter — one ``adapter.delete(...)`` call
        per destination instead of one per (salt, destination) pair.
        Adapters with non-trivial per-call overhead (NIXL handle setup,
        FS sync, etc.) see this as a real win when many salts go over
        budget in the same cycle.
        """
        assert self._quota_manager is not None
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio
        usage = state.adapter.get_usage()

        # destination -> accumulated keys across all over-budget salts.
        pending: dict[EvictionDestination, list[ObjectKey]] = {}

        for cache_salt, user_bytes in usage.bytes_by_cache_salt.items():
            if user_bytes <= 0:
                continue
            limit = self._quota_manager.get_limit_bytes(cache_salt)
            # Trigger on ``>=`` to match the global branch's ``usage <
            # watermark`` short-circuit. Salts with no quota (limit=0)
            # always land here because ``user_bytes > 0 >= 0``.
            if user_bytes < watermark * limit:
                continue

            # Unregistered / zero-quota salts: wipe everything.
            # Registered salts: evict the configured ratio of their list.
            effective_ratio = 1.0 if limit == 0 else eviction_ratio
            logger.info(
                "cache_salt=%r over quota (bytes=%d, limit=%d, "
                "watermark=%.2f); evicting ratio=%.2f.",
                cache_salt,
                user_bytes,
                limit,
                watermark,
                effective_ratio,
            )
            actions = state.eviction_policy.get_eviction_actions(
                effective_ratio, cache_salt=cache_salt
            )
            for action in actions:
                pending.setdefault(action.destination, []).extend(action.keys)

        for destination, keys in pending.items():
            self._execute_eviction_action(
                state.adapter,
                EvictionAction(keys=keys, destination=destination),
            )

    def _execute_eviction_action(
        self, adapter: L2AdapterInterface, action: EvictionAction
    ):
        if action.destination == EvictionDestination.DISCARD:
            adapter.delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            adapter.delete(action.keys)

        if action.keys:
            get_event_bus().publish(
                Event(
                    event_type=EventType.L2_KEYS_EVICTED,
                    metadata={
                        "key_count": len(action.keys),
                        "key_count_per_salt": Counter(
                            k.cache_salt for k in action.keys
                        ),
                    },
                )
            )
