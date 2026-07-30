# SPDX-License-Identifier: Apache-2.0
"""Ordering and ownership tests for LMCache-driven IPC transfers."""

# Standard
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as transfer_mod


class _ImportedEvent:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def wait(self, *, stream: object) -> None:
        del stream
        self._calls.append("wait")

    def record(self) -> None:
        self._calls.append("record")


def _module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    LMCacheDrivenTransferModule,
    MagicMock,
    MagicMock,
    object,
    list[str],
]:
    calls: list[str] = []
    imported_event = _ImportedEvent(calls)
    handle = b"worker-owned-event"

    class _EventFactory:
        @staticmethod
        def from_ipc_handle(device: object, event_handle: bytes) -> _ImportedEvent:
            assert device == "cuda:0"
            assert event_handle == handle
            return imported_event

    fake_torch_dev = SimpleNamespace(
        device=lambda _device: nullcontext(),
        stream=lambda _stream: nullcontext(),
        Event=_EventFactory,
    )
    monkeypatch.setattr(transfer_mod, "torch_dev", fake_torch_dev)
    monkeypatch.setattr(
        transfer_mod,
        "check_interprocess_event_support",
        lambda: None,
    )
    monkeypatch.setattr(
        transfer_mod,
        "downsample_and_stage_block_ids",
        lambda *_args: calls.append("stage") or [[0]],
    )
    monkeypatch.setattr(
        transfer_mod,
        "transfer_kv_per_object_group",
        lambda *_args, **_kwargs: calls.append("transfer"),
    )
    monkeypatch.setattr(
        transfer_mod,
        "submit_callback_to_stream",
        MagicMock(),
    )
    monkeypatch.setattr(transfer_mod, "get_layout_desc", MagicMock())

    object_key = object()
    memory_obj = MagicMock()
    memory_obj.get_size.return_value = 128

    cache_context = MagicMock()
    cache_context.device = "cuda:0"
    cache_context.stream = object()
    cache_context.cupy_stream = object()
    cache_context.max_batch_size = 1
    cache_context.calculate_num_blocks.return_value = 1
    cache_context.kv_layer_groups_manager.num_object_groups = 1
    cache_context.kv_layer_groups_manager.num_kernel_groups = 1

    server_context = MagicMock()
    server_context.chunk_size = 16
    server_context.resolve_obj_keys.return_value = [[object_key]]
    server_context.storage_manager.reserve_write.return_value = {object_key: memory_obj}

    @contextmanager
    def _read_prefetched_results(_keys: list[object]):
        yield [memory_obj]

    server_context.storage_manager.read_prefetched_results = _read_prefetched_results

    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = server_context
    module.get_and_touch_context_entry = MagicMock(
        return_value=SimpleNamespace(
            cache_context=cache_context,
            model_name="model",
        )
    )
    return module, server_context, cache_context, handle, calls


def _key() -> SimpleNamespace:
    return SimpleNamespace(request_id="request", cache_salt="")


def test_store_rerecords_worker_owned_event_after_d2h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _server_context, _cache_context, handle, calls = _module(monkeypatch)

    result = module.store(_key(), 1, [[0]], handle)

    assert result == (handle, True)
    assert calls == ["stage", "wait", "transfer", "record"]


def test_retrieve_waits_for_worker_fence_before_h2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _server_context, _cache_context, handle, calls = _module(monkeypatch)

    result = module.retrieve(_key(), 1, [[0]], handle)

    assert result == (handle, True)
    assert calls == ["wait", "stage", "transfer", "record"]
