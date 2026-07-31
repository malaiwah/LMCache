# SPDX-License-Identifier: Apache-2.0
"""CPU-safe tests for CUDA IPC export ownership state.

The CUDA runtime calls are replaced with counters so these tests prove the
one-import-or-one-release contract without opening a GPU handle.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import MagicMock
import pickle
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.base_ipc_wrapper import (
    DeviceIPCWrapper,
    acquire_ipc_export_lease,
    release_ipc_exports,
    wrap_ipc_tensors_rollback_safe,
)
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper
from lmcache.v1.multiprocess.custom_types import get_customized_encoder
import lmcache.v1.platform.cuda.ipc_wrapper as cuda_ipc_mod


def _wrapper(*, receiver: bool = True) -> CudaIPCWrapper:
    wrapper = CudaIPCWrapper.__new__(CudaIPCWrapper)
    wrapper.handle = (
        0,
        b"memory-handle",
        16,
        0,
        b"refcounter-handle",
        7,
        b"event-handle",
        False,
    )
    wrapper.dtype = torch.float32
    wrapper.shape = (1,)
    wrapper.stride = (1,)
    wrapper.storage_offset = 0
    wrapper.device_uuid = "GPU-test"
    wrapper._initialize_ipc_ownership(auto_release=receiver)
    return wrapper


def test_to_tensor_imports_once_across_concurrent_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper()
    imported_tensor = MagicMock()
    call_lock = threading.Lock()
    import_calls = 0

    def import_storage(
        _self: CudaIPCWrapper, _device_index: int
    ) -> torch.UntypedStorage:
        nonlocal import_calls
        with call_lock:
            import_calls += 1
        time.sleep(0.01)
        return cast(torch.UntypedStorage, MagicMock())

    monkeypatch.setattr(
        CudaIPCWrapper,
        "_get_device_index_from_uuid",
        classmethod(lambda _cls, _uuid: 0),
    )
    monkeypatch.setattr(cuda_ipc_mod.torch, "empty", lambda *_a, **_kw: imported_tensor)
    monkeypatch.setattr(CudaIPCWrapper, "_open_shared_storage", import_storage)

    with ThreadPoolExecutor(max_workers=8) as pool:
        tensors = list(pool.map(lambda _idx: wrapper.to_tensor(), range(32)))

    assert import_calls == 1
    assert all(tensor is imported_tensor for tensor in tensors)
    assert wrapper.release_ipc_export() is False


def test_release_is_idempotent_and_prevents_late_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper()
    release_calls = 0

    def release_counter(_self: CudaIPCWrapper) -> None:
        nonlocal release_calls
        release_calls += 1

    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    assert wrapper.release_ipc_export() is True
    assert wrapper.release_ipc_export() is False
    assert release_calls == 1
    with pytest.raises(RuntimeError, match="no longer available"):
        wrapper.to_tensor()


def test_concurrent_release_and_transfer_have_one_terminal_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper()
    release_calls = 0

    def release_counter(_self: CudaIPCWrapper) -> None:
        nonlocal release_calls
        release_calls += 1

    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)
    DeviceIPCWrapper.Serialize(wrapper)
    actions = [
        ("release", wrapper.release_ipc_export),
        ("transfer", wrapper.mark_ipc_export_transferred),
    ] * 16

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda item: (item[0], item[1]()), actions))

    release_wins = sum(won for action, won in outcomes if action == "release")
    transfer_wins = sum(won for action, won in outcomes if action == "transfer")
    assert release_wins + transfer_wins == 1
    assert release_calls == release_wins


def test_import_failure_releases_reservation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper()
    release_calls = 0

    def fail_device_lookup(_cls: type[CudaIPCWrapper], _uuid: str) -> int:
        raise ValueError("cannot open handle")

    def release_counter(_self: CudaIPCWrapper) -> None:
        nonlocal release_calls
        release_calls += 1

    monkeypatch.setattr(
        CudaIPCWrapper,
        "_get_device_index_from_uuid",
        classmethod(fail_device_lookup),
    )
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    with pytest.raises(ValueError, match="cannot open handle"):
        wrapper.to_tensor()
    assert release_calls == 1
    assert wrapper.release_ipc_export() is False


def test_tensor_rebuild_failure_uses_imported_storage_deleter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once storage opens, its deleter—not an explicit release—owns the count."""
    wrapper = _wrapper()
    tensor_shell = MagicMock()
    tensor_shell.set_.side_effect = ValueError("invalid tensor view")
    release_counter = MagicMock()

    monkeypatch.setattr(
        CudaIPCWrapper,
        "_get_device_index_from_uuid",
        classmethod(lambda _cls, _uuid: 0),
    )
    monkeypatch.setattr(
        cuda_ipc_mod.torch, "empty", lambda *_args, **_kwargs: tensor_shell
    )
    monkeypatch.setattr(
        CudaIPCWrapper,
        "_open_shared_storage",
        lambda _self, _device_index: MagicMock(),
    )
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    with pytest.raises(ValueError, match="invalid tensor view"):
        wrapper.to_tensor()

    release_counter.assert_not_called()
    assert wrapper.release_ipc_export() is False


def test_native_storage_failure_is_quarantined_without_explicit_decrement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A C++ RAII failure may already own the sole refcount decrement."""
    wrapper = _wrapper()
    release_counter = MagicMock()

    monkeypatch.setattr(
        CudaIPCWrapper,
        "_get_device_index_from_uuid",
        classmethod(lambda _cls, _uuid: 0),
    )
    monkeypatch.setattr(cuda_ipc_mod.torch, "empty", lambda *_a, **_kw: MagicMock())
    monkeypatch.setattr(
        CudaIPCWrapper,
        "_open_shared_storage",
        lambda _self, _device_index: (_ for _ in ()).throw(
            RuntimeError("StorageImpl construction failed after DataPtr")
        ),
    )
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    with pytest.raises(RuntimeError, match="StorageImpl construction failed"):
        wrapper.to_tensor()

    release_counter.assert_not_called()
    assert wrapper._ipc_state == wrapper._QUARANTINED
    assert wrapper.release_ipc_export() is False


def test_constructor_failure_after_share_releases_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata discovery failure cannot leak a completed _share_cuda_ call."""

    class _FakeStorage:
        def _share_cuda_(self) -> tuple[object, ...]:
            return _wrapper(receiver=False).handle

    class _FakeTensor:
        dtype = torch.float32
        shape = (1,)
        device = MagicMock(index=0)

        def untyped_storage(self) -> _FakeStorage:
            return _FakeStorage()

        def stride(self) -> tuple[int, ...]:
            return (1,)

        def storage_offset(self) -> int:
            return 0

    release_counter = MagicMock()

    def fail_uuid(_cls: type[CudaIPCWrapper], _device_index: int) -> str:
        raise ValueError("UUID unavailable")

    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.kv_format.contiguity."
        "attempt_permute_to_contiguous_view",
        lambda tensor: tensor,
    )
    monkeypatch.setattr(
        CudaIPCWrapper,
        "_get_device_uuid",
        classmethod(fail_uuid),
    )
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    with pytest.raises(ValueError, match="UUID unavailable"):
        CudaIPCWrapper(cast(torch.Tensor, _FakeTensor()))

    release_counter.assert_called_once()


def test_receiver_can_serialize_and_transfer_ownership_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _wrapper(receiver=False)
    release_calls = 0

    def release_counter(_self: CudaIPCWrapper) -> None:
        nonlocal release_calls
        release_calls += 1

    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    encoded = DeviceIPCWrapper.Serialize(sender)
    assert sender.mark_ipc_export_transferred() is True
    receiver = DeviceIPCWrapper.Deserialize(encoded)
    downstream_bytes = DeviceIPCWrapper.Serialize(receiver)
    assert receiver.mark_ipc_export_transferred() is True
    assert receiver.release_ipc_export() is False
    downstream = DeviceIPCWrapper.Deserialize(downstream_bytes)
    assert downstream.release_ipc_export() is True
    assert downstream.release_ipc_export() is False
    assert release_calls == 1


def test_unused_receiver_destructor_releases_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receiver = _wrapper()
    release_calls = 0

    def release_counter(_self: CudaIPCWrapper) -> None:
        nonlocal release_calls
        release_calls += 1

    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    receiver.__del__()
    receiver.__del__()

    assert release_calls == 1


def test_imported_wrapper_cannot_be_serialized_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper()
    imported_tensor = MagicMock()
    monkeypatch.setattr(
        CudaIPCWrapper,
        "_get_device_index_from_uuid",
        classmethod(lambda _cls, _uuid: 0),
    )
    monkeypatch.setattr(cuda_ipc_mod.torch, "empty", lambda *_a, **_kw: imported_tensor)
    monkeypatch.setattr(
        CudaIPCWrapper,
        "_open_shared_storage",
        lambda _self, _device_index: cast(torch.UntypedStorage, MagicMock()),
    )

    wrapper.to_tensor()

    with pytest.raises(RuntimeError, match="more than once.*state=imported"):
        DeviceIPCWrapper.Serialize(wrapper)


def test_serialization_claim_is_one_shot_and_duplicate_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper(receiver=False)
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    encoded = DeviceIPCWrapper.Serialize(wrapper)
    assert encoded
    assert wrapper._ipc_state == wrapper._ENCODED
    with pytest.raises(RuntimeError, match="more than once.*state=encoded"):
        DeviceIPCWrapper.Serialize(wrapper)

    # The rejected duplicate encoder did not steal or release the reservation.
    release_counter.assert_not_called()
    assert wrapper.release_ipc_export() is True
    release_counter.assert_called_once()


def test_raw_pickle_cannot_bypass_managed_one_shot_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python spawn/pickle is not a second transport for CUDA exports."""
    wrapper = _wrapper(receiver=False)
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    with pytest.raises(RuntimeError, match="DeviceIPCWrapper.Serialize exactly once"):
        pickle.dumps(wrapper)

    assert wrapper._ipc_state == wrapper._UNCONSUMED
    assert wrapper.release_ipc_export() is True
    release_counter.assert_called_once()


def test_duplicate_wrapper_identity_in_one_payload_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper(receiver=False)
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)
    encoder = get_customized_encoder(list[DeviceIPCWrapper])

    with pytest.raises(RuntimeError, match="more than once.*state=encoded"):
        encoder.encode([wrapper, wrapper])

    assert wrapper.release_ipc_export() is True
    release_counter.assert_called_once()


def test_serialization_failure_releases_its_claim_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper(receiver=False)
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)
    monkeypatch.setattr(
        cuda_ipc_mod.pickle,
        "dumps",
        lambda _obj: (_ for _ in ()).throw(ValueError("encode failed")),
    )

    with pytest.raises(ValueError, match="encode failed"):
        DeviceIPCWrapper.Serialize(wrapper)

    release_counter.assert_called_once()
    assert wrapper.release_ipc_export() is False


def test_transport_lease_defers_handler_cleanup_until_async_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper(receiver=True)
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    lease = acquire_ipc_export_lease([wrapper])
    # Simulate handler-finally cleanup before an async polling loop encodes.
    release_ipc_exports([wrapper])
    assert wrapper._ipc_release_pending is True
    release_counter.assert_not_called()

    encoded = DeviceIPCWrapper.Serialize(wrapper)
    assert encoded
    assert wrapper.mark_ipc_export_transferred() is True
    lease.release()

    release_counter.assert_not_called()
    assert wrapper._ipc_state == wrapper._TRANSFERRED


def test_transport_lease_releases_once_when_async_send_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper(receiver=True)
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    lease = acquire_ipc_export_lease([wrapper])
    release_ipc_exports([wrapper])
    DeviceIPCWrapper.Serialize(wrapper)
    # The send failure cleanup remains deferred until the async owner drops
    # its lease.
    assert wrapper.release_ipc_export() is False
    release_counter.assert_not_called()
    lease.release()

    release_counter.assert_called_once()
    assert wrapper._ipc_state == wrapper._RELEASED


def test_encode_release_race_has_exactly_one_reservation_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for _ in range(100):
        wrapper = _wrapper(receiver=False)
        release_calls = 0
        release_lock = threading.Lock()

        def release_counter(_self: CudaIPCWrapper, lock: Any = release_lock) -> None:
            nonlocal release_calls
            with lock:
                release_calls += 1

        monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)
        barrier = threading.Barrier(2)

        def encode(
            target: CudaIPCWrapper = wrapper,
            gate: threading.Barrier = barrier,
        ) -> bool:
            gate.wait()
            try:
                DeviceIPCWrapper.Serialize(target)
                return True
            except RuntimeError:
                return False

        def release(
            target: CudaIPCWrapper = wrapper,
            gate: threading.Barrier = barrier,
        ) -> bool:
            gate.wait()
            return target.release_ipc_export()

        with ThreadPoolExecutor(max_workers=2) as pool:
            encoded = pool.submit(encode)
            released = pool.submit(release)
            encode_won = encoded.result()
            release_won = released.result()

        assert encode_won or release_won
        if encode_won and not release_won:
            assert wrapper.release_ipc_export() is True
        assert release_calls == 1


def test_partial_producer_batch_construction_rolls_back_every_wrapper() -> None:
    class _Probe(DeviceIPCWrapper):
        def __init__(self) -> None:
            self.release_calls = 0

        def release_ipc_export(self) -> bool:
            self.release_calls += 1
            return True

    built: list[_Probe] = []

    def factory(index: int) -> _Probe:
        if index == 3:
            raise RuntimeError("fourth wrapper failed")
        wrapper = _Probe()
        built.append(wrapper)
        return wrapper

    with pytest.raises(RuntimeError, match="fourth wrapper failed"):
        wrap_ipc_tensors_rollback_safe(range(6), factory)

    assert len(built) == 3
    assert [wrapper.release_calls for wrapper in built] == [1, 1, 1]
