# SPDX-License-Identifier: Apache-2.0
"""CPU-safe tests for CUDA IPC export ownership state.

The CUDA runtime calls are replaced with counters so these tests prove the
one-import-or-one-release contract without opening a GPU handle.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import cast
from unittest.mock import MagicMock
import pickle
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper
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
    imported_tensor = cast(torch.Tensor, object())
    call_lock = threading.Lock()
    import_calls = 0

    def import_tensor(_self: CudaIPCWrapper) -> torch.Tensor:
        nonlocal import_calls
        with call_lock:
            import_calls += 1
        time.sleep(0.01)
        return imported_tensor

    monkeypatch.setattr(CudaIPCWrapper, "_import_tensor_from_handle", import_tensor)

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


def test_pickle_receiver_can_transfer_ownership_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _wrapper(receiver=False)
    encoded = pickle.dumps(sender)
    receiver = pickle.loads(encoded)
    downstream = pickle.loads(pickle.dumps(receiver))
    release_calls = 0

    def release_counter(_self: CudaIPCWrapper) -> None:
        nonlocal release_calls
        release_calls += 1

    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    assert receiver.mark_ipc_export_transferred() is True
    assert receiver.release_ipc_export() is False
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
    monkeypatch.setattr(
        CudaIPCWrapper,
        "_import_tensor_from_handle",
        lambda _self: cast(torch.Tensor, object()),
    )

    wrapper.to_tensor()

    with pytest.raises(RuntimeError, match="after its export was imported"):
        pickle.dumps(wrapper)
