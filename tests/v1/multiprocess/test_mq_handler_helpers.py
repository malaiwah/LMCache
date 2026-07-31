# SPDX-License-Identifier: Apache-2.0
"""
Helper handler functions for MessageQueue tests.

These handlers are defined at module level to allow them to be pickled
and passed between processes during multiprocessing tests.
"""

# Standard
import gc
import weakref

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    KVCache,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocol import KeyType
from lmcache.v1.platform.base_ipc_wrapper import (
    DeviceIPCWrapper,
    release_ipc_exports,
)


# The CUDA MQ lifecycle test deliberately retains imported tensors until the
# same producer asks the server to unregister them.  This mirrors the
# production cache-context ownership contract rather than treating REGISTER as
# a one-shot serialization smoke test.
_REGISTERED_CUDA_KV_CACHES: dict[int, list[torch.Tensor]] = {}
_CUDA_KV_LIFETIME_PROBES: dict[
    int,
    tuple[
        list[weakref.ReferenceType[DeviceIPCWrapper]],
        list[weakref.ReferenceType[torch.Tensor]],
    ],
] = {}

# ==============================================================================
# NOOP Request Handlers
# ==============================================================================


def noop_handler() -> str:
    """
    Dummy handler for NOOP requests.
    Takes no arguments and returns a simple string response.
    """
    return "NOOP_OK"


def failing_noop_handler() -> str:
    """Raise a deterministic error for RPC error propagation tests."""
    raise ValueError("intentional sync handler failure")


# ==============================================================================
# REGISTER_KV_CACHE Request Handlers
# ==============================================================================


def register_kv_cache_handler(
    gpu_id: int,
    kv_cache: KVCache,
    model_name: str,
    world_size: int,
    engine_type: EngineType,
    layout_hints: LayoutHints,
    engine_group_infos: list[EngineGroupInfo],
) -> None:
    """
    Dummy handler for REGISTER_KV_CACHE requests.

    Args:
        gpu_id: GPU device ID
        kv_cache: List of CudaIPCWrapper objects representing KV cache
        model_name: Name of the model associated with this KV cache
        world_size: World size associated with this KV cache
        engine_type: Which serving engine produced the caches
        layout_hints: Engine-provided hints dict.
        engine_group_infos: Engine-neutral KV cache group metadata,
            msgspec-decoded from the request payload.

    Returns:
        None
    """
    # In a real implementation, this would register the KV cache
    # For testing, release the one-shot exports without importing them, then
    # validate the inputs. This mirrors the production NOOP ownership path and
    # prevents the fixture itself from leaking producer refcounts.
    release_ipc_exports(kv_cache)
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(kv_cache, list), (
        f"Expected kv_cache to be list, got {type(kv_cache)}"
    )
    assert isinstance(model_name, str), (
        f"Expected model_name to be str, got {type(model_name)}"
    )
    assert isinstance(world_size, int), (
        f"Expected world_size to be int, got {type(world_size)}"
    )
    assert isinstance(engine_type, EngineType), (
        f"Expected engine_type to be EngineType, got {type(engine_type)}"
    )
    assert isinstance(layout_hints, dict), (
        f"Expected layout_hints to be dict, got {type(layout_hints)}"
    )
    assert isinstance(engine_group_infos, list), (
        f"Expected engine_group_infos to be a list, got {type(engine_group_infos)}"
    )
    # No return value (returns None implicitly)


def register_and_retain_cuda_kv_cache_handler(
    gpu_id: int,
    kv_cache: KVCache,
    model_name: str,
    world_size: int,
    engine_type: EngineType,
    layout_hints: LayoutHints,
    engine_group_infos: list[EngineGroupInfo],
) -> None:
    """Import and retain a CUDA KV cache until explicit unregister.

    This handler is intentionally stateful.  The producer-side test client
    remains alive after REGISTER succeeds, sends UNREGISTER over the same MQ
    connection, and waits for that acknowledgement before it exits.
    """
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert gpu_id not in _REGISTERED_CUDA_KV_CACHES, (
        f"GPU ID {gpu_id} already has a retained KV cache"
    )
    assert isinstance(kv_cache, list), (
        f"Expected kv_cache to be list, got {type(kv_cache)}"
    )
    assert kv_cache, "Expected a non-empty CUDA KV cache"
    assert isinstance(model_name, str), (
        f"Expected model_name to be str, got {type(model_name)}"
    )
    assert isinstance(world_size, int), (
        f"Expected world_size to be int, got {type(world_size)}"
    )
    assert isinstance(engine_type, EngineType), (
        f"Expected engine_type to be EngineType, got {type(engine_type)}"
    )
    assert isinstance(layout_hints, dict), (
        f"Expected layout_hints to be dict, got {type(layout_hints)}"
    )
    assert isinstance(engine_group_infos, list), (
        f"Expected engine_group_infos to be a list, got {type(engine_group_infos)}"
    )

    imported_tensors = [wrapper.to_tensor() for wrapper in kv_cache]
    assert len(imported_tensors) == len(kv_cache)
    _REGISTERED_CUDA_KV_CACHES[gpu_id] = imported_tensors
    _CUDA_KV_LIFETIME_PROBES[gpu_id] = (
        [weakref.ref(wrapper) for wrapper in kv_cache],
        [weakref.ref(tensor) for tensor in imported_tensors],
    )


# ==============================================================================
# UNREGISTER_KV_CACHE Request Handlers
# ==============================================================================


def unregister_kv_cache_handler(gpu_id: int) -> None:
    """
    Dummy handler for UNREGISTER_KV_CACHE requests.

    Args:
        gpu_id: GPU device ID

    Returns:
        None
    """
    # In a real implementation, this would unregister the KV cache for the given GPU
    # For testing, we just validate the input is received correctly
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    # No return value (returns None implicitly)


def unregister_and_release_cuda_kv_cache_handler(gpu_id: int) -> None:
    """Drop retained imports and reclaim CUDA IPC state before replying."""
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert gpu_id in _REGISTERED_CUDA_KV_CACHES, (
        f"GPU ID {gpu_id} has no retained KV cache to unregister"
    )
    assert gpu_id in _CUDA_KV_LIFETIME_PROBES, (
        f"GPU ID {gpu_id} has no lifetime probes to unregister"
    )

    wrapper_refs, tensor_refs = _CUDA_KV_LIFETIME_PROBES.pop(gpu_id)
    gc.collect()
    live_wrappers = sum(ref() is not None for ref in wrapper_refs)
    assert live_wrappers == 0, (
        f"MQ dispatch retained {live_wrappers} decoded REGISTER wrapper(s) "
        "after the REGISTER response"
    )

    imported_tensors = _REGISTERED_CUDA_KV_CACHES.pop(gpu_id)
    assert imported_tensors, "Expected retained CUDA tensors before unregister"
    imported_tensors.clear()
    del imported_tensors
    gc.collect()

    # Match the production teardown contract closely: the UNREGISTER response
    # is not sent until imported storage has been dropped and CUDA IPC cleanup
    # has had an opportunity to return the producer refcounts.
    if torch_dev.is_available():
        torch_dev.synchronize()
        torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            ipc_collect()

    live_tensors = sum(ref() is not None for ref in tensor_refs)
    assert live_tensors == 0, (
        f"Receiver retained {live_tensors} imported REGISTER tensor(s) "
        "after UNREGISTER cleanup"
    )
    assert gpu_id not in _REGISTERED_CUDA_KV_CACHES
    assert gpu_id not in _CUDA_KV_LIFETIME_PROBES


# ==============================================================================
# STORE Request Handlers
# ==============================================================================


def store_handler(
    key: KeyType, gpu_id: int, gpu_block_ids: list[list[int]], ipc_handle: bytes
) -> tuple[bytes, bool]:
    """
    Dummy handler for STORE requests.

    Args:
        key: Cache key to store
        gpu_id: GPU device ID
        gpu_block_ids: GPU block IDs per KV cache group
        ipc_handle: CUDA event IPC handle

    Returns:
        tuple[bytes, bool]: (event handle, success flag)
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(gpu_block_ids)}"
    )
    assert all(isinstance(block_ids, list) for block_ids in gpu_block_ids), (
        "Expected gpu_block_ids to be list[list[int]]"
    )
    assert isinstance(ipc_handle, bytes), (
        f"Expected ipc_handle to be bytes, got {type(ipc_handle)}"
    )
    return b"\x01" * 64, True


# ==============================================================================
# RETRIEVE Request Handlers
# ==============================================================================


def retrieve_handler(
    key: KeyType,
    gpu_id: int,
    gpu_block_ids: list[list[int]],
    event_handler: bytes,
    skip_first_n_tokens: int = 0,
) -> tuple[bytes, bool]:
    """
    Dummy handler for RETRIEVE requests.

    Args:
        key: Cache key to retrieve
        gpu_id: GPU device ID
        gpu_block_ids: GPU block IDs per KV cache group
        event_handler: CUDA event IPC handle
        skip_first_n_tokens: Number of tokens to skip at retrieve start

    Returns:
        tuple[bytes, bool]: (event handle, success flag)
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(gpu_block_ids)}"
    )
    assert all(isinstance(block_ids, list) for block_ids in gpu_block_ids), (
        "Expected gpu_block_ids to be list[list[int]]"
    )
    assert isinstance(event_handler, bytes), (
        f"Expected event_handler to be bytes, got {type(event_handler)}"
    )
    assert isinstance(skip_first_n_tokens, int), (
        f"Expected skip_first_n_tokens to be int, got {type(skip_first_n_tokens)}"
    )
    return b"\x01" * 64, True


# ==============================================================================
# LOOKUP Request Handlers
# ==============================================================================


def lookup_handler(key: KeyType, tp_size: int) -> None:
    """
    Dummy handler for LOOKUP requests.

    Args:
        key: Cache key to look up (request_id embedded in the key)
        tp_size: Tensor-parallel size for MLA
            multi-reader locking

    Returns:
        None: LOOKUP registers the job server-side; poll via QUERY_PREFETCH_STATUS.
    """
    # In a real implementation, this would look up the key in the cache
    # For testing, we just validate the input
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(tp_size, int), f"Expected tp_size to be int, got {type(tp_size)}"


def failing_lookup_handler(key: KeyType, tp_size: int) -> None:
    """Raise a deterministic error from a blocking request handler."""
    del key, tp_size
    raise OSError("intentional blocking handler failure")


# ==============================================================================
# FREE_LOOKUP_LOCKS Request Handlers
# ==============================================================================


def free_locks_handler(key: KeyType, tp_size: int) -> None:
    """
    Dummy handler for FREE_LOOKUP_LOCKS requests.

    Args:
        key: Cache key whose read locks should be released
        tp_size: Tensor-parallel size for MLA
            multi-reader locking

    Returns:
        None
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(tp_size, int), f"Expected tp_size to be int, got {type(tp_size)}"


# ==============================================================================
# REPORT_BLOCK_ALLOCATION Request Handlers
# ==============================================================================


def report_block_allocations_handler(
    instance_id: int,
    model_name: str,
    records: list[BlockAllocationRecord],
) -> None:
    """
    Dummy handler for REPORT_BLOCK_ALLOCATION requests.

    Args:
        instance_id: The scheduler instance ID.
        model_name: The model name from the adapter.
        records: List of BlockAllocationRecord with per-request
            block and token allocation deltas.

    Returns:
        None
    """
    assert isinstance(records, list), (
        f"Expected records to be list, got {type(records)}"
    )
    for rec in records:
        assert isinstance(rec, BlockAllocationRecord), (
            f"Expected BlockAllocationRecord, got {type(rec)}"
        )
        assert isinstance(rec.req_id, str)
        assert isinstance(rec.new_block_ids, list)
        assert isinstance(rec.new_token_ids, list)
