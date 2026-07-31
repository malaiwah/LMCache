# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC wrapper implementations.

:class:`CudaIPCWrapper` handles tensors backed by PyTorch's caching
allocator (vLLM default).  :class:`RawCudaIPCWrapper` handles tensors
allocated outside PyTorch (e.g. TRT-LLM's ``cudaMalloc``'d pool).

:class:`CudaIPCWrapper` is bound to ``device_type="cuda"`` via
:attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`, so the
multiprocess adapter dispatches to it via
:func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.
:class:`RawCudaIPCWrapper` is not exposed on the spec -- callers (the
TRT-LLM adapter) instantiate it directly.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, ClassVar, cast
import threading

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.platform.base_ipc_wrapper import DeviceIPCWrapper

logger = init_logger(__name__)


class CudaIPCWrapper(DeviceIPCWrapper):
    """One-shot PyTorch CUDA IPC export with process-local ownership state.

    ``_share_cuda_`` creates one producer-refcount reservation. The receiving
    process must consume it through :meth:`to_tensor` exactly once or return it
    through :meth:`release_ipc_export`; repeated calls are idempotent or reuse
    the cached tensor.
    """

    #: ``torch.device.type`` this wrapper handles. Kept as a class-level
    #: constant so external tooling / tests can introspect the binding.
    device_type: ClassVar[str] = "cuda"
    _UNCONSUMED: ClassVar[str] = "unconsumed"
    _IMPORTED: ClassVar[str] = "imported"
    _RELEASED: ClassVar[str] = "released"
    _TRANSFERRED: ClassVar[str] = "transferred"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "CudaIPCWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: A CUDA tensor backed by PyTorch's caching allocator.

        Returns:
            A new :class:`CudaIPCWrapper` wrapping ``tensor`` for the
            multiprocess wire.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Permute any non-contiguous view (e.g. vLLM's NHD-over-HND) so the
        # shape/stride we encode across IPC reflects the physical layout.
        # Offset is preserved by the wrapper's storage_offset field.
        tensor = cast(torch.Tensor, attempt_permute_to_contiguous_view(tensor))

        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self._initialize_ipc_ownership(auto_release=False)
        try:
            self.dtype = tensor.dtype
            self.shape = tuple(tensor.shape)
            self.stride = tuple(tensor.stride())
            self.storage_offset = int(tensor.storage_offset())

            device_index = tensor.device.index
            self.device_uuid = self._get_device_uuid(device_index)
        except BaseException:
            # Once _share_cuda_ succeeds, this constructor owns one producer
            # reservation even if later metadata discovery fails.
            self.release_ipc_export()
            raise

    def _initialize_ipc_ownership(self, *, auto_release: bool) -> None:
        """Initialize process-local state excluded from the wire payload."""
        self._ipc_state = self._UNCONSUMED
        self._ipc_state_lock = threading.Lock()
        self._cached_tensor: torch.Tensor | None = None
        # A deserialized receiver owns the reservation and must release it if
        # the handler never imports it. The sender transfers ownership only
        # after the atomic transport send succeeds.
        self._auto_release = auto_release

    def __getstate__(self) -> dict[str, Any]:
        """Serialize only the CUDA handle metadata, never local ownership state."""
        with self._ipc_state_lock:
            if self._ipc_state != self._UNCONSUMED:
                raise RuntimeError(
                    "Cannot serialize a CUDA IPC wrapper after its export was "
                    f"{self._ipc_state}"
                )
            return {
                "handle": self.handle,
                "dtype": self.dtype,
                "shape": self.shape,
                "stride": self.stride,
                "storage_offset": self.storage_offset,
                "device_uuid": self.device_uuid,
            }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore wire metadata and make this process the reservation owner."""
        self.handle = state["handle"]
        self.dtype = state["dtype"]
        self.shape = state["shape"]
        self.stride = state["stride"]
        self.storage_offset = state["storage_offset"]
        self.device_uuid = state["device_uuid"]
        self._initialize_ipc_ownership(auto_release=True)

    def _import_tensor_from_handle(self) -> torch.Tensor:
        """Import the one consumer storage represented by this wrapper."""
        storage: torch.UntypedStorage | None = None
        try:
            device_index = self._get_device_index_from_uuid(self.device_uuid)
            # Allocate the tensor shell before opening the shared storage. If
            # either step fails before ``storage`` is assigned, no storage
            # deleter exists and we must return the reservation explicitly.
            tensor = torch.empty(
                (), device=f"{torch_device_type}:{device_index}", dtype=self.dtype
            )
            storage = self._open_shared_storage(device_index)
            tensor.set_(storage, self.storage_offset, self.shape, self.stride)
            return tensor
        except BaseException:
            if storage is None:
                self._release_counter_noexcept()
            # Once storage exists, its deleter owns the one decrement even if
            # rebuilding the tensor view fails.
            raise

    def _open_shared_storage(self, device_index: int) -> torch.UntypedStorage:
        """Open the CUDA storage and attach its consumer-side refcounter."""
        return torch.UntypedStorage._new_shared_cuda(  # noqa: SLF001
            device_index, *self.handle[1:]
        )

    def _release_counter(self) -> None:
        """Decrement the producer counter without importing the CUDA storage."""
        torch.UntypedStorage._release_ipc_counter_cuda(  # noqa: SLF001
            self.handle[4], self.handle[5]
        )

    def _release_counter_noexcept(self) -> None:
        """Best-effort counter release that cannot mask the primary failure."""
        try:
            self._release_counter()
        except Exception:
            logger.exception("Failed to release CUDA IPC producer refcounter")

    def to_tensor(self) -> torch.Tensor:
        """Import and cache the single consumer tensor represented by this handle.

        PyTorch initializes each ``_share_cuda_`` refcounter to one. Re-importing
        the same handle would attach multiple storage deleters to that one
        reservation and underflow it during teardown, so repeated and concurrent
        calls return the same process-local tensor.

        Note:
            This function may break if the accelerator is not initialized.
            We should call ``torch_dev.init()`` before using this function
            (guarded by hasattr since not all backends expose init()).

        Returns:
            The imported tensor. Repeated calls return the same object.

        Raises:
            RuntimeError: If ownership was already released or transferred.
        """
        with self._ipc_state_lock:
            if self._ipc_state == self._IMPORTED:
                assert self._cached_tensor is not None
                return self._cached_tensor
            if self._ipc_state != self._UNCONSUMED:
                raise RuntimeError(
                    "CUDA IPC export is no longer available for import "
                    f"(state={self._ipc_state})"
                )

            try:
                tensor = self._import_tensor_from_handle()
            except BaseException:
                # _import_tensor_from_handle either returned the reservation
                # explicitly or attached it to a storage deleter.
                self._ipc_state = self._RELEASED
                raise

            self._cached_tensor = tensor
            self._ipc_state = self._IMPORTED
            return tensor

    def release_ipc_export(self) -> bool:
        """Release this reservation if no consumer tensor was imported.

        Returns:
            ``True`` for the call that claimed the reservation; ``False`` if
            another terminal action already won.
        """
        with self._ipc_state_lock:
            if self._ipc_state != self._UNCONSUMED:
                return False
            self._ipc_state = self._RELEASED
            self._release_counter_noexcept()
            return True

    def mark_ipc_export_transferred(self) -> bool:
        """Relinquish this process's copy after a successful transport send.

        Returns:
            ``True`` for the call that transferred ownership; ``False`` if
            another terminal action already won.
        """
        with self._ipc_state_lock:
            if self._ipc_state != self._UNCONSUMED:
                return False
            self._ipc_state = self._TRANSFERRED
            self._auto_release = False
            return True

    def __del__(self) -> None:
        """Best-effort fallback for a receiver that never consumed its handle."""
        if not getattr(self, "_auto_release", False):
            return
        try:
            self.release_ipc_export()
        except BaseException:
            # Destructors run during exception unwinding and interpreter
            # shutdown; cleanup failures must not mask the primary failure.
            pass


class RawCudaIPCWrapper(DeviceIPCWrapper):
    """IPC wrapper for CUDA tensors allocated outside PyTorch's caching
    allocator.

    PyTorch's ``UntypedStorage._share_cuda_()`` only works for tensors
    backed by its own caching allocator. TRT-LLM publishes its KV pool
    via ``at::for_blob`` over a ``cudaMalloc``'d buffer, which raises in
    ``_share_cuda_()``. This subclass bypasses that path: it calls
    ``cudaIpcGetMemHandle`` on the raw data pointer, then reconstructs
    the tensor on the receiving side via ``cudaIpcOpenMemHandle`` plus
    a CuPy ``UnownedMemory`` → DLPack → ``torch`` round-trip.

    Sharing the ``DeviceIPCWrapper`` base (rather than introducing a
    parallel class with its own msgspec ext code) is load-bearing —
    msgspec does not support unions of custom ext-encoded types. With a
    common base, ``KVCache = list[DeviceIPCWrapper]`` type-checks, the
    single ext code 1 round-trips every wrapper, and pickle preserves
    the concrete subclass identity through the wire so ``to_tensor``
    dispatches correctly.
    """

    #: Same ``torch.device.type`` as ``CudaIPCWrapper``, but not exposed
    #: on :attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`
    #: -- callers (TRT-LLM adapter) instantiate it directly.
    device_type: ClassVar[str] = "cuda"

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import assert_contiguous

        assert_contiguous(tensor)

        try:
            # Third Party
            from cuda.bindings import runtime as cudart
        except ImportError:
            # Third Party
            from cuda import cudart

        data_ptr = tensor.data_ptr()
        err, ipc_handle = cudart.cudaIpcGetMemHandle(data_ptr)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaIpcGetMemHandle failed: {err} (ptr=0x{data_ptr:x})"
            )

        # Store only what's needed for reconstruction.
        self._ipc_handle_reserved = bytes(ipc_handle.reserved)
        self._nbytes = tensor.untyped_storage().nbytes()

        # DeviceIPCWrapper interface fields. ``handle`` is unused —
        # ``to_tensor`` is overridden to bypass it — but kept (None) so
        # the base-class equality check has a value to compare.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via raw CUDA IPC."""
        # Third Party
        import cupy

        try:
            # Third Party
            from cuda.bindings import runtime as cudart
        except ImportError:
            # Third Party
            from cuda import cudart

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        handle = cudart.cudaIpcMemHandle_t()
        handle.reserved = self._ipc_handle_reserved
        err, ptr = cudart.cudaIpcOpenMemHandle(
            handle, cudart.cudaIpcMemLazyEnablePeerAccess
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed: {err}")

        # Wrap as a flat ``uint8`` CuPy array, DLPack to torch, then view
        # as the original dtype/shape. ``uint8`` avoids dtype-conversion
        # gaps (bfloat16, fp8 have no direct CuPy/NumPy equivalent without
        # ml_dtypes).
        with cupy.cuda.Device(device_index):
            mem = cupy.cuda.UnownedMemory(ptr, self._nbytes, owner=self)
            memptr = cupy.cuda.MemoryPointer(mem, 0)
            cp_flat = cupy.ndarray(self._nbytes, dtype=cupy.uint8, memptr=memptr)

        raw = torch.from_dlpack(cp_flat)
        return raw.view(self.dtype).reshape(self.shape)
