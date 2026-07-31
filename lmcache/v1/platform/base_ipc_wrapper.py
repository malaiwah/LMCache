# SPDX-License-Identifier: Apache-2.0
"""Base class for device IPC wrappers.

:class:`DeviceIPCWrapper` is the abstract base for KV-cache IPC wrapper
implementations.  Every concrete wrapper (e.g. :class:`~.cpu.shm.CpuShmTensorWrapper`,
:class:`~.cuda.ipc_wrapper.CudaIPCWrapper`) subclasses it so they share
the single msgspec ext code (1) -- pickle preserves the concrete
subclass identity across the wire so ``to_tensor`` dispatches correctly
on the receiving side.

Refcounted exports follow a one-shot ownership contract: after export,
exactly one receiver must either import the handle once or explicitly release
it. A successful transport send transfers that obligation from sender to
receiver; failed or abandoned sends leave it with the sender.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator
from contextlib import ExitStack, nullcontext
from typing import Any, ContextManager, Tuple
import pickle
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger

logger = init_logger(__name__)


class DeviceIPCWrapper:
    """Base class for KV-cache IPC wrapper.

    Holds the device-agnostic mechanism shared by all transports: the
    interface fields (``dtype``/``shape``/``stride``/``storage_offset``/
    ``device_uuid``), UUID<->ordinal discovery via the ``torch_dev``
    abstraction, equality, and pickle-based (de)serialization.

    Every wire-level wrapper subclasses this so they share the single
    msgspec ext code (1) registered for ``DeviceIPCWrapper``: pickle
    preserves the concrete subclass identity across the wire so
    ``to_tensor`` dispatches correctly on the receiving side.

    Subclasses implement ``__init__`` (populate the interface fields from a
    tensor) and ``to_tensor`` (reconstruct the tensor from the handle).

    The default wrapper for each device is bound to that device's
    :class:`~lmcache.v1.platform.base_device_spec.DeviceSpec` via
    :attr:`~lmcache.v1.platform.base_device_spec.DeviceSpec.ipc_wrapper_cls`;
    :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory` reads that
    binding and returns the wrapper's ``wrap`` classmethod so callers can
    dispatch by ``tensor.device.type`` without any if/elif chain.
    """

    # Interface fields populated by each concrete subclass's
    # ``__init__``.  Declared here so the base-class ``__eq__`` (and
    # type-checkers) can see them; ``handle`` is intentionally typed as
    # ``Any`` because each backend stores a different opaque payload
    # (``tuple`` for CUDA shared-storage IPC, ``None`` for the SHM /
    # raw-CUDA paths that override ``to_tensor``).
    handle: Any
    dtype: torch.dtype
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    storage_offset: int
    device_uuid: str

    _discovered_device_mapping: dict[str, int] = {}
    _device_mapping_lock = threading.Lock()

    @classmethod
    def _get_device_uuid(cls, device_index: int) -> str:
        """Get the UUID of a device given its index."""
        return str(torch_dev.get_device_properties(device_index).uuid)

    @classmethod
    def _discover_devices(cls) -> None:
        """Discover all available accelerator devices and map their UUIDs
        to the physical device ordinals.
        """
        if not torch_dev.is_available():
            return

        num_devices = torch_dev.device_count()
        with DeviceIPCWrapper._device_mapping_lock:
            if DeviceIPCWrapper._discovered_device_mapping:
                return  # Already discovered

            for i in range(num_devices):
                device_uuid = cls._get_device_uuid(i)
                DeviceIPCWrapper._discovered_device_mapping[device_uuid] = i

    @classmethod
    def _get_device_index_from_uuid(cls, device_uuid: str) -> int:
        """Get the physical device ordinal from its UUID."""
        cls._discover_devices()

        with DeviceIPCWrapper._device_mapping_lock:
            device_index = DeviceIPCWrapper._discovered_device_mapping.get(
                device_uuid, None
            )

        if device_index is None:
            raise RuntimeError(
                f"Device UUID {device_uuid} not found in the discovered "
                "devices. Please make sure the process can see all the "
                "accelerator devices"
            )
        return device_index

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process from the IPC handle.

        Subclasses implement the transport-specific reconstruction.
        """
        raise NotImplementedError

    def release_ipc_export(self) -> bool:
        """Release an exported handle that will not be imported.

        Refcounted transports override this method. The default is a no-op
        because raw device handles and POSIX-SHM wrappers do not use PyTorch's
        CUDA IPC refcounter.

        Returns:
            ``True`` when this call released an export reservation.
        """
        return False

    def mark_ipc_export_transferred(self) -> bool:
        """Mark this process's export copy as owned by the receiver.

        Refcounted transports override this method. After a successful
        transport send, the sender must not release the same reservation that
        the receiver will import or explicitly release.

        Returns:
            ``True`` when ownership changed to the receiver.
        """
        return False

    def ipc_export_requires_transfer(self) -> bool:
        """Whether this wrapper participates in refcounted transfer claims."""
        return False

    def ipc_export_transfer_guard(self) -> ContextManager[Any]:
        """Return the lock/context guarding transfer validation and commit."""
        return nullcontext()

    def validate_ipc_export_transfer(self) -> None:
        """Raise unless a post-send ownership transfer can be committed."""

    def quarantine_ipc_export_after_send(self) -> None:
        """Prevent release after an accepted send with ambiguous bookkeeping."""

    def acquire_ipc_export_lease(self) -> bool:
        """Pin a one-shot export while asynchronous code takes ownership.

        Refcounted transports override this method.  A successful lease keeps
        handler-finally cleanup from releasing the reservation before an
        asynchronous sender has serialized and either sent or rejected it.

        Returns:
            ``True`` when this wrapper acquired a lease.
        """
        return False

    def release_ipc_export_lease(self) -> None:
        """Release one lease acquired by :meth:`acquire_ipc_export_lease`."""

    def __eq__(self, other: object) -> bool:
        # ``isinstance`` first so type-checkers can narrow ``other`` to
        # ``DeviceIPCWrapper`` before we touch its attributes; the
        # exact-type check that follows then enforces that, e.g., a
        # ``CudaIPCWrapper`` is never considered equal to a
        # ``RawCudaIPCWrapper`` even though they share a base class.
        if not isinstance(other, DeviceIPCWrapper):
            return False
        if type(self) is not type(other):
            return False
        return (
            self.handle == other.handle
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.stride == other.stride
            and self.storage_offset == other.storage_offset
            and self.device_uuid == other.device_uuid
        )

    @staticmethod
    def Serialize(obj: "DeviceIPCWrapper") -> bytes:
        """Pickle ``obj`` for the multiprocess wire.

        Pickle (rather than msgspec) is used so the concrete subclass
        identity round-trips: every wrapper shares the single msgspec
        ext code (1), and the receiver relies on the unpickled type to
        dispatch ``to_tensor`` correctly.

        Args:
            obj: The wrapper instance to serialize.

        Returns:
            The pickled bytes payload.
        """
        serialize_for_wire = getattr(obj, "_serialize_for_wire", None)
        if serialize_for_wire is not None:
            return serialize_for_wire()
        return pickle.dumps(obj)

    @staticmethod
    def Deserialize(data: bytes) -> "DeviceIPCWrapper":
        """Inverse of :meth:`Serialize`.

        Args:
            data: The pickled bytes payload produced by :meth:`Serialize`.

        Returns:
            The reconstructed wrapper instance, with its concrete
            subclass identity preserved.
        """
        return pickle.loads(data)


def _walk_ipc_wrappers(value: Any) -> Iterator[DeviceIPCWrapper]:
    """Yield IPC wrappers nested in protocol container payloads."""
    if isinstance(value, DeviceIPCWrapper):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_ipc_wrappers(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _walk_ipc_wrappers(item)


def _unique_ipc_wrappers(value: Any) -> Iterator[DeviceIPCWrapper]:
    """Yield each wrapper identity once, preserving traversal order."""
    seen: set[int] = set()
    for wrapper in _walk_ipc_wrappers(value):
        identity = id(wrapper)
        if identity in seen:
            continue
        seen.add(identity)
        yield wrapper


class IPCExportLease:
    """Explicit lifetime pin for asynchronously forwarded IPC exports.

    Callers must retain this object until the queued transport has either
    transferred or rejected every wrapper.  ``release`` is idempotent, and a
    destructor fallback prevents an abandoned pre-send lease from stranding a
    reservation indefinitely.
    """

    def __init__(self, wrappers: tuple[DeviceIPCWrapper, ...]) -> None:
        self._wrappers = wrappers
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        """Drop every wrapper lease exactly once."""
        with self._lock:
            if self._released:
                return
            self._released = True
            wrappers = self._wrappers
            self._wrappers = ()
        for wrapper in reversed(wrappers):
            try:
                wrapper.release_ipc_export_lease()
            except Exception:
                logger.exception(
                    "Failed to release IPC export lease for %s",
                    type(wrapper).__name__,
                )

    def __enter__(self) -> "IPCExportLease":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except BaseException:
            pass


def acquire_ipc_export_lease(value: Any) -> IPCExportLease:
    """Atomically lease all refcounted wrappers nested in ``value``.

    Partial acquisition is rolled back before propagating an error.  The
    returned object must stay alive until transport success/failure is known.
    """
    acquired: list[DeviceIPCWrapper] = []
    try:
        for wrapper in _unique_ipc_wrappers(value):
            if wrapper.acquire_ipc_export_lease():
                acquired.append(wrapper)
    except BaseException:
        IPCExportLease(tuple(acquired)).release()
        raise
    return IPCExportLease(tuple(acquired))


def wrap_ipc_tensors_rollback_safe(
    tensors: Any, wrapper_factory: Any
) -> list[DeviceIPCWrapper]:
    """Build an IPC wrapper batch and release a partial batch on failure."""
    wrappers: list[DeviceIPCWrapper] = []
    try:
        for tensor in tensors:
            wrappers.append(wrapper_factory(tensor))
    except BaseException:
        release_ipc_exports(wrappers)
        raise
    return wrappers


def release_ipc_exports(value: Any) -> None:
    """Best-effort release of every unconsumed export nested in a payload.

    Args:
        value: A wrapper or protocol container containing wrappers.
    """
    for wrapper in _unique_ipc_wrappers(value):
        try:
            wrapper.release_ipc_export()
        except Exception:
            # Cleanup must neither strand later wrappers in the same batch nor
            # replace the request/handler exception that led us here.
            logger.exception(
                "Failed to release IPC export for %s", type(wrapper).__name__
            )


def mark_ipc_exports_transferred(value: Any) -> None:
    """Transfer nested IPC export ownership after an atomic transport send.

    Args:
        value: A wrapper or protocol container containing wrappers.
    """
    for wrapper in _unique_ipc_wrappers(value):
        try:
            wrapper.mark_ipc_export_transferred()
        except Exception:
            # The bytes have already been accepted atomically by the transport,
            # so releasing here could race the receiver and double-decrement.
            logger.exception(
                "Failed to record IPC export transfer for %s",
                type(wrapper).__name__,
            )


def mark_ipc_exports_transferred_strict(value: Any) -> bool:
    """Atomically commit a refcounted wrapper batch after accepted send.

    All participating wrapper locks are acquired in stable identity order.
    Validation completes for the entire unique batch before the first state is
    changed. If validation/commit still fails unexpectedly, every participant
    is moved to a non-releasable quarantine: the transport already owns the
    bytes, so leaking until process teardown is safer than a double decrement.

    Returns:
        ``True`` for a fully recorded transfer, ``False`` for safe quarantine.
    """
    # This function runs only *after* an atomic transport send returned.  It is
    # therefore a noexcept boundary: propagating even an exotic wrapper/guard
    # failure would route the caller through ordinary pre-send cleanup and
    # could double-decrement a reservation now visible to the receiver.
    discovered: list[DeviceIPCWrapper] = []
    participants: list[DeviceIPCWrapper] = []
    try:
        for wrapper in _unique_ipc_wrappers(value):
            # Record the wrapper before invoking extension code so an
            # exception in ipc_export_requires_transfer() can still quarantine
            # the object locally.  The MQ layer additionally retains the full
            # payload process-lifetime whenever this function returns False.
            discovered.append(wrapper)
            if wrapper.ipc_export_requires_transfer():
                participants.append(wrapper)
        participants.sort(key=id)
        if not participants:
            return True

        with ExitStack() as stack:
            for wrapper in participants:
                stack.enter_context(wrapper.ipc_export_transfer_guard())
            for wrapper in participants:
                wrapper.validate_ipc_export_transfer()
            for wrapper in participants:
                if not wrapper.mark_ipc_export_transferred():
                    raise RuntimeError(
                        f"{type(wrapper).__name__} rejected validated transfer"
                    )
    except BaseException:
        try:
            logger.exception(
                "Accepted IPC send had ambiguous ownership bookkeeping; "
                "quarantining %d wrapper(s)",
                len(discovered),
            )
        except BaseException:
            # Logging must not turn the post-send noexcept boundary back into
            # an exception path during interpreter or logger teardown.
            pass
        for wrapper in discovered:
            try:
                wrapper.quarantine_ipc_export_after_send()
            except BaseException:
                try:
                    logger.exception(
                        "Failed to quarantine accepted IPC export for %s",
                        type(wrapper).__name__,
                    )
                except BaseException:
                    pass
        return False
    return True
