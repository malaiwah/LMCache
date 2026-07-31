# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar, get_type_hints
import enum
import inspect
import itertools
import queue
import threading
import time

# Third Party
import msgspec
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.custom_types import (
    DeviceIPCWrapper,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.futures import (
    MessagingFuture,
)
from lmcache.v1.multiprocess.protocol import (
    HandlerType,
    RequestType,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.platform import EventNotifier, create_event_notifier
from lmcache.v1.platform.base_ipc_wrapper import (
    IPCExportLease,
    acquire_ipc_export_lease,
    mark_ipc_exports_transferred_strict,
    release_ipc_exports,
    snapshot_ipc_exports,
)

logger = init_logger(__name__)

T = TypeVar("T")

# Internal type used for the client-server communication
RequestUID = int

_ERROR_RESPONSE_MARKER = b"LMCACHE_RPC_ERROR_V1"
_MAX_REMOTE_ERROR_MESSAGE_BYTES = 4096
_CLIENT_SNDHWM = 1000
_CLIENT_CONTROL_TIMEOUT_S = 5.0
_CLIENT_THREAD_JOIN_TIMEOUT_S = 5.0
_SERVER_CLOSE_TIMEOUT_S = 30.0

# A sent request that loses its transport session has no cancellation/consume
# acknowledgement. Releasing its CUDA IPC reservation could race a daemon that
# received the bytes just before disconnect. Retain the payload (and therefore
# producer tensor) for process lifetime instead. This is deliberately a safe
# quarantine, not a leak-free reset protocol.
_SENT_UNANSWERED_IPC_QUARANTINE: list[Any] = []
_SENT_UNANSWERED_IPC_QUARANTINE_LOCK = threading.Lock()
# Completion events and other future-owned transport resources are not IPC
# export ownership records. Keep their outage quarantine separate so ordinary
# requests never manufacture an empty ``_IPCTransportOwnership`` merely to pin
# an event.
_SENT_UNANSWERED_TRANSPORT_QUARANTINE: list[Any] = []
_SENT_UNANSWERED_TRANSPORT_QUARANTINE_LOCK = threading.Lock()


def _quarantine_sent_unanswered_ipc_exports(value: Any) -> None:
    """Retain sent-but-unacknowledged exports until process teardown."""
    with _SENT_UNANSWERED_IPC_QUARANTINE_LOCK:
        if any(retained is value for retained in _SENT_UNANSWERED_IPC_QUARANTINE):
            return
        _SENT_UNANSWERED_IPC_QUARANTINE.append(value)


def _remove_provisional_ipc_quarantine(value: Any) -> bool:
    """Remove the newest identity-matching provisional quarantine entry.

    The sender installs this lifetime pin immediately before ``send_multipart``.
    It is removed only after either a rejected send (the producer releases the
    reservation) or a fully committed ownership transfer.  A failed post-send
    commit deliberately leaves it behind for process lifetime.
    """
    with _SENT_UNANSWERED_IPC_QUARANTINE_LOCK:
        for index in range(len(_SENT_UNANSWERED_IPC_QUARANTINE) - 1, -1, -1):
            if _SENT_UNANSWERED_IPC_QUARANTINE[index] is value:
                del _SENT_UNANSWERED_IPC_QUARANTINE[index]
                return True
    return False


def _quarantine_sent_unanswered_transport_resources(value: Any) -> None:
    """Retain non-IPC transport resources after an ambiguous session loss."""
    with _SENT_UNANSWERED_TRANSPORT_QUARANTINE_LOCK:
        _SENT_UNANSWERED_TRANSPORT_QUARANTINE.append(value)


def _snapshot_request_payloads(values: list[Any]) -> tuple[Any, ...]:
    """Detach built-in protocol containers from caller mutation.

    Device wrappers and opaque leaf objects retain identity; list/dict/tuple/
    set containers (the same graph shapes traversed by IPC ownership helpers)
    are recursively copied. The polling thread serializes only this private
    graph, and ownership is captured from it once at submission.
    """
    in_progress = object()
    memo: dict[int, Any] = {}

    def clone(value: Any) -> Any:
        if isinstance(value, DeviceIPCWrapper):
            return value
        identity = id(value)
        if identity in memo:
            cloned = memo[identity]
            if cloned is in_progress:
                raise ValueError("Recursive immutable MQ payload container")
            return cloned
        if isinstance(value, list):
            cloned_list: list[Any] = []
            memo[identity] = cloned_list
            cloned_list.extend(clone(item) for item in value)
            return cloned_list
        if isinstance(value, dict):
            cloned_dict: dict[Any, Any] = {}
            memo[identity] = cloned_dict
            for key, item in value.items():
                cloned_dict[key] = clone(item)
            return cloned_dict
        if isinstance(value, tuple):
            memo[identity] = in_progress
            cloned_tuple = tuple(clone(item) for item in value)
            memo[identity] = cloned_tuple
            return cloned_tuple
        if isinstance(value, set):
            cloned_set: set[Any] = set()
            memo[identity] = cloned_set
            for item in value:
                cloned_set.add(clone(item))
            return cloned_set
        if isinstance(value, frozenset):
            memo[identity] = in_progress
            cloned_frozenset = frozenset(clone(item) for item in value)
            memo[identity] = cloned_frozenset
            return cloned_frozenset
        return value

    return tuple(clone(value) for value in values)


@dataclass(frozen=True)
class _IPCTransportOwnership:
    """Immutable wrapper identity set and its asynchronous lifetime lease."""

    exports: tuple[DeviceIPCWrapper, ...]
    lease: IPCExportLease


class RemoteHandlerError(RuntimeError):
    """A request handler failed in the remote LMCache process."""

    def __init__(
        self, request_type: RequestType, error_type: str, message: str
    ) -> None:
        self.request_type = request_type
        self.error_type = error_type
        self.remote_message = message
        super().__init__(f"{request_type.name} handler failed: {error_type}: {message}")

    def _lmcache_traceback_free_clone(self) -> "RemoteHandlerError":
        """Rebuild the typed remote error without traceback-owned frames."""
        return type(self)(self.request_type, self.error_type, self.remote_message)


class _RetireResponseSession(RuntimeError):
    """A response cannot be correlated safely; replace the DEALER session."""


class _RemoteErrorPayload(msgspec.Struct, frozen=True):
    error_type: str
    message: str


# Helper functions
def encode_request_uid(uid: RequestUID) -> bytes:
    return msgspec.msgpack.encode(uid)


def decode_request_uid(b_uid: bytes) -> RequestUID:
    return msgspec.msgpack.decode(b_uid, type=RequestUID)


def unwrap_request_payloads(
    b_payloads: list[bytes], payload_clss: list[Any]
) -> list[Any]:
    if len(b_payloads) != len(payload_clss):
        _decode_and_release_known_wire_payloads(b_payloads, payload_clss)
        raise ValueError("Payload count does not match expected count")

    decoded_payloads: list[Any] = []
    try:
        for payload, cls in zip(b_payloads, payload_clss, strict=False):
            decoded_payloads.append(msgspec_decode(payload, cls=cls))
    except BaseException:
        # A later payload can fail after a CUDA wrapper was already decoded.
        # Deterministically return every still-unconsumed export reservation.
        release_ipc_exports(decoded_payloads)
        raise
    return decoded_payloads


def _decode_and_release_known_wire_payloads(
    b_payloads: list[bytes], payload_clss: list[Any]
) -> None:
    """Best-effort consume/release wrappers in rejected wire payloads.

    A sender transfers CUDA IPC ownership when ZeroMQ accepts the multipart
    message. Even when the receiver has no handler, rejects the frame count, or
    cannot decode the request type, every frame must be decoded exactly once so
    one-shot wrappers can explicitly return the producer reservation.

    Rejection cleanup intentionally ignores the declared schema and uses the
    customized ``Any`` decoder for *all* frames. A typed decoder can materialize
    a wrapper and then fail on a later field; retrying that same frame as Any
    would create a second receiver for one refcounter reservation.
    """
    del payload_clss  # Kept in the signature for compatibility with callers.
    decoded_payloads: list[Any] = []
    generic_decoder = get_customized_decoder(Any)
    for payload in b_payloads:
        try:
            decoded_payloads.append(generic_decoder.decode(payload))
        except BaseException:
            logger.exception("Failed to decode rejected MQ payload frame")
    release_ipc_exports(decoded_payloads)


def _invoke_handler_with_ipc_cleanup(
    handler: Callable[..., Any], decoded_payloads: list[Any]
) -> Any:
    """Run a handler and release any IPC wrappers it did not import."""
    try:
        return handler(*decoded_payloads)
    finally:
        release_ipc_exports(decoded_payloads)


class _BlockingPayloadCleanupOwner:
    """Exactly-once owner for decoded payloads queued to an executor.

    The MQ thread has already materialized receiver-owned IPC exports before
    ``submit``. A queued task may be cancelled without ever invoking its
    callable, so cleanup cannot live only in the callable's ``finally`` block.
    The cancellation callback and task entry race for this owner: cancellation
    may release only ``pending`` payloads, while a started task transitions the
    owner to ``running`` and becomes the sole cleanup path.
    """

    def __init__(self, decoded_payloads: list[Any]) -> None:
        self._decoded_payloads: list[Any] | None = decoded_payloads
        self._state = "pending"
        self._lock = threading.Lock()

    def invoke(self, handler: Callable[..., Any]) -> Any:
        """Transfer cleanup ownership to this worker and run ``handler``."""
        with self._lock:
            if self._state != "pending" or self._decoded_payloads is None:
                raise RuntimeError("Blocking payload cleanup owner is unavailable")
            self._state = "running"
            decoded_payloads = self._decoded_payloads

        try:
            return handler(*decoded_payloads)
        finally:
            self._release_running(decoded_payloads)

    def release_if_pending(self) -> bool:
        """Release a cancelled/rejected task only if no worker started it."""
        with self._lock:
            if self._state != "pending" or self._decoded_payloads is None:
                return False
            decoded_payloads = self._decoded_payloads
            self._decoded_payloads = None
            self._state = "released"
        release_ipc_exports(decoded_payloads)
        return True

    def _release_running(self, decoded_payloads: list[Any]) -> None:
        with self._lock:
            if self._state != "running":
                return
            self._decoded_payloads = None
            self._state = "released"
        release_ipc_exports(decoded_payloads)


_SPECIAL_ENCODER_DECODERS = {
    DeviceIPCWrapper: (
        get_customized_encoder(DeviceIPCWrapper),
        get_customized_decoder(DeviceIPCWrapper),
    ),
    list[DeviceIPCWrapper]: (
        get_customized_encoder(list[DeviceIPCWrapper]),
        get_customized_decoder(list[DeviceIPCWrapper]),
    ),
    MemoryLayoutDesc: (
        get_customized_encoder(MemoryLayoutDesc),
        get_customized_decoder(MemoryLayoutDesc),
    ),
}


def msgspec_encode(obj: Any, cls: Any) -> bytes:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        encoder, _ = _SPECIAL_ENCODER_DECODERS[cls]
        return encoder.encode(obj)
    # Defensive guard: coerce obj to the declared cls so that
    # e.g. a bool passed as int (or vice-versa) is encoded in the
    # wire format that msgspec_decode expects for that cls.
    if cls in (bool, int):
        obj = cls(obj)
    return msgspec.msgpack.encode(obj)


def msgspec_decode(b_obj: bytes, cls: Any) -> Any:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        _, decoder = _SPECIAL_ENCODER_DECODERS[cls]
        return decoder.decode(b_obj)
    # Defensive guard: msgspec strict-validates wire format
    # (bool ≠ int in msgpack), but runtime type may not match
    # declared cls. Decode untyped, then coerce.
    if cls in (bool, int):
        return cls(msgspec.msgpack.decode(b_obj))
    return msgspec.msgpack.decode(b_obj, type=cls)


# Shared polling loop for MessageQueueClient instances


class _OpKind(enum.Enum):
    REGISTER = "register"
    UNREGISTER = "unregister"
    RESET = "reset"


@dataclass
class _PollOp:
    kind: _OpKind
    client: "MessageQueueClient"
    completion: Future[None]
    abandoned: bool = False
    state_lock: threading.Lock = field(default_factory=threading.Lock)


class ClientPollingLoop:
    """Singleton polling loop shared by all MessageQueueClient instances.

    Instead of each client running its own daemon thread and zmq.Poller,
    a single loop polls all clients' DEALER sockets and dispatches
    inbound/outbound work.

    Use ``get_instance()`` / ``release_instance()`` for lifecycle
    management — the loop starts lazily on first client and stops
    automatically when the last client releases.
    """

    _instance: "ClientPollingLoop | None" = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._ref_count: int = 0
        self._is_finished = threading.Event()
        self._notifier: EventNotifier = create_event_notifier()
        self._ops_queue: queue.Queue[_PollOp] = queue.Queue()
        self._poller = zmq.Poller()
        self._poller.register(self._notifier.fileno(), zmq.POLLIN)
        self._socket_to_client: dict[zmq.Socket, "MessageQueueClient"] = {}
        self._thread = threading.Thread(
            target=self._main_loop, daemon=True, name="mq-client-shared-loop"
        )
        self._thread.start()

    @classmethod
    def get_instance(cls) -> "ClientPollingLoop":
        """Get or create the singleton, incrementing the ref count.

        Returns:
            ClientPollingLoop: The shared polling loop instance.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ClientPollingLoop()
            cls._instance._ref_count += 1
            return cls._instance

    @classmethod
    def release_instance(cls) -> None:
        """Decrement the ref count; tear down the loop when it reaches 0."""
        with cls._instance_lock:
            inst = cls._instance
            if inst is None:
                return
            inst._ref_count -= 1
            if inst._ref_count > 0:
                return
            inst._is_finished.set()
            inst._notifier.notify()
            cls._instance = None

        inst._thread.join(timeout=_CLIENT_THREAD_JOIN_TIMEOUT_S)
        if inst._thread.is_alive():
            # The notifier must outlive a thread that may still resume and use
            # it. The polling thread closes its notifier when it eventually
            # exits.
            logger.error(
                "ClientPollingLoop did not stop within %.1fs; "
                "deferring notifier close to polling thread",
                _CLIENT_THREAD_JOIN_TIMEOUT_S,
            )
            return

    def _queue_op(
        self, kind: _OpKind, client: "MessageQueueClient"
    ) -> tuple[bool, bool]:
        completion: Future[None] = Future()
        op = _PollOp(kind=kind, client=client, completion=completion)
        self._ops_queue.put(op)
        self._notifier.notify()
        try:
            completion.result(timeout=_CLIENT_CONTROL_TIMEOUT_S)
        except FutureTimeoutError:
            with op.state_lock:
                # Resolve the deadline race with _process_ops. If it completed
                # while result() raised, consume that result as normal.
                if completion.done():
                    completion.result()
                    return True, False
                op.abandoned = True
                # An unclaimed REGISTER can be cancelled and closed by its
                # constructor. UNREGISTER and RESET must remain queued so the
                # polling thread eventually stops using the old socket before
                # it is closed with zero linger.
                cancelled = completion.cancel() if kind is _OpKind.REGISTER else False
            logger.error(
                "Timed out after %.1fs waiting to %s MessageQueueClient",
                _CLIENT_CONTROL_TIMEOUT_S,
                kind.value,
            )
            return False, cancelled
        return True, False

    def register(self, client: "MessageQueueClient") -> None:
        """Register a client without allowing construction to hang forever."""
        completed, cancelled = self._queue_op(_OpKind.REGISTER, client)
        if not completed:
            if cancelled:
                client._close_socket()
            raise TimeoutError("Timed out registering MessageQueueClient")

    def unregister(self, client: "MessageQueueClient") -> bool:
        """Unregister a client, returning false rather than hanging shutdown."""
        completed, _cancelled = self._queue_op(_OpKind.UNREGISTER, client)
        return completed

    def reset(self, client: "MessageQueueClient") -> bool:
        """Replace one client's socket without blocking the shared loop."""
        completed, _cancelled = self._queue_op(_OpKind.RESET, client)
        return completed

    def notify(self) -> None:
        """Wake the polling loop to process outbound tasks."""
        self._notifier.notify()

    def _retire_client(self, client: "MessageQueueClient") -> None:
        """Remove a client from this poller before closing its socket."""
        try:
            if client.socket in self._socket_to_client:
                self._poller.unregister(client.socket)
        finally:
            self._socket_to_client.pop(client.socket, None)
            client._fail_outstanding("LMCache MQ client closed")
            client._close_socket()

    def _reset_client(self, client: "MessageQueueClient") -> None:
        """Discard one transport session and register a fresh DEALER socket."""
        old_socket = client.socket
        try:
            if old_socket in self._socket_to_client:
                self._poller.unregister(old_socket)
        finally:
            self._socket_to_client.pop(old_socket, None)

        client._fail_outstanding(
            "LMCache MQ request abandoned after server became unhealthy"
        )
        client._replace_socket()
        self._poller.register(client.socket, zmq.POLLIN)
        self._socket_to_client[client.socket] = client

    def _process_ops(self) -> None:
        """Drain queued control operations and wake every waiting caller."""
        while True:
            try:
                op = self._ops_queue.get_nowait()
            except queue.Empty:
                return
            if not op.completion.set_running_or_notify_cancel():
                continue
            try:
                if op.kind is _OpKind.REGISTER:
                    with op.state_lock:
                        abandoned = op.abandoned
                    if not abandoned:
                        self._poller.register(op.client.socket, zmq.POLLIN)
                        self._socket_to_client[op.client.socket] = op.client
                        logger.debug("Registered client socket %s", op.client.socket)
                    with op.state_lock:
                        if op.abandoned:
                            self._retire_client(op.client)
                        op.completion.set_result(None)
                elif op.kind is _OpKind.UNREGISTER:
                    self._retire_client(op.client)
                    with op.state_lock:
                        op.completion.set_result(None)
                    logger.debug("Unregistered client socket %s", op.client.socket)
                elif op.kind is _OpKind.RESET:
                    self._reset_client(op.client)
                    with op.state_lock:
                        op.completion.set_result(None)
                    logger.debug("Reset client socket %s", op.client.socket)
            except Exception as exc:
                op.client._close_socket()
                with op.state_lock:
                    op.completion.set_exception(exc)

    def _main_loop(self) -> None:
        """Unified poll loop for all registered clients."""
        notifier_fd = self._notifier.fileno()

        try:
            while not self._is_finished.is_set():
                self._process_ops()

                socks = dict(self._poller.poll(1000))

                # Outbound: shared notifier woke us — drain it, then flush
                # all clients' output queues.
                if socks.get(notifier_fd) and socks[notifier_fd] & zmq.POLLIN:
                    self._notifier.consume()
                    # Operations queued while poll() slept (especially RESET)
                    # must take effect before any old-session outbound work.
                    self._process_ops()
                    for client in self._socket_to_client.values():
                        try:
                            client.process_outbound_task()
                        except Exception:
                            # One bad socket/client must not kill the singleton loop
                            # and strand every other client's futures/control ops.
                            logger.exception("Unhandled outbound client error")

                # Inbound: dispatch each ready DEALER socket to its client.
                for sock, event in socks.items():
                    if sock is notifier_fd:
                        continue
                    if event & zmq.POLLIN:
                        owner = self._socket_to_client.get(sock)
                        if owner is not None:
                            self._process_inbound_client(owner)
        finally:
            # Drain remaining ops so waiting threads unblock, then close the
            # notifier here so even a delayed thread exit releases it safely.
            self._process_ops()
            self._notifier.close()
            logger.debug("ClientPollingLoop shut down")

    def _process_inbound_client(self, client: "MessageQueueClient") -> None:
        """Dispatch one response and retire an uncorrelatable session."""
        try:
            client.process_inbound()
        except _RetireResponseSession as exc:
            error_message = str(exc)
            logger.error(
                "Retiring malformed LMCache MQ response session: %s", error_message
            )
            try:
                self._reset_client(client)
            except Exception:
                logger.exception("Failed to retire malformed response session")
        except Exception:
            logger.exception("Unhandled inbound client error")


# Main classes
class MessageQueueClient:
    @dataclass
    class WrappedRequest:
        request_uid: RequestUID
        future: MessagingFuture[Any]
        request_type: RequestType
        request_payloads: tuple[Any, ...] | list[Any]
        ipc_ownership: _IPCTransportOwnership | None = None

        def __post_init__(self) -> None:
            # Compatibility for lightweight tests/callers that construct this
            # internal record directly rather than through submit_request().
            if self.ipc_ownership is None:
                exports = snapshot_ipc_exports(self.request_payloads)
                if exports:
                    self.ipc_ownership = _IPCTransportOwnership(
                        exports, acquire_ipc_export_lease(exports)
                    )

    def __init__(self, server_url: str, context: zmq.Context):
        self.ctx = context
        self.server_url = server_url
        self.socket = self._create_socket()

        # Input queue
        self.input_queue: queue.Queue = queue.Queue()

        # Pending job's futures
        self._request_counter = itertools.count()
        self.pending_futures: dict[int, MessagingFuture[Any]] = {}
        self._pending_request_types: dict[int, RequestType] = {}
        self._inflight_ownership: dict[int, _IPCTransportOwnership] = {}
        self._closed = False
        self._socket_closed = False
        self._socket_close_lock = threading.Lock()

        # Register with the shared polling loop.
        self._polling_loop = ClientPollingLoop.get_instance()
        try:
            self._polling_loop.register(self)
        except Exception:
            ClientPollingLoop.release_instance()
            raise

    def process_outbound_task(self) -> None:
        # Some tests and compatibility callers allocate lightweight clients
        # through __new__; initialize the additive tracking map lazily too.
        if not hasattr(self, "_inflight_ownership"):
            self._inflight_ownership = {}
        if not hasattr(self, "_pending_request_types"):
            self._pending_request_types = {}
        # Reclaim only transport-complete futures here. A caller timeout after
        # send is terminal to the caller but the remote side may still open,
        # wait on, or re-record CUDA IPC handles embedded in the request.
        for request_uid, future in list(self.pending_futures.items()):
            if future.transport_complete:
                self.pending_futures.pop(request_uid, None)
                self._pending_request_types.pop(request_uid, None)
                self._inflight_ownership.pop(request_uid, None)

        while True:
            try:
                wrapped_request = self.input_queue.get_nowait()
            except queue.Empty:
                return
            ownership = wrapped_request.ipc_ownership
            if wrapped_request.future.query():
                # The caller's deadline elapsed before this request left the
                # lifecycle-aware input queue. The remote never saw its
                # transport resources, so they are safe to release now.
                if ownership is not None:
                    release_ipc_exports(ownership.exports)
                    ownership.lease.release()
                wrapped_request.future.complete_transport()
                continue

            request_uid = wrapped_request.request_uid
            provisionally_quarantined = False
            try:
                b_request_uid = msgspec_encode(request_uid, cls=RequestUID)
                b_request_type = msgspec_encode(
                    wrapped_request.request_type, cls=RequestType
                )
                payload_classes = get_payload_classes(wrapped_request.request_type)
                if len(payload_classes) != len(wrapped_request.request_payloads):
                    expected_classes = [cls.__name__ for cls in payload_classes]
                    actual_classes = [
                        type(p).__name__ for p in wrapped_request.request_payloads
                    ]
                    raise ValueError(
                        f"Payload count mismatch for "
                        f"{wrapped_request.request_type}: "
                        f"expected {len(payload_classes)} payloads "
                        f"{expected_classes}, "
                        f"got {len(wrapped_request.request_payloads)} payloads "
                        f"{actual_classes}. "
                        f"This is likely caused by a version mismatch between "
                        f"the lmcache client and lmcache server."
                    )

                b_payloads = [
                    msgspec_encode(payload, cls=cls)
                    for payload, cls in zip(
                        wrapped_request.request_payloads,
                        payload_classes,
                        strict=False,
                    )
                ]
                # Register immediately before the atomic nonblocking send so a
                # fast response cannot race pending-future publication.
                self.pending_futures[request_uid] = wrapped_request.future
                self._pending_request_types[request_uid] = wrapped_request.request_type
                if ownership is not None:
                    self._inflight_ownership[request_uid] = ownership
                    # Establish a strong, process-lifetime producer pin
                    # *before* the atomic send. If any post-send ownership
                    # bookkeeping is ambiguous, the pin stays installed even
                    # after a fast reply removes ordinary in-flight tracking.
                    _quarantine_sent_unanswered_ipc_exports(ownership)
                    provisionally_quarantined = True
                self.socket.send_multipart(
                    [b_request_uid, b_request_type] + b_payloads,
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                self.pending_futures.pop(request_uid, None)
                self._pending_request_types.pop(request_uid, None)
                self._inflight_ownership.pop(request_uid, None)
                if ownership is not None:
                    if provisionally_quarantined:
                        _remove_provisional_ipc_quarantine(ownership)
                    release_ipc_exports(ownership.exports)
                    ownership.lease.release()
                wrapped_request.future.set_exception(
                    RuntimeError("LMCache MQ send queue is full; server is unreachable")
                )
                logger.error(
                    "Nonblocking LMCache MQ send failed for request_uid=%d: "
                    "send queue full",
                    request_uid,
                )
            except Exception as exc:
                self.pending_futures.pop(request_uid, None)
                self._pending_request_types.pop(request_uid, None)
                self._inflight_ownership.pop(request_uid, None)
                if ownership is not None:
                    if provisionally_quarantined:
                        _remove_provisional_ipc_quarantine(ownership)
                    release_ipc_exports(ownership.exports)
                    ownership.lease.release()
                wrapped_request.future.set_exception(exc)
                # Do not hand a traceback-owning exception to asynchronous
                # log handlers: that would retain this frame, the client, and
                # its payload graph independently of the sanitized future.
                error_message = str(exc)
                logger.error(
                    "Cannot send LMCache MQ request_uid=%d: %s: %s",
                    request_uid,
                    type(exc).__name__,
                    error_message,
                )
            else:
                if ownership is None:
                    continue
                # send_multipart is atomic: after it returns, the receiver owns
                # every one-shot IPC export embedded in the request.  The
                # strict helper is itself noexcept, but keep this boundary
                # defensive because releasing on any post-send exception would
                # race the receiver and could double-decrement the CUDA IPC
                # refcounter.
                try:
                    transfer_recorded = mark_ipc_exports_transferred_strict(
                        ownership.exports
                    )
                except BaseException:
                    transfer_recorded = False
                    try:
                        logger.exception(
                            "Accepted LMCache MQ send escaped ownership "
                            "bookkeeping for request_uid=%d; retaining producer "
                            "payload for process lifetime",
                            request_uid,
                        )
                    except BaseException:
                        pass
                if transfer_recorded:
                    try:
                        _remove_provisional_ipc_quarantine(ownership)
                    except BaseException:
                        # Failure to remove only leaks the already-transferred
                        # producer payload; it must never enter pre-send
                        # release cleanup after an accepted send.
                        try:
                            logger.exception(
                                "Failed to remove committed IPC quarantine "
                                "for request_uid=%d",
                                request_uid,
                            )
                        except BaseException:
                            pass
                else:
                    try:
                        logger.error(
                            "Accepted LMCache MQ send has ambiguous IPC ownership "
                            "for request_uid=%d; retaining producer payload for "
                            "process lifetime",
                            request_uid,
                        )
                    except BaseException:
                        pass
                try:
                    if transfer_recorded:
                        ownership.lease.release()
                    # On ambiguity, retain the ownership record *with its live
                    # lease* process-lifetime. This suppresses deferred handler
                    # cleanup even if a wrapper-specific quarantine hook failed
                    # or strict discovery stopped before visiting later exports.
                except BaseException:
                    # Lease cleanup failures are also non-fatal post-send. The
                    # accepted transport must remain eligible for its reply.
                    try:
                        logger.exception(
                            "Failed to release accepted-send IPC lease for "
                            "request_uid=%d",
                            request_uid,
                        )
                    except BaseException:
                        pass

    def process_inbound(self) -> None:
        """Process one inbound response from the server.

        Called by the shared ClientPollingLoop when the DEALER socket
        is readable.  Only touches ``pending_futures``, which is
        exclusively accessed from the loop thread.
        """
        if not hasattr(self, "_inflight_ownership"):
            self._inflight_ownership = {}
        if not hasattr(self, "_pending_request_types"):
            self._pending_request_types = {}
        msg = self.socket.recv_multipart()
        if not msg:
            raise _RetireResponseSession(
                "empty response header cannot identify a pending request"
            )

        b_request_uid, *remaining = msg
        try:
            request_uid = decode_request_uid(b_request_uid)
        except Exception as exc:
            raise _RetireResponseSession(
                f"invalid response request UID ({type(exc).__name__}: {exc})"
            ) from None

        if not remaining:
            self._terminalize_response_error(
                request_uid,
                "Malformed LMCache MQ response: truncated header after request UID",
            )
            raise _RetireResponseSession(
                f"truncated response header for request_uid={request_uid}"
            )

        b_request_type, *b_response = remaining
        try:
            request_type = msgspec_decode(b_request_type, cls=RequestType)
        except Exception as exc:
            correlated = self._terminalize_response_error(
                request_uid,
                "Malformed LMCache MQ response request type: "
                f"{type(exc).__name__}: {exc}",
            )
            if not correlated:
                raise _RetireResponseSession(
                    "invalid response request type for unknown "
                    f"request_uid={request_uid}"
                ) from None
            return

        expected_type = self._pending_request_types.get(request_uid)
        if expected_type is not None and request_type is not expected_type:
            self._terminalize_response_error(
                request_uid,
                "Malformed LMCache MQ response request type mismatch: "
                f"expected {expected_type.name}, got {request_type.name}",
            )
            return

        response_cls = get_response_class(request_type)

        if request_uid in self.pending_futures:
            future = self.pending_futures[request_uid]
            try:
                if b_response and b_response[0] == _ERROR_RESPONSE_MARKER:
                    if len(b_response) != 2:
                        raise RuntimeError("Malformed LMCache RPC error response")
                    error = msgspec_decode(b_response[1], cls=_RemoteErrorPayload)
                    future.set_exception(
                        RemoteHandlerError(
                            request_type=request_type,
                            error_type=error.error_type,
                            message=error.message,
                        )
                    )
                elif b_response:
                    response = msgspec_decode(b_response[0], cls=response_cls)
                    future.set_result(response)
                else:
                    future.set_result(None)
            except Exception as exc:
                # A response proves the remote transport is finished even if
                # this client cannot decode it. Resolve the future with the
                # decode error instead of dropping its only tracking entry and
                # leaving it permanently unresolved.
                error_message = str(exc)
                logger.error(
                    "Failed to decode LMCache MQ response for request_uid=%d: %s: %s",
                    request_uid,
                    type(exc).__name__,
                    error_message,
                )
                # Never store the caught exception itself: its traceback owns
                # this process_inbound frame, which owns ``self`` and would
                # keep the complete MessageQueueClient graph alive as long as
                # callers retain the failed future.
                future.set_exception(
                    RuntimeError(
                        "Failed to decode LMCache MQ response: "
                        f"{type(exc).__name__}: {exc}"
                    )
                )
            finally:
                future.complete_transport()
                self.pending_futures.pop(request_uid, None)
                self._pending_request_types.pop(request_uid, None)
                self._inflight_ownership.pop(request_uid, None)

    def _terminalize_response_error(
        self, request_uid: RequestUID, message: str
    ) -> bool:
        """Fail one correlatable response and release its ordinary tracking."""
        future = self.pending_futures.get(request_uid)
        if future is None:
            logger.error("%s (unknown request_uid=%d)", message, request_uid)
            return False
        future.set_exception(RuntimeError(message))
        future.complete_transport()
        self.pending_futures.pop(request_uid, None)
        self._pending_request_types.pop(request_uid, None)
        self._inflight_ownership.pop(request_uid, None)
        return True

    def submit_request(
        self,
        request_type: RequestType,
        request_payloads: list[Any],
        response_cls: Optional[T] = None,
    ) -> MessagingFuture[T]:
        """Submit a request to the server.

        Args:
            request_type (RequestType): The type of the request.
            request_payloads (list[Any]): The payloads of the request.
            response_cls (Optional[T]): The expected response class.
                This should be get from `get_response_class(request_type)`.

        Returns:
            MessagingFuture[T]: A future that will hold the response.
        """
        request_uid = next(self._request_counter)
        future: MessagingFuture[T] = MessagingFuture(
            on_timeout=self._polling_loop.notify
        )
        # Detach the complete built-in container graph before leasing or
        # encoding it. The polling thread and immutable export tuple then refer
        # to the same private snapshot even if the caller clears/reorders a
        # payload list while send_multipart accepts the already-encoded frames.
        transport_payloads = _snapshot_request_payloads(request_payloads)
        ipc_exports = snapshot_ipc_exports(transport_payloads)
        ipc_ownership: _IPCTransportOwnership | None = None
        if ipc_exports:
            # Lease at submission—not later in the polling thread. A
            # forwarding handler may return immediately after enqueueing; its
            # finally-cleanup must not release wrappers before serialization.
            ipc_ownership = _IPCTransportOwnership(
                ipc_exports, acquire_ipc_export_lease(ipc_exports)
            )
        try:
            self.input_queue.put(
                MessageQueueClient.WrappedRequest(
                    request_uid=request_uid,
                    future=future,
                    request_type=request_type,
                    request_payloads=transport_payloads,
                    ipc_ownership=ipc_ownership,
                )
            )
        except BaseException:
            if ipc_ownership is not None:
                release_ipc_exports(ipc_ownership.exports)
                ipc_ownership.lease.release()
            raise
        self._polling_loop.notify()
        return future

    def reset_connection(self) -> bool:
        """Discard queued work after a server outage and reconnect.

        The reset runs on the shared polling thread. Closing the old DEALER
        with zero linger disposes requests buffered by ZeroMQ, while a fresh
        socket identity prevents late responses from matching new work.

        Returns:
            bool: True when the reset completed within the control timeout.
        """
        if self._closed:
            return False
        return self._polling_loop.reset(self)

    def _create_socket(self) -> zmq.Socket:
        """Create a configured DEALER for the current server URL."""
        socket = self.ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.SNDHWM, _CLIENT_SNDHWM)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.server_url)
        return socket

    def _fail_outstanding(self, message: str) -> None:
        """Fail work while preserving unsafe-to-release sent exports.

        Pending requests were atomically accepted by ZeroMQ, but reset/close
        provides no proof that the daemon did not receive them. Their payloads
        are retained in a process-lifetime quarantine. Only requests still in
        ``input_queue`` are known unsent and may be released.
        """
        inflight_ownership = getattr(self, "_inflight_ownership", {})
        pending_request_types = getattr(self, "_pending_request_types", {})
        sent_futures = list(self.pending_futures.items())
        for request_uid, future in sent_futures:
            ownership = inflight_ownership.get(request_uid)
            if ownership is not None:
                _quarantine_sent_unanswered_ipc_exports(ownership)
            retained = future.quarantine_transport_resources()
            if retained:
                _quarantine_sent_unanswered_transport_resources(retained)
        self.pending_futures.clear()
        pending_request_types.clear()
        inflight_ownership.clear()
        unsent_futures: list[MessagingFuture[Any]] = []
        while True:
            try:
                wrapped_request = self.input_queue.get_nowait()
            except queue.Empty:
                break
            ownership = wrapped_request.ipc_ownership
            if ownership is not None:
                release_ipc_exports(ownership.exports)
                ownership.lease.release()
            unsent_futures.append(wrapped_request.future)
        for _request_uid, future in sent_futures:
            future.set_exception(ConnectionError(message))
        for future in unsent_futures:
            future.set_exception(ConnectionError(message))

    def _replace_socket(self) -> None:
        """Close the unregistered socket and install a fresh one."""
        with self._socket_close_lock:
            self.socket.close(linger=0)
            self.socket = self._create_socket()
            self._socket_closed = False

    def _close_socket(self) -> None:
        """Close only after this socket is absent from the shared poller."""
        with self._socket_close_lock:
            if self._socket_closed:
                return
            self._socket_closed = True
            self.socket.close(linger=0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._polling_loop.unregister(self)
        except Exception:
            logger.exception("Failed to unregister MessageQueueClient")
        finally:
            # unregister() owns socket retirement, even after a timeout. This
            # caller may return while the live polling thread still references
            # the client, so it must not close the socket itself.
            ClientPollingLoop.release_instance()


ResponseType = TypeVar("ResponseType", covariant=True)
StateType = TypeVar("StateType", covariant=True)


class RequestHandlerBase(Generic[ResponseType]):
    def __call__(self, payloads: list[bytes]):
        raise NotImplementedError

    def get_response_class(self) -> ResponseType:
        raise NotImplementedError

    def get_handler_type(self) -> HandlerType:
        raise NotImplementedError


class SyncRequestHandler(RequestHandlerBase[ResponseType]):
    """
    The handler for those "fast" functions that can be executed in the main loop
    """

    def __init__(
        self,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.payload_clss = payload_clss
        self.response_cls = response_cls
        self.handler = handler

    def __call__(self, payloads: list[bytes]) -> ResponseType:
        decoded_payloads = unwrap_request_payloads(payloads, self.payload_clss)
        return _invoke_handler_with_ipc_cleanup(self.handler, decoded_payloads)

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.SYNC


class BlockingRequestHandler(RequestHandlerBase[ResponseType]):
    """
    Returns the future of the response.

    The ``executor`` field is initially ``None`` and must be assigned via
    :meth:`MessageQueueServer.add_normal_thread_pool` or
    :meth:`MessageQueueServer.add_affinity_thread_pool` before the server
    is started.
    """

    def __init__(
        self,
        payload_clss: list[Any],
        response_cls: ResponseType,
        handler: Callable[..., ResponseType],
    ):
        self.executor: ThreadPoolExecutor | AffinityThreadPool | None = None
        self.payload_clss = payload_clss
        self.handler = handler
        self.response_cls = response_cls

    def __call__(
        self, payloads: list[bytes], affinity_key: int = 0
    ) -> Future[ResponseType]:
        assert self.executor is not None, (
            "BlockingRequestHandler has no executor assigned. "
            "Call add_normal_thread_pool or add_affinity_thread_pool first."
        )
        decoded_payloads = unwrap_request_payloads(payloads, self.payload_clss)
        cleanup_owner = _BlockingPayloadCleanupOwner(decoded_payloads)
        try:
            if isinstance(self.executor, AffinityThreadPool):
                future = self.executor.submit(
                    cleanup_owner.invoke,
                    self.handler,
                    affinity_key=affinity_key,
                )
            else:
                future = self.executor.submit(cleanup_owner.invoke, self.handler)
        except BaseException:
            # The worker never took ownership when task submission failed.
            cleanup_owner.release_if_pending()
            raise

        # ThreadPoolExecutor and AffinityThreadPool both complete cancelled
        # futures without invoking the queued callable. Install this callback
        # before returning the future to the server's shutdown tracker so a
        # concurrent close cannot observe/cancel it first.
        def _release_cancelled_payloads(completed: Future[Any]) -> None:
            if completed.cancelled():
                cleanup_owner.release_if_pending()

        future.add_done_callback(_release_cancelled_payloads)
        return future

    def get_response_class(self) -> ResponseType:
        return self.response_cls

    def get_handler_type(self) -> HandlerType:
        return HandlerType.BLOCKING


class NonBlockingRequestHandler(Generic[ResponseType, StateType]):
    """
    The handler for the "fire and probe" functions that launch async tasks
    and have special mechanism to probe the task status.

    It requires 2 callables as the input:
    - the first one is to launch the async task. This function should return
        a 'state handle' that can be used to probe the task status later.
    - the second one is to probe the task status and get the return value
        with the 'state handle' returned by the first function.
    """

    # TODO: implement this in the future versions if needed
    pass


class MessageQueueServer:
    def __init__(self, bind_url: str, context: zmq.Context):
        # Socket
        self.ctx = context
        self.socket = self.ctx.socket(zmq.ROUTER)
        self.socket.bind(bind_url)
        # Use a cross-platform Notifier instead of zmq PUSH/PULL sockets
        # because blocking handler callbacks run on ThreadPoolExecutor
        # threads, and zmq sockets are not thread-safe. Notifier.notify()
        # is atomic (eventfd on Linux, self-pipe elsewhere).
        self._output_efd = create_event_notifier()
        self.output_queue: queue.Queue = queue.Queue()

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self._output_efd.fileno(), zmq.POLLIN)

        # Main loop thread
        self.is_finished = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._main_loop, daemon=True, name="mq-server-thread"
        )
        # Linearizes the last possible recv() against close()'s stop-intake
        # boundary without holding a lock while a handler executes.
        self._intake_lock = threading.Lock()

        # Registered handlers: request_type -> (payload_cls, handler)
        self.handlers: dict[RequestType, RequestHandlerBase[Any]] = {}

        # Thread pools assigned via add_normal_thread_pool / add_affinity_thread_pool
        self.extra_pools: list[ThreadPoolExecutor | AffinityThreadPool] = []
        self._handler_futures: set[Future[Any]] = set()
        self._handler_futures_cv = threading.Condition()
        self._close_lock = threading.Lock()
        self._closed = False

    def _queue_error_response(
        self, prefix_frames: list[bytes], exception: BaseException
    ) -> None:
        """Return a bounded error response without touching ZMQ off-thread."""
        message = str(exception).encode("utf-8")[:_MAX_REMOTE_ERROR_MESSAGE_BYTES]
        payload = _RemoteErrorPayload(
            error_type=type(exception).__name__,
            message=message.decode("utf-8", errors="replace"),
        )
        self.output_queue.put(
            prefix_frames
            + [
                _ERROR_RESPONSE_MARKER,
                msgspec_encode(payload, cls=_RemoteErrorPayload),
            ]
        )
        self._output_efd.notify()

    def _call_sync_handler(
        self,
        handler_entry: SyncRequestHandler[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        """
        Call the sync handler and send the response back to the client.

        Args:
            handler_entry (SyncRequestHandler[Any]): The handler entry.
            payloads (list[bytes]): The payloads of the request.
            prefix_frames (list[bytes]): The prefix frames to send back.
        """
        response = handler_entry(payloads)
        response_cls = handler_entry.get_response_class()
        b_response = msgspec_encode(response, cls=response_cls)
        if response is not None:
            self.socket.send_multipart(prefix_frames + [b_response])
        else:
            self.socket.send_multipart(prefix_frames)

    def _call_blocking_handler(
        self,
        handler_entry: BlockingRequestHandler[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        """
        Call the blocking handler in a separate thread and send the response
        back to the client.

        Args:
            handler_entry (BlockingRequestHandler[Any]): The handler entry.
            payloads (list[bytes]): The payloads of the request.
            prefix_frames (list[bytes]): The prefix frames to send back.
                prefix_frames[0] is the zmq identity used as affinity key.
        """
        affinity_key = hash(prefix_frames[0])
        future = handler_entry(payloads, affinity_key=affinity_key)
        with self._handler_futures_cv:
            self._handler_futures.add(future)

        def _notify_response(fut: Future):
            try:
                if self.is_finished.is_set():
                    return
                response = fut.result()
                response_cls = handler_entry.get_response_class()
                b_response = msgspec_encode(response, cls=response_cls)
                frames_to_send = (
                    prefix_frames + [b_response]
                    if response is not None
                    else prefix_frames
                )

                self.output_queue.put(frames_to_send)
                self._output_efd.notify()

            except Exception as exc:
                logger.exception("Error in blocking handler")
                self._queue_error_response(prefix_frames, exc)
            finally:
                with self._handler_futures_cv:
                    self._handler_futures.discard(fut)
                    self._handler_futures_cv.notify_all()

        future.add_done_callback(_notify_response)

    def _call_handler(
        self,
        handler_entry: RequestHandlerBase[Any],
        payloads: list[bytes],
        prefix_frames: list[bytes],
    ) -> Any:
        match handler_entry.get_handler_type():
            case HandlerType.SYNC:
                assert isinstance(handler_entry, SyncRequestHandler)
                self._call_sync_handler(handler_entry, payloads, prefix_frames)
            case HandlerType.BLOCKING:
                assert isinstance(handler_entry, BlockingRequestHandler)
                self._call_blocking_handler(handler_entry, payloads, prefix_frames)
            case HandlerType.NON_BLOCKING:
                raise NotImplementedError("Non-blocking handler is not supported yet")
            case _:
                raise ValueError("Unknown handler type")

    def _main_loop(self):
        output_fd = self._output_efd.fileno()
        while not self.is_finished.is_set():
            socks = dict(self.poller.poll(1000))
            # close() sets the flag before waking poll(). Never accept another
            # request from the readiness snapshot observed during shutdown.
            if self.is_finished.is_set():
                break
            inbound_state = socks.get(self.socket, None)
            outbound_state = socks.get(output_fd, None)

            # Process the incoming requests
            if inbound_state and inbound_state & zmq.POLLIN:
                with self._intake_lock:
                    if self.is_finished.is_set():
                        break
                    msg = self.socket.recv_multipart()
                if len(msg) < 3:
                    logger.error(
                        "Malformed MQ request: expected at least 3 frames "
                        "[identity, request_uid, request_type, *payloads], got %d",
                        len(msg),
                    )
                    continue

                identity, b_request_uid, b_request_type, *payloads = msg
                try:
                    request_type = msgspec_decode(b_request_type, cls=RequestType)
                except Exception as exc:
                    # Version skew can make a newer enum value unknown to this
                    # daemon. The sender already transferred every IPC frame
                    # after ZeroMQ accepted the multipart message, so routing
                    # rejection must still materialize/release them exactly
                    # once and must not terminate the serving loop.
                    logger.exception("Cannot decode LMCache MQ request type")
                    _decode_and_release_known_wire_payloads(payloads, [])
                    self._queue_error_response(
                        [identity, b_request_uid, b_request_type], exc
                    )
                    continue

                if handler_entry := self.handlers.get(request_type):
                    try:
                        self._call_handler(
                            handler_entry=handler_entry,
                            payloads=payloads,
                            prefix_frames=[identity, b_request_uid, b_request_type],
                        )
                    except Exception as exc:
                        logger.exception("Error handling request %s", request_type)
                        self._queue_error_response(
                            [identity, b_request_uid, b_request_type], exc
                        )
                else:
                    logger.error(
                        "No handler registered for request type %s", request_type
                    )
                    logger.error("Available handlers: %s", list(self.handlers.keys()))
                    _decode_and_release_known_wire_payloads(payloads, [])
                    self._queue_error_response(
                        [identity, b_request_uid, b_request_type],
                        RuntimeError(
                            "No handler registered for request type "
                            f"{request_type.name}"
                        ),
                    )

            # Send the responses
            if outbound_state and outbound_state & zmq.POLLIN:
                # Consume the notifier counter (resets atomically)
                self._output_efd.consume()

                # Process the output tasks
                try:
                    while frames_to_send := self.output_queue.get_nowait():
                        self.socket.send_multipart(frames_to_send)
                except queue.Empty:
                    pass

    def _inspect_handler_signature(self, request_type: RequestType, handler) -> bool:
        """Inspect the handler signature to ensure it matches the expected
        payload classes.

        Args:
            handler (callable): The handler function.

        Returns:
            bool: True if the signature matches, False otherwise.
        """

        def same_type(a, b) -> bool:
            if a is None:
                a = type(None)
            if b is None:
                b = type(None)
            return a == b

        sig = inspect.signature(handler)
        hints = get_type_hints(handler)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        payload_clss = get_payload_classes(request_type)
        if len(params) != len(payload_clss):
            logger.error(
                "Handler for %s expects %d arguments, but got %d",
                request_type,
                len(payload_clss),
                len(params),
            )
            return False

        for i, (param, expected_cls) in enumerate(
            zip(params, payload_clss, strict=False)
        ):
            ann = hints.get(param.name, param.annotation)
            if not same_type(ann, expected_cls):
                logger.error(
                    "Handler for %s argument %d expects type %s, but got %s",
                    request_type,
                    i,
                    expected_cls,
                    ann,
                )
                return False

        return_ann = hints.get("return", sig.return_annotation)
        expected_return_cls = get_response_class(request_type)
        if not same_type(return_ann, expected_return_cls):
            logger.error(
                "Handler for %s expects return type %s, but got %s",
                request_type,
                expected_return_cls,
                return_ann,
            )
            return False
        return True

    def add_handler(
        self,
        request_type: RequestType,
        payload_clss: list[Any],
        handler_type: HandlerType,
        handler,
    ) -> None:
        """Register a handler for a specific request type.

        Args:
            request_type (RequestType): The type of the request to handle.
            payload_clss (list[Any]): The expected payload classes for the request.
                This should be get from `get_payload_classes(request_type)`.
            handler (callable): The handler function that takes the payloads
                as arguments.
        """
        if not self._inspect_handler_signature(request_type, handler):
            raise ValueError(
                f"Handler signature does not match for request type: {request_type}"
            )

        match handler_type:
            case HandlerType.SYNC:
                self.add_sync_handler(request_type, payload_clss, handler)
            case HandlerType.BLOCKING:
                self.add_blocking_handler(request_type, payload_clss, handler)
            case HandlerType.NON_BLOCKING:
                raise NotImplementedError("Non-blocking handler is not supported yet")
            case _:
                raise ValueError(f"Unknown handler type: {handler_type}")

    def add_sync_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = SyncRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_blocking_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        response_cls = get_response_class(request_type)
        self.handlers[request_type] = BlockingRequestHandler(
            payload_clss, response_cls, handler
        )

    def add_nonblocking_handler(
        self, request_type: RequestType, payload_clss: list[Any], handler
    ) -> None:
        raise NotImplementedError

    def _validate_blocking_handlers(
        self,
        request_types: list[RequestType],
        method_name: str,
    ) -> None:
        """Validate that all request types are registered BlockingRequestHandlers."""
        for request_type in request_types:
            handler = self.handlers.get(request_type)
            if handler is None:
                raise ValueError(
                    f"No handler registered for request type: {request_type}. "
                    f"Register handlers before calling {method_name}."
                )
            if not isinstance(handler, BlockingRequestHandler):
                raise TypeError(
                    f"Handler for {request_type} is "
                    f"{type(handler).__name__}, not BlockingRequestHandler. "
                    f"Only blocking handlers can use thread pools."
                )

    def add_normal_thread_pool(
        self,
        request_types: list[RequestType],
        max_workers: int,
    ) -> None:
        """Assign a ThreadPoolExecutor to specific request types.

        Use this for non-GPU blocking handlers (e.g. LOOKUP, END_SESSION).

        Must be called after the handlers are registered (via add_handler /
        add_blocking_handler) and before start().  Each request_type must
        already be registered as a BlockingRequestHandler; otherwise a
        ValueError or TypeError is raised.

        Args:
            request_types: The request types that should use this pool.
            max_workers: Number of worker threads in the pool.
        """
        self._validate_blocking_handlers(request_types, "add_normal_thread_pool")
        if not request_types:
            return

        pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"normal-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created normal thread pool (max_workers=%d) for request types: %s",
            max_workers,
            [rt.name for rt in request_types],
        )

    def add_affinity_thread_pool(
        self,
        request_types: list[RequestType],
        max_workers: int,
    ) -> None:
        """Assign an AffinityThreadPool to specific request types.

        Use this for GPU-bound blocking handlers (e.g. STORE, RETRIEVE).
        Requests from the same zmq client identity are always dispatched
        to the same worker thread, eliminating the need for per-instance
        GPU transfer locks.

        Must be called after the handlers are registered (via add_handler /
        add_blocking_handler) and before start().

        Args:
            request_types: The request types that should use this pool.
            max_workers: Number of worker threads in the pool.
        """
        self._validate_blocking_handlers(request_types, "add_affinity_thread_pool")
        if not request_types:
            return

        pool = AffinityThreadPool(
            max_workers=max_workers,
            thread_name_prefix=f"affinity-pool-{len(self.extra_pools)}",
        )
        self.extra_pools.append(pool)
        for request_type in request_types:
            handler = self.handlers[request_type]
            assert isinstance(handler, BlockingRequestHandler)
            handler.executor = pool

        logger.debug(
            "Created affinity thread pool (max_workers=%d) for request types: %s",
            max_workers,
            [rt.name for rt in request_types],
        )

    def start(self):
        # Validate all blocking handlers have an executor assigned
        for rt, handler in self.handlers.items():
            if isinstance(handler, BlockingRequestHandler) and handler.executor is None:
                raise RuntimeError(
                    f"BlockingRequestHandler for {rt} has no thread pool "
                    f"assigned. Call add_normal_thread_pool or "
                    f"add_affinity_thread_pool before start()."
                )
        self.worker_thread.start()

    def close(self, handler_timeout_s: float = _SERVER_CLOSE_TIMEOUT_S) -> None:
        """Stop intake, drain/cancel handlers, then retire transport resources.

        A timeout deliberately leaves the socket and notifier open and raises:
        callers must not tear down cache/storage resources under a live handler.
        Once the blocked handler finishes, ``close`` may be retried safely.
        """
        if handler_timeout_s < 0:
            raise ValueError("handler_timeout_s must be non-negative")
        with self._close_lock:
            if self._closed:
                return
            deadline = time.monotonic() + handler_timeout_s

            # Stop accepting new work first. Wake poll() so this phase is
            # bounded even when no socket is active.
            with self._intake_lock:
                self.is_finished.set()
            if self.worker_thread.is_alive():
                self._output_efd.notify()
                self.worker_thread.join(timeout=max(0.0, deadline - time.monotonic()))
                if self.worker_thread.is_alive():
                    raise TimeoutError(
                        "Timed out stopping LMCache MQ request intake; "
                        "transport resources remain open"
                    )

            # Cancel queued work while allowing already-running handlers to
            # finish. Pool callbacks remove futures only after they can no
            # longer touch the response notifier.
            for pool in self.extra_pools:
                pool.shutdown(wait=False, cancel_futures=True)

            with self._handler_futures_cv:
                for future in tuple(self._handler_futures):
                    future.cancel()
                while self._handler_futures:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Timed out draining LMCache MQ handlers; "
                            "transport resources remain open"
                        )
                    self._handler_futures_cv.wait(timeout=remaining)

            # Every task and done callback is now quiescent, so joins are
            # immediate and no handler can race socket/notifier/cache teardown.
            for pool in self.extra_pools:
                pool.shutdown(wait=True, cancel_futures=True)
            self.socket.close()
            self._output_efd.close()
            self._closed = True
