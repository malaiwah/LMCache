# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar, get_type_hints
import enum
import inspect
import itertools
import queue
import threading

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

logger = init_logger(__name__)

T = TypeVar("T")

# Internal type used for the client-server communication
RequestUID = int

_ERROR_RESPONSE_MARKER = b"LMCACHE_RPC_ERROR_V1"
_MAX_REMOTE_ERROR_MESSAGE_BYTES = 4096
_CLIENT_SNDHWM = 1000
_CLIENT_CONTROL_TIMEOUT_S = 5.0
_CLIENT_THREAD_JOIN_TIMEOUT_S = 5.0


class RemoteHandlerError(RuntimeError):
    """A request handler failed in the remote LMCache process."""

    def __init__(
        self, request_type: RequestType, error_type: str, message: str
    ) -> None:
        self.request_type = request_type
        self.error_type = error_type
        self.remote_message = message
        super().__init__(f"{request_type.name} handler failed: {error_type}: {message}")


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
        raise ValueError("Payload count does not match expected count")

    decoded_payloads = [
        msgspec_decode(payload, cls=cls)
        for payload, cls in zip(b_payloads, payload_clss, strict=False)
    ]
    return decoded_payloads


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
                            try:
                                owner.process_inbound()
                            except Exception:
                                logger.exception("Unhandled inbound client error")
        finally:
            # Drain remaining ops so waiting threads unblock, then close the
            # notifier here so even a delayed thread exit releases it safely.
            self._process_ops()
            self._notifier.close()
            logger.debug("ClientPollingLoop shut down")


# Main classes
class MessageQueueClient:
    @dataclass
    class WrappedRequest:
        request_uid: RequestUID
        future: MessagingFuture[Any]
        request_type: RequestType
        request_payloads: list[Any]

    def __init__(self, server_url: str, context: zmq.Context):
        self.ctx = context
        self.server_url = server_url
        self.socket = self._create_socket()

        # Input queue
        self.input_queue: queue.Queue = queue.Queue()

        # Pending job's futures
        self._request_counter = itertools.count()
        self.pending_futures: dict[int, MessagingFuture[Any]] = {}
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
        # A timed-out future is terminal. Reclaim it only here, on the polling
        # thread, so a late response cannot race a caller-side dict mutation.
        for request_uid, future in list(self.pending_futures.items()):
            if future.query():
                self.pending_futures.pop(request_uid, None)

        while True:
            try:
                wrapped_request = self.input_queue.get_nowait()
            except queue.Empty:
                return
            if wrapped_request.future.query():
                # The caller's deadline elapsed before this request left the
                # lifecycle-aware input queue.
                continue

            request_uid = wrapped_request.request_uid
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
                self.socket.send_multipart(
                    [b_request_uid, b_request_type] + b_payloads,
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                self.pending_futures.pop(request_uid, None)
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
                wrapped_request.future.set_exception(exc)
                logger.exception("Cannot send LMCache MQ request_uid=%d", request_uid)

    def process_inbound(self) -> None:
        """Process one inbound response from the server.

        Called by the shared ClientPollingLoop when the DEALER socket
        is readable.  Only touches ``pending_futures``, which is
        exclusively accessed from the loop thread.
        """
        msg = self.socket.recv_multipart()
        if len(msg) < 2:
            logger.error(
                "Malformed response: expected at least 2 message parts "
                "[request_uid, request_type, *response], got %d",
                len(msg),
            )
            return
        b_request_uid, b_request_type, *b_response = msg
        request_uid = msgspec_decode(b_request_uid, cls=RequestUID)
        request_type = msgspec_decode(b_request_type, cls=RequestType)
        response_cls = get_response_class(request_type)

        if request_uid in self.pending_futures:
            future = self.pending_futures.pop(request_uid)
            if b_response and b_response[0] == _ERROR_RESPONSE_MARKER:
                if len(b_response) != 2:
                    future.set_exception(
                        RuntimeError("Malformed LMCache RPC error response")
                    )
                    return
                try:
                    error = msgspec_decode(b_response[1], cls=_RemoteErrorPayload)
                except Exception:
                    future.set_exception(
                        RuntimeError("Failed to decode LMCache RPC error response")
                    )
                    logger.exception("Failed to decode RPC error response")
                    return
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
        self.input_queue.put(
            MessageQueueClient.WrappedRequest(
                request_uid=request_uid,
                future=future,
                request_type=request_type,
                request_payloads=request_payloads,
            )
        )
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
        """Fail pending and not-yet-sent work on the polling thread."""
        futures = list(self.pending_futures.values())
        self.pending_futures.clear()
        while True:
            try:
                futures.append(self.input_queue.get_nowait().future)
            except queue.Empty:
                break
        for future in futures:
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
        return self.handler(*unwrap_request_payloads(payloads, self.payload_clss))

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
        if isinstance(self.executor, AffinityThreadPool):
            return self.executor.submit(
                self.handler, *decoded_payloads, affinity_key=affinity_key
            )
        return self.executor.submit(self.handler, *decoded_payloads)

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

        # Registered handlers: request_type -> (payload_cls, handler)
        self.handlers: dict[RequestType, RequestHandlerBase[Any]] = {}

        # Thread pools assigned via add_normal_thread_pool / add_affinity_thread_pool
        self.extra_pools: list[ThreadPoolExecutor | AffinityThreadPool] = []

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

        def _notify_response(fut: Future):
            try:
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
            inbound_state = socks.get(self.socket, None)
            outbound_state = socks.get(output_fd, None)

            # Process the incoming requests
            if inbound_state and inbound_state & zmq.POLLIN:
                msg = self.socket.recv_multipart()
                assert len(msg) >= 3, (
                    "Expected at least 3 message parts "
                    "[identity, request_uid, request_type, *payloads]"
                )

                identity, b_request_uid, b_request_type, *payloads = msg
                request_type = msgspec_decode(b_request_type, cls=RequestType)

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

    def close(self) -> None:
        self.is_finished.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join()
        self.socket.close()
        for pool in self.extra_pools:
            pool.shutdown(wait=False)
        self._output_efd.close()
