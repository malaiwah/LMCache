# SPDX-License-Identifier: Apache-2.0
# Standard
from multiprocessing.synchronize import Event as EventClass
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock
import gc
import itertools
import multiprocessing as mp
import queue
import sys
import threading
import time
import weakref

# Third Party
import msgspec
import pytest
import torch
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.mq import (
    BlockingRequestHandler,
    ClientPollingLoop,
    MessageQueueClient,
    MessageQueueServer,
    RemoteHandlerError,
)
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.server import add_handler_helper
from lmcache.v1.platform.base_ipc_wrapper import (
    DeviceIPCWrapper,
    release_ipc_exports,
)
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

# Test helpers
from tests.v1.multiprocess import test_mq_handler_helpers
import lmcache.v1.multiprocess.mq as mq_mod

# ==============================================================================
# MessageQueueServer and MessageQueueClient Tests Infrastructure
# ==============================================================================


class _TrackedIPCWrapper(DeviceIPCWrapper):
    """CPU-only ownership probe for MQ send and teardown paths."""

    def __init__(
        self,
        failure_phase: str | None = None,
        producer_resource: Any = None,
    ) -> None:
        self.state = "unconsumed"
        self.release_calls = 0
        self.transfer_calls = 0
        self.lease_count = 0
        self.release_pending = False
        self.lock = threading.RLock()
        self.failure_phase = failure_phase
        # Model CudaIPCWrapper._producer_tensor: quarantine must keep this
        # allocation alive even after a fast response drops in-flight state.
        self.producer_resource = producer_resource

    def release_ipc_export(self) -> bool:
        with self.lock:
            if self.state != "unconsumed":
                return False
            if self.lease_count:
                self.release_pending = True
                return False
            self.state = "released"
            self.release_calls += 1
            return True

    def mark_ipc_export_transferred(self) -> bool:
        with self.lock:
            if self.failure_phase == "mark_false":
                return False
            if self.failure_phase == "mark_raise":
                raise RuntimeError("injected mark failure")
            if self.state != "unconsumed":
                return False
            self.state = "transferred"
            self.release_pending = False
            self.transfer_calls += 1
            return True

    def acquire_ipc_export_lease(self) -> bool:
        with self.lock:
            if self.state != "unconsumed":
                raise RuntimeError("not leasable")
            self.lease_count += 1
            return True

    def release_ipc_export_lease(self) -> None:
        with self.lock:
            self.lease_count -= 1
            if self.lease_count == 0 and self.release_pending:
                self.state = "released"
                self.release_pending = False
                self.release_calls += 1

    def ipc_export_requires_transfer(self) -> bool:
        if self.failure_phase == "requires_raise":
            raise RuntimeError("injected participant-discovery failure")
        return True

    def ipc_export_transfer_guard(self) -> Any:
        if self.failure_phase == "guard_enter_raise":

            class _FailingGuard:
                def __enter__(self) -> None:
                    raise RuntimeError("injected guard-enter failure")

                def __exit__(self, *_exc: object) -> None:
                    return None

            return _FailingGuard()
        return self.lock

    def validate_ipc_export_transfer(self) -> None:
        if self.state != "unconsumed":
            raise RuntimeError("not transferable")

    def quarantine_ipc_export_after_send(self) -> None:
        self.state = "quarantined"
        self.release_pending = False


class _LegacyTrackedIPCWrapper(DeviceIPCWrapper):
    """Refcounted legacy extension predating the explicit capability probe."""

    def __init__(self) -> None:
        self.state = "unconsumed"
        self.release_calls = 0
        self.transfer_calls = 0

    def release_ipc_export(self) -> bool:
        if self.state != "unconsumed":
            return False
        self.state = "released"
        self.release_calls += 1
        return True

    def mark_ipc_export_transferred(self) -> bool:
        if self.state != "unconsumed":
            return False
        self.state = "transferred"
        self.transfer_calls += 1
        return True


def _cuda_wrapper_probe(*, auto_release: bool = False) -> CudaIPCWrapper:
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
    wrapper._initialize_ipc_ownership(auto_release=auto_release)
    return wrapper


def test_handler_exception_releases_unconsumed_ipc_export() -> None:
    """Handler failures cannot orphan already-decoded one-shot exports."""
    wrapper = _TrackedIPCWrapper()

    def fail(_wrapper: _TrackedIPCWrapper) -> None:
        raise RuntimeError("handler failed")

    with pytest.raises(RuntimeError, match="handler failed"):
        mq_mod._invoke_handler_with_ipc_cleanup(fail, [wrapper])

    assert wrapper.release_calls == 1
    assert wrapper.transfer_calls == 0


def test_later_decode_failure_releases_earlier_ipc_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial request decoding cannot orphan an already-created receiver."""
    wrapper = _TrackedIPCWrapper()
    decoded = iter([wrapper, ValueError("malformed second payload")])

    def decode(_payload: bytes, *, cls: type[Any]) -> Any:
        value = next(decoded)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(mq_mod, "msgspec_decode", decode)

    with pytest.raises(ValueError, match="malformed second payload"):
        mq_mod.unwrap_request_payloads([b"wrapper", b"broken"], [DeviceIPCWrapper, str])

    assert wrapper.release_calls == 1
    assert wrapper.transfer_calls == 0


def test_no_handler_rejection_decodes_and_releases_wire_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _cuda_wrapper_probe()
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    wire = mq_mod.msgspec_encode(sender, cls=DeviceIPCWrapper)
    assert sender.mark_ipc_export_transferred() is True
    mq_mod._decode_and_release_known_wire_payloads([wire], [DeviceIPCWrapper])

    release_counter.assert_called_once()


def test_rejected_known_slot_schema_mismatch_generically_releases_wire_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA Ext in an old receiver's scalar slot is decoded exactly once."""
    sender = _cuda_wrapper_probe()
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    wire = mq_mod.msgspec_encode(sender, cls=DeviceIPCWrapper)
    assert sender.mark_ipc_export_transferred() is True
    mq_mod._decode_and_release_known_wire_payloads([wire], [int])

    release_counter.assert_called_once()


def test_server_no_handler_path_releases_transferred_wire_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _cuda_wrapper_probe()
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)
    request_type = RequestType.CB_REGISTER_KV_CACHE
    payload_classes = get_payload_classes(request_type)
    payload_values = [3, [sender], "model", 1]
    payloads = [
        mq_mod.msgspec_encode(value, cls=cls)
        for value, cls in zip(payload_values, payload_classes, strict=True)
    ]
    assert sender.mark_ipc_export_transferred() is True

    server = MessageQueueServer.__new__(MessageQueueServer)
    server.socket = MagicMock()
    server._output_efd = MagicMock()
    server._output_efd.fileno.return_value = 99
    server.poller = MagicMock()
    server.is_finished = threading.Event()
    server.handlers = {}
    server._queue_error_response = MagicMock()
    server.socket.recv_multipart.return_value = [
        b"identity",
        mq_mod.msgspec_encode(4, cls=mq_mod.RequestUID),
        mq_mod.msgspec_encode(request_type, cls=RequestType),
        *payloads,
    ]
    poll_count = 0

    def poll(_timeout: int) -> dict[Any, int]:
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return {server.socket: zmq.POLLIN}
        server.is_finished.set()
        return {}

    server.poller.poll.side_effect = poll
    server._main_loop()

    release_counter.assert_called_once()
    server._queue_error_response.assert_called_once()


def test_server_unknown_request_type_releases_wire_export_and_stays_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enum version skew is a per-message rejection, not a daemon crash."""
    sender = _cuda_wrapper_probe()
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)
    wire = mq_mod.msgspec_encode(sender, cls=DeviceIPCWrapper)
    assert sender.mark_ipc_export_transferred() is True

    server = MessageQueueServer.__new__(MessageQueueServer)
    server.socket = MagicMock()
    server._output_efd = MagicMock()
    server._output_efd.fileno.return_value = 99
    server.poller = MagicMock()
    server.is_finished = threading.Event()
    server.handlers = {}
    server._queue_error_response = MagicMock()
    server.socket.recv_multipart.return_value = [
        b"identity",
        mq_mod.msgspec_encode(5, cls=mq_mod.RequestUID),
        msgspec.msgpack.encode(2**31 - 1),
        wire,
    ]
    poll_count = 0

    def poll(_timeout: int) -> dict[Any, int]:
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return {server.socket: zmq.POLLIN}
        server.is_finished.set()
        return {}

    server.poller.poll.side_effect = poll
    server._main_loop()

    assert poll_count == 2
    release_counter.assert_called_once()
    server._queue_error_response.assert_called_once()


def test_count_mismatch_releases_wrapper_in_extra_unknown_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender = _cuda_wrapper_probe()
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)

    known = mq_mod.msgspec_encode(7, cls=int)
    extra = mq_mod.msgspec_encode(sender, cls=DeviceIPCWrapper)
    assert sender.mark_ipc_export_transferred() is True

    with pytest.raises(ValueError, match="Payload count"):
        mq_mod.unwrap_request_payloads([known, extra], [int])

    release_counter.assert_called_once()


def test_blocking_submit_failure_releases_decoded_ipc_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor rejection leaves ownership with the server dispatch thread."""
    wrapper = _TrackedIPCWrapper()
    handler = BlockingRequestHandler([DeviceIPCWrapper], str, lambda _wrapper: "ok")
    handler.executor = MagicMock()
    handler.executor.submit.side_effect = RuntimeError("executor stopped")
    monkeypatch.setattr(mq_mod, "unwrap_request_payloads", lambda *_args: [wrapper])

    with pytest.raises(RuntimeError, match="executor stopped"):
        handler([b"wrapper"])

    assert wrapper.release_calls == 1
    assert wrapper.transfer_calls == 0


def test_register_fixture_releases_unconsumed_ipc_export() -> None:
    """The CUDA REGISTER fixture must not be the producer-refcount owner."""
    wrapper = _TrackedIPCWrapper()

    test_mq_handler_helpers.register_kv_cache_handler(
        0,
        [wrapper],
        "model",
        1,
        EngineType.VLLM,
        {},
        [],
    )

    assert wrapper.release_calls == 1
    assert wrapper.transfer_calls == 0


def create_cache_key(index: int, model: str = "testmodel") -> IPCCacheServerKey:
    """
    Create a cache key for testing.
    """
    chunk_size = 256
    token_ids = [index] * chunk_size
    return IPCCacheServerKey.from_token_ids(
        model,
        1,
        0,
        token_ids,
        start=0,
        end=chunk_size,
        request_id=f"test_request_{index}",
    )


def _server_process(
    server_url: str,
    ready_event: EventClass,
    shutdown_event: EventClass,
    request_handlers: dict[RequestType, Callable],
):
    """
    Server process that runs the MessageQueueServer.

    Args:
        server_url: URL to bind the server to
        ready_event: Event to signal when server is ready
        shutdown_event: Event to signal server shutdown
        request_handlers: Dict mapping RequestType to handler functions
    """
    # First Party
    from lmcache.v1.multiprocess.protocol import HandlerType

    context = zmq.Context.instance()
    server = MessageQueueServer(server_url, context)

    # Register all handlers
    blocking_types: list[RequestType] = []
    for request_type, handler in request_handlers.items():
        payload_classes = get_payload_classes(request_type)
        handler_type = get_handler_type(request_type)
        server.add_handler(request_type, payload_classes, handler_type, handler)
        if handler_type == HandlerType.BLOCKING:
            blocking_types.append(request_type)

    # Assign a normal pool for all blocking handlers in tests
    if blocking_types:
        server.add_normal_thread_pool(blocking_types, max_workers=4)

    server.start()

    # Signal that server is ready
    ready_event.set()

    # Wait for shutdown signal
    shutdown_event.wait()

    # Cleanup
    server.close()


def _run_client_test(
    server_url: str,
    ready_event: EventClass,
    request_type: RequestType,
    payloads: list[Any],
    expected_response: Any,
    num_requests: int = 1,
    client_id: int = 0,
    payload_factory: Callable[[], list[Any]] | None = None,
) -> None:
    """
    Client process that sends requests and validates responses.

    Args:
        server_url: URL to connect to
        ready_event: Event to wait for server to be ready
        request_type: Type of request to send
        payloads: List of payloads for the request
        expected_response: Expected response from server
        num_requests: Number of requests to send
        client_id: ID of this client (for debugging)
        payload_factory: Optional module-level factory invoked inside this
            spawned client for each request. CUDA IPC wrappers must be created
            here so their first and only transport is the managed MQ encoder,
            rather than Python spawn pickling the test-helper arguments.

    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Wait for server to be ready
    if not ready_event.wait(timeout=5):
        print(f"Client {client_id}: Server failed to start within timeout")
        sys.exit(1)

    # Small delay to ensure server is fully initialized
    time.sleep(0.1)

    context = zmq.Context.instance()
    client = MessageQueueClient(server_url, context)
    successful = True

    try:
        futures = []
        # Submit requests
        for _ in range(num_requests):
            request_payloads = payload_factory() if payload_factory else payloads
            future = client.submit_request(request_type, request_payloads)  # type: ignore
            futures.append(future)

        # Validate responses
        for i, future in enumerate(futures):
            response = future.result(timeout=5)
            if response != expected_response:
                print(
                    f"Client {client_id}, Request {i}: Expected "
                    f"{expected_response}, got {response}"
                )

                # Exit with error code
                client.close()
                sys.exit(1)

    except Exception as e:
        print(f"Client {client_id} test failed with exception: {e}")
        successful = False
    finally:
        client.close()
        if not successful:
            sys.exit(1)


class MessageQueueTestHelper:
    """
    Helper class to facilitate testing MessageQueueServer and MessageQueueClient.

    Supports testing with single or multiple concurrent clients, where each client
    can send multiple requests to the server.

    Usage:
        1. Create an instance with server URL
        2. Register handlers for different RequestTypes
        3. Call run_test() to execute the test with client requests

    Example:
        helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5556")
        helper.register_handler(RequestType.NOOP, noop_handler)
        helper.run_test(
            request_type=RequestType.NOOP,
            payloads=[],
            expected_response="NOOP_OK",
            num_requests=10,  # Each client sends 10 requests
            num_clients=3,    # Start 3 concurrent clients
        )
    """

    def __init__(self, server_url: str = "tcp://127.0.0.1:5556"):
        self.server_url = server_url
        self.handlers: dict[RequestType, Callable] = {}
        self.ctx = mp.get_context("spawn")

    def register_handler(
        self,
        request_type: RequestType,
        handler: Callable,
    ) -> "MessageQueueTestHelper":
        """
        Register a handler for a specific RequestType.

        Args:
            request_type: The type of request to handle
            handler: Handler function that matches the protocol signature

        Returns:
            self for method chaining
        """
        self.handlers[request_type] = handler
        return self

    def run_test(
        self,
        request_type: RequestType,
        payloads: list[Any],
        expected_response: Any,
        num_requests: int = 1,
        num_clients: int = 1,
        timeout: float = 10.0,
        payload_factory: Callable[[], list[Any]] | None = None,
    ) -> None:
        """
        Run a test by starting server and client processes.

        Args:
            request_type: Type of request to send
            payloads: List of payloads for the request
            expected_response: Expected response from server
            num_requests: Number of requests each client should send
            num_clients: Number of client processes to start
            timeout: Maximum time to wait for test completion
            payload_factory: Optional module-level per-request payload factory
                executed inside each spawned client.

        Raises:
            AssertionError: If test fails
        """
        ready_event = self.ctx.Event()
        shutdown_event = self.ctx.Event()

        # Start server process
        server_process = self.ctx.Process(
            target=_server_process,
            args=(self.server_url, ready_event, shutdown_event, self.handlers),
        )
        server_process.start()

        # Start multiple client processes
        client_processes = []
        for client_id in range(num_clients):
            client_process = self.ctx.Process(
                target=_run_client_test,
                args=(
                    self.server_url,
                    ready_event,
                    request_type,
                    payloads,
                    expected_response,
                    num_requests,
                    client_id,
                    payload_factory,
                ),
            )
            client_process.start()
            client_processes.append(client_process)

        # Wait for all clients to complete
        failed_clients = []
        for client_id, client_process in enumerate(client_processes):
            client_process.join(timeout=timeout)

            # Check if client completed successfully
            if client_process.is_alive():
                client_process.terminate()
                client_process.join()
                failed_clients.append((client_id, "timeout"))
            elif client_process.exitcode != 0:
                failed_clients.append(
                    (client_id, f"exit code {client_process.exitcode}")
                )

        # Shutdown server
        shutdown_event.set()
        server_process.join(timeout=2)

        if server_process.is_alive():
            server_process.terminate()
            server_process.join()

        # Report any failures
        if failed_clients:
            failure_details = ", ".join(
                [f"Client {cid}: {reason}" for cid, reason in failed_clients]
            )
            pytest.fail(f"Some clients failed: {failure_details}")

        if server_process.exitcode != 0:
            pytest.fail(
                f"Server process failed with exit code {server_process.exitcode}"
            )


# ==============================================================================
# Tests for Different RequestTypes
# ==============================================================================


def test_mq_noop_request():
    """
    Test MessageQueue with NOOP request type.
    NOOP takes no payloads and returns a string response.
    """
    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5556")
    helper.register_handler(RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    # Run test with single request
    helper.run_test(
        request_type=RequestType.NOOP,
        payloads=[],
        expected_response="NOOP_OK",
        num_requests=1,
    )


def test_mq_noop_multiple_requests():
    """
    Test MessageQueue with multiple NOOP requests.
    Verifies that server can handle multiple sequential requests.
    """
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5557")
    helper.register_handler(RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    # Run test with multiple requests
    helper.run_test(
        request_type=RequestType.NOOP,
        payloads=[],
        expected_response="NOOP_OK",
        num_requests=10,
    )


def test_mq_noop_multiple_clients():
    """
    Test MessageQueue with multiple concurrent clients.
    Verifies that server can handle requests from multiple clients simultaneously.
    """
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5558")
    helper.register_handler(RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    # Run test with multiple clients, each sending multiple requests
    helper.run_test(
        request_type=RequestType.NOOP,
        payloads=[],
        expected_response="NOOP_OK",
        num_requests=5,
        num_clients=3,
    )


def _make_register_kv_cache_payloads() -> list[Any]:
    """Create one-shot CUDA exports inside the spawned MQ client process."""
    kv_cache = [CudaIPCWrapper(torch.randn(2, 4, device="cuda")) for _ in range(3)]
    return [
        0,
        kv_cache,
        "testmodel",
        1,
        EngineType.VLLM,
        {"vllm_block_size": 16},
        [],
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for REGISTER_KV_CACHE tests",
)
def test_mq_register_kv_cache():
    """
    Test MessageQueue with REGISTER_KV_CACHE request type.
    REGISTER_KV_CACHE takes (gpu_id: int, kv_cache: KVCache) and returns None.
    """
    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5559")
    helper.register_handler(
        RequestType.REGISTER_KV_CACHE, test_mq_handler_helpers.register_kv_cache_handler
    )

    # Run test with REGISTER_KV_CACHE request
    helper.run_test(
        request_type=RequestType.REGISTER_KV_CACHE,
        payloads=[],
        payload_factory=_make_register_kv_cache_payloads,
        expected_response=None,
        num_requests=1,
    )


def test_mq_unregister_kv_cache():
    """
    Test MessageQueue with UNREGISTER_KV_CACHE request type.
    UNREGISTER_KV_CACHE takes (gpu_id: int) and returns None.
    """
    gpu_id = 0

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5560")
    helper.register_handler(
        RequestType.UNREGISTER_KV_CACHE,
        test_mq_handler_helpers.unregister_kv_cache_handler,
    )

    # Run test with UNREGISTER_KV_CACHE request
    helper.run_test(
        request_type=RequestType.UNREGISTER_KV_CACHE,
        payloads=[gpu_id],
        expected_response=None,
        num_requests=1,
    )


def test_mq_unregister_kv_cache_multiple_clients():
    """
    Test MessageQueue with UNREGISTER_KV_CACHE from multiple clients.
    Verifies that multiple clients can unregister KV caches concurrently.
    """
    gpu_id = 0

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5561")
    helper.register_handler(
        RequestType.UNREGISTER_KV_CACHE,
        test_mq_handler_helpers.unregister_kv_cache_handler,
    )

    # Run test with multiple clients
    helper.run_test(
        request_type=RequestType.UNREGISTER_KV_CACHE,
        payloads=[gpu_id],
        expected_response=None,
        num_requests=3,
        num_clients=2,
    )


def test_mq_store():
    """
    Test MessageQueue with STORE request type.
    STORE takes (key: KeyType, gpu_id: int, gpu_block_ids: list[list[int]],
    event_ipc_handle: bytes) and returns (bytes, bool).
    """
    # Create test key
    key = create_cache_key(0)
    gpu_id = 0
    gpu_block_ids = [[0, 1, 2]]
    test_handle = b"\x00" * 64

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5562")
    helper.register_handler(RequestType.STORE, test_mq_handler_helpers.store_handler)

    # Run test with STORE request
    helper.run_test(
        request_type=RequestType.STORE,
        payloads=[key, gpu_id, gpu_block_ids, test_handle],
        expected_response=(b"\x01" * 64, True),
        num_requests=1,
    )


def test_mq_retrieve():
    """
    Test MessageQueue with RETRIEVE request type.
    RETRIEVE takes (key: KeyType, gpu_id: int, gpu_block_ids: list[list[int]],
    event_ipc_handle: bytes) and returns (bytes, bool).
    """
    # Create test key
    key = create_cache_key(0)
    gpu_id = 0
    gpu_block_ids = [[0, 1, 2]]
    test_handle = b"\x00" * 64

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5563")
    helper.register_handler(
        RequestType.RETRIEVE, test_mq_handler_helpers.retrieve_handler
    )

    # Run test with RETRIEVE request
    helper.run_test(
        request_type=RequestType.RETRIEVE,
        payloads=[key, gpu_id, gpu_block_ids, test_handle, 0],
        expected_response=(b"\x01" * 64, True),
        num_requests=1,
    )


def test_mq_lookup():
    """
    Test MessageQueue with LOOKUP request type.
    LOOKUP takes (key: KeyType, tp_size: int) and returns None.
    The job is tracked server-side by request_id; poll via QUERY_PREFETCH_STATUS.
    """
    # Create a single test key
    key = create_cache_key(0)

    # Expected response: None (LOOKUP no longer returns a job_id)
    expected_response = None

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5564")
    helper.register_handler(RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler)

    # Run test with LOOKUP request
    helper.run_test(
        request_type=RequestType.LOOKUP,
        payloads=[key, 1],
        expected_response=expected_response,
        num_requests=1,
    )


def test_mq_lookup_with_different_key():
    """
    Test MessageQueue with LOOKUP request type with a different key.
    Tests that the handler correctly processes a single key.
    """
    # Create a different test key
    key = create_cache_key(42)

    # Expected response: None (LOOKUP no longer returns a job_id)
    expected_response = None

    # Create test helper and register handler
    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5565")
    helper.register_handler(RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler)

    # Run test with LOOKUP request
    helper.run_test(
        request_type=RequestType.LOOKUP,
        payloads=[key, 1],
        expected_response=expected_response,
        num_requests=1,
    )


def test_mq_report_block_allocation():
    """
    Test MessageQueue with REPORT_BLOCK_ALLOCATION request type.
    REPORT_BLOCK_ALLOCATION takes (instance_id, model_name, records)
    and returns None.
    """
    records = [
        BlockAllocationRecord(
            req_id="req-1",
            new_block_ids=[0, 1, 2],
            new_token_ids=[100, 200, 300],
        ),
        BlockAllocationRecord(
            req_id="req-2",
            new_block_ids=[3, 4],
            new_token_ids=[400, 500],
        ),
    ]

    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5566")
    helper.register_handler(
        RequestType.REPORT_BLOCK_ALLOCATION,
        test_mq_handler_helpers.report_block_allocations_handler,
    )

    helper.run_test(
        request_type=RequestType.REPORT_BLOCK_ALLOCATION,
        payloads=[42, "test-model", records],
        expected_response=None,
        num_requests=1,
    )


def test_mq_report_block_allocation_empty():
    """
    Test REPORT_BLOCK_ALLOCATION with an empty records list.
    """
    records: list[BlockAllocationRecord] = []

    helper = MessageQueueTestHelper(server_url="tcp://127.0.0.1:5567")
    helper.register_handler(
        RequestType.REPORT_BLOCK_ALLOCATION,
        test_mq_handler_helpers.report_block_allocations_handler,
    )

    helper.run_test(
        request_type=RequestType.REPORT_BLOCK_ALLOCATION,
        payloads=[0, "", records],
        expected_response=None,
        num_requests=1,
    )


# ==============================================================================
# Shared Polling Loop Lifecycle Tests
# ==============================================================================


def test_shared_loop_lifecycle():
    """
    Test that multiple clients share a single ClientPollingLoop and
    that the loop is torn down when all clients close.
    """
    # First Party
    from lmcache.v1.multiprocess.mq import ClientPollingLoop

    context = zmq.Context.instance()

    # No loop before any clients exist
    assert ClientPollingLoop._instance is None

    client_a = MessageQueueClient("tcp://127.0.0.1:16000", context)
    client_b = MessageQueueClient("tcp://127.0.0.1:16001", context)

    # Both share the same singleton
    loop = ClientPollingLoop._instance
    assert loop is not None
    assert loop._ref_count == 2
    assert len(loop._socket_to_client) == 2

    # Close one — loop persists
    client_a.close()
    assert ClientPollingLoop._instance is loop
    assert loop._ref_count == 1
    assert len(loop._socket_to_client) == 1

    # Close the last — loop is destroyed
    client_b.close()
    assert ClientPollingLoop._instance is None


def test_shared_loop_dispatch():
    """
    Test that the shared polling loop correctly dispatches responses
    to multiple clients connected to the same server.

    Server and clients run in the same process (different threads),
    so both clients share one ClientPollingLoop.
    """
    # First Party
    from lmcache.v1.multiprocess.mq import ClientPollingLoop

    server_url = "tcp://127.0.0.1:16020"
    context = zmq.Context.instance()

    # Start server in-process
    server = MessageQueueServer(server_url, context)
    add_handler_helper(server, RequestType.NOOP, test_mq_handler_helpers.noop_handler)
    server.start()

    try:
        # Create two clients sharing the same polling loop
        client_a = MessageQueueClient(server_url, context)
        client_b = MessageQueueClient(server_url, context)

        loop = ClientPollingLoop._instance
        assert loop is not None
        assert loop._ref_count == 2

        # Both clients submit requests concurrently
        futures_a = [client_a.submit_request(RequestType.NOOP, []) for _ in range(5)]
        futures_b = [client_b.submit_request(RequestType.NOOP, []) for _ in range(5)]

        # All futures should resolve with the correct response
        for future in futures_a:
            assert future.result(timeout=5) == "NOOP_OK"
        for future in futures_b:
            assert future.result(timeout=5) == "NOOP_OK"

        client_a.close()
        client_b.close()
        assert ClientPollingLoop._instance is None
    finally:
        server.close()


def test_full_dead_client_queue_does_not_block_healthy_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A muted DEALER fails locally without wedging the singleton loop."""
    monkeypatch.setattr(mq_mod, "_CLIENT_SNDHWM", 1)
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:16023", context)
    add_handler_helper(server, RequestType.NOOP, test_mq_handler_helpers.noop_handler)
    server.start()

    dead_client = MessageQueueClient("tcp://127.0.0.1:16024", context)
    healthy_client = MessageQueueClient("tcp://127.0.0.1:16023", context)
    try:
        dead_futures = [
            dead_client.submit_request(RequestType.NOOP, []) for _ in range(32)
        ]

        # The healthy client's request shares the same loop and must still
        # leave the process and receive its response.
        assert (
            healthy_client.submit_request(RequestType.NOOP, []).result(timeout=2)
            == "NOOP_OK"
        )

        deadline = time.monotonic() + 2
        while not any(future.query() for future in dead_futures):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        completed = [future for future in dead_futures if future.query()]
        assert completed
        with pytest.raises(RuntimeError, match="send queue is full"):
            completed[0].result()
    finally:
        dead_client.close()
        healthy_client.close()
        server.close()


def test_sent_timeout_stays_pending_until_late_response_is_transport_safe() -> None:
    """Caller expiry preserves the sent UID until its late reply is consumed."""
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = iter([17])
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()

    future = client.submit_request(RequestType.NOOP, [])
    client.process_outbound_task()
    assert client.pending_futures == {17: future}

    with pytest.raises(TimeoutError, match="not available within timeout"):
        future.result(timeout=0)
    client._polling_loop.notify.assert_called()

    # A sent request cannot be reclaimed merely because its caller expired:
    # the remote may still be using request-owned IPC resources.
    client.process_outbound_task()
    assert client.pending_futures == {17: future}

    client.socket.recv_multipart.return_value = [
        mq_mod.msgspec_encode(17, cls=mq_mod.RequestUID),
        mq_mod.msgspec_encode(RequestType.NOOP, cls=RequestType),
        mq_mod.msgspec_encode("NOOP_OK", cls=str),
    ]
    client.process_inbound()
    assert client.pending_futures == {}
    assert future.transport_complete
    with pytest.raises(TimeoutError, match="not available within timeout"):
        future.result()


def test_malformed_success_response_completes_transport_without_client_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad reply cannot leave an unresolved, untracked future/resource."""

    class _TransportResource:
        pass

    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(18)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    client_ref = weakref.ref(client)
    # Captured logger traceback records independently retain the frame. The
    # ownership contract under test is the exception stored by the future.
    monkeypatch.setattr(mq_mod.logger, "exception", lambda *_args, **_kwargs: None)

    future: mq_mod.MessagingFuture[Any] = client.submit_request(RequestType.NOOP, [])
    resource = _TransportResource()
    resource_ref = weakref.ref(resource)
    future.retain_until_transport_complete(resource)
    del resource
    client.process_outbound_task()
    assert resource_ref() is not None

    client.socket.recv_multipart.return_value = [
        mq_mod.msgspec_encode(18, cls=mq_mod.RequestUID),
        mq_mod.msgspec_encode(RequestType.NOOP, cls=RequestType),
        msgspec.msgpack.encode({"not": "the expected string"}),
    ]
    client.process_inbound()
    gc.collect()

    assert future.transport_complete
    assert client.pending_futures == {}
    assert client._inflight_ownership == {}
    assert resource_ref() is None
    assert future.exception_ is not None
    assert future.exception_.__traceback__ is None

    # Retaining the failed future must not retain process_inbound's ``self``.
    # Check before result() raises (and naturally attaches its caller traceback
    # to the otherwise sanitized exception).
    del client
    gc.collect()
    assert client_ref() is None

    with pytest.raises(
        RuntimeError, match="Failed to decode.*ValidationError.*Expected `str`"
    ):
        future.result()


def test_timeout_before_send_releases_transport_resources_without_sending() -> None:
    """An expired queued request is safe to cancel before the socket sees it."""

    class _Resource:
        pass

    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(21)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()

    wrapper = _TrackedIPCWrapper()
    future: mq_mod.MessagingFuture[Any] = client.submit_request(
        RequestType.NOOP, [wrapper]
    )
    resource = _Resource()
    resource_ref = weakref.ref(resource)
    future.retain_until_transport_complete(resource)
    del resource

    with pytest.raises(TimeoutError):
        future.result(timeout=0)
    gc.collect()
    assert resource_ref() is not None

    client.process_outbound_task()
    gc.collect()

    assert future.transport_complete
    assert resource_ref() is None
    assert client.pending_futures == {}
    assert wrapper.release_calls == 1
    assert wrapper.transfer_calls == 0
    client.socket.send_multipart.assert_not_called()


def test_successful_send_transfers_ipc_export_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An atomic socket send makes the receiver the sole export owner."""
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(22)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    wrapper = _TrackedIPCWrapper()
    monkeypatch.setattr(
        mq_mod, "get_payload_classes", lambda _request_type: [DeviceIPCWrapper]
    )
    monkeypatch.setattr(mq_mod, "msgspec_encode", lambda *_args, **_kwargs: b"encoded")

    future: mq_mod.MessagingFuture[Any] = client.submit_request(
        RequestType.NOOP, [wrapper]
    )
    client.process_outbound_task()

    assert client.pending_futures == {22: future}
    assert wrapper.transfer_calls == 1
    assert wrapper.release_calls == 0


def test_legacy_refcounted_wrapper_participates_without_new_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old mark/release subclasses fail safe without implementing new hooks."""
    monkeypatch.setattr(mq_mod, "_SENT_UNANSWERED_IPC_QUARANTINE", [])
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(26)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    wrapper = _LegacyTrackedIPCWrapper()
    monkeypatch.setattr(
        mq_mod, "get_payload_classes", lambda _request_type: [DeviceIPCWrapper]
    )
    monkeypatch.setattr(mq_mod, "msgspec_encode", lambda *_args, **_kwargs: b"encoded")

    client.submit_request(RequestType.NOOP, [wrapper])
    client.process_outbound_task()

    assert wrapper.state == "transferred"
    assert wrapper.transfer_calls == 1
    assert wrapper.release_calls == 0
    assert mq_mod._SENT_UNANSWERED_IPC_QUARANTINE == []


def test_caller_payload_mutation_after_send_cannot_hide_encoded_cuda_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire bookkeeping uses its frozen wrapper tuple, not the caller list."""
    monkeypatch.setattr(mq_mod, "_SENT_UNANSWERED_IPC_QUARANTINE", [])
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(27)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    wrapper = _cuda_wrapper_probe(auto_release=True)
    release_counter = MagicMock()
    monkeypatch.setattr(CudaIPCWrapper, "_release_counter", release_counter)
    caller_payloads = [wrapper]
    monkeypatch.setattr(
        mq_mod, "get_payload_classes", lambda _request_type: [DeviceIPCWrapper]
    )

    def accept_and_mutate(*_args: Any, **_kwargs: Any) -> None:
        caller_payloads.clear()

    client.socket.send_multipart.side_effect = accept_and_mutate
    client.submit_request(RequestType.NOOP, caller_payloads)
    client.process_outbound_task()

    assert caller_payloads == []
    assert wrapper._ipc_state == wrapper._TRANSFERRED
    assert wrapper._ipc_lease_count == 0
    assert mq_mod._SENT_UNANSWERED_IPC_QUARANTINE == []
    wrapper.__del__()
    release_counter.assert_not_called()


def test_ambiguous_discovery_quarantines_full_async_forward_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An early hook failure cannot release an undiscovered accepted export."""
    monkeypatch.setattr(mq_mod, "_SENT_UNANSWERED_IPC_QUARANTINE", [])
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(28)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    first = _cuda_wrapper_probe(auto_release=True)
    second = _cuda_wrapper_probe(auto_release=True)
    first.handle = (*first.handle[:5], 1, *first.handle[6:])
    second.handle = (*second.handle[:5], 2, *second.handle[6:])
    releases: list[int] = []
    monkeypatch.setattr(
        CudaIPCWrapper,
        "_release_counter",
        lambda self: releases.append(int(self.handle[5])),
    )
    monkeypatch.setattr(
        first,
        "ipc_export_requires_transfer",
        lambda: (_ for _ in ()).throw(RuntimeError("early discovery failure")),
    )
    monkeypatch.setattr(
        mq_mod,
        "get_payload_classes",
        lambda _request_type: [list[DeviceIPCWrapper]],
    )
    caller_payloads = [[first, second]]

    client.submit_request(RequestType.NOOP, caller_payloads)
    # Model a forwarding handler's finally-cleanup racing the queued sender.
    release_ipc_exports(caller_payloads)
    assert first._ipc_release_pending and second._ipc_release_pending
    client.process_outbound_task()

    assert first._ipc_state == first._QUARANTINED
    assert second._ipc_state == second._QUARANTINED
    assert first._ipc_lease_count == 1
    assert second._ipc_lease_count == 1
    assert releases == []
    assert len(mq_mod._SENT_UNANSWERED_IPC_QUARANTINE) == 1


def test_healthy_send_response_cycles_do_not_grow_process_quarantine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provisional producer pin is absent after every clean transfer."""
    monkeypatch.setattr(mq_mod, "_SENT_UNANSWERED_IPC_QUARANTINE", [])
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(30)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    monkeypatch.setattr(
        mq_mod, "get_payload_classes", lambda _request_type: [DeviceIPCWrapper]
    )
    monkeypatch.setattr(mq_mod, "msgspec_encode", lambda *_args, **_kwargs: b"encoded")

    def decode_response(data: bytes, *, cls: type[Any]) -> Any:
        if cls is RequestType:
            return RequestType.NOOP
        if cls is str:
            return "NOOP_OK"
        return int(data)

    monkeypatch.setattr(mq_mod, "msgspec_decode", decode_response)

    for request_uid in range(30, 35):
        wrapper = _TrackedIPCWrapper()
        future: mq_mod.MessagingFuture[Any] = client.submit_request(
            RequestType.NOOP, [wrapper]
        )
        client.process_outbound_task()

        assert wrapper.state == "transferred"
        assert mq_mod._SENT_UNANSWERED_IPC_QUARANTINE == []

        client.socket.recv_multipart.return_value = [
            str(request_uid).encode(),
            b"type",
            b"response",
        ]
        client.process_inbound()
        assert future.result() == "NOOP_OK"
        assert client.pending_futures == {}
        assert client._inflight_ownership == {}
        assert mq_mod._SENT_UNANSWERED_IPC_QUARANTINE == []


def test_failed_send_releases_ipc_export_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request rejected by ZeroMQ never transfers its export reservation."""
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(23)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    client.socket.send_multipart.side_effect = zmq.Again()
    wrapper = _TrackedIPCWrapper()
    monkeypatch.setattr(
        mq_mod, "get_payload_classes", lambda _request_type: [DeviceIPCWrapper]
    )
    monkeypatch.setattr(mq_mod, "msgspec_encode", lambda *_args, **_kwargs: b"encoded")

    future: mq_mod.MessagingFuture[Any] = client.submit_request(
        RequestType.NOOP, [wrapper]
    )
    client.process_outbound_task()

    with pytest.raises(RuntimeError, match="send queue is full"):
        future.result()
    assert client.pending_futures == {}
    assert wrapper.release_calls == 1
    assert wrapper.transfer_calls == 0


@pytest.mark.parametrize(
    "failure_phase",
    ["requires_raise", "guard_enter_raise", "mark_false", "mark_raise"],
)
def test_accepted_send_quarantine_survives_fast_response_and_gc(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    class _ProducerAllocation:
        pass

    monkeypatch.setattr(mq_mod, "_SENT_UNANSWERED_IPC_QUARANTINE", [])
    # A captured logger.exception record retains the traceback frame (and thus
    # the wrapper) independently of the transport quarantine. Suppress that
    # diagnostic sink so the final GC assertion isolates the intended owner.
    monkeypatch.setattr(
        "lmcache.v1.platform.base_ipc_wrapper.logger.exception",
        lambda *_args, **_kwargs: None,
    )
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(24)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    producer = _ProducerAllocation()
    producer_ref = weakref.ref(producer)
    wrapper = _TrackedIPCWrapper(
        failure_phase=failure_phase,
        producer_resource=producer,
    )
    wrapper_ref = weakref.ref(wrapper)
    monkeypatch.setattr(
        mq_mod, "get_payload_classes", lambda _request_type: [DeviceIPCWrapper]
    )
    monkeypatch.setattr(mq_mod, "msgspec_encode", lambda *_args, **_kwargs: b"encoded")

    future: mq_mod.MessagingFuture[Any] = client.submit_request(
        RequestType.NOOP, [wrapper]
    )
    client.process_outbound_task()

    client.socket.send_multipart.assert_called_once()
    assert client.pending_futures == {24: future}
    assert wrapper.state == "quarantined"
    assert wrapper.release_calls == 0
    assert wrapper.lease_count == 1
    assert len(mq_mod._SENT_UNANSWERED_IPC_QUARANTINE) == 1

    # A reply may race immediately after send and removes both pending_futures
    # and _inflight_ownership. The process quarantine must remain the strong
    # producer-allocation owner after every ordinary reference is gone.
    decoded = iter([24, RequestType.NOOP, "NOOP_OK"])
    monkeypatch.setattr(
        mq_mod, "msgspec_decode", lambda *_args, **_kwargs: next(decoded)
    )
    client.socket.recv_multipart.return_value = [b"uid", b"type", b"response"]
    client.process_inbound()
    assert client.pending_futures == {}
    assert client._inflight_ownership == {}
    assert future.result() == "NOOP_OK"

    del wrapper, producer, future
    gc.collect()
    assert wrapper_ref() is not None
    assert producer_ref() is not None

    # Prove that the global quarantine—not a hidden request reference—was the
    # lifetime pin. Production intentionally never clears this list.
    mq_mod._SENT_UNANSWERED_IPC_QUARANTINE.clear()
    gc.collect()
    assert wrapper_ref() is None
    assert producer_ref() is None


def test_accepted_send_quarantines_entire_batch_after_partial_mark_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(25)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    wrappers = [_TrackedIPCWrapper(), _TrackedIPCWrapper()]
    monkeypatch.setattr(
        mq_mod,
        "get_payload_classes",
        lambda _request_type: [list[DeviceIPCWrapper]],
    )
    monkeypatch.setattr(mq_mod, "msgspec_encode", lambda *_args, **_kwargs: b"encoded")
    # Fail the wrapper that sorts last so the first commit has definitely
    # happened before the injected exception.
    first, last = sorted(wrappers, key=id)
    last.failure_phase = "mark_raise"

    client.submit_request(RequestType.NOOP, [wrappers])
    client.process_outbound_task()

    assert first.transfer_calls == 1
    assert all(wrapper.state == "quarantined" for wrapper in wrappers)
    assert all(wrapper.release_calls == 0 for wrapper in wrappers)
    assert all(wrapper.lease_count == 1 for wrapper in wrappers)


def test_reset_after_accepted_send_retains_exact_ambiguous_payload_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated teardown cannot duplicate an already quarantined request."""

    class _ProducerAllocation:
        pass

    monkeypatch.setattr(mq_mod, "_SENT_UNANSWERED_IPC_QUARANTINE", [])
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.input_queue = queue.Queue()
    client._request_counter = itertools.count(40)
    client.pending_futures = {}
    client._polling_loop = MagicMock()
    client.socket = MagicMock()
    producer = _ProducerAllocation()
    producer_ref = weakref.ref(producer)
    wrapper = _TrackedIPCWrapper(producer_resource=producer)
    wrapper_ref = weakref.ref(wrapper)
    payloads = [wrapper]
    monkeypatch.setattr(
        mq_mod, "get_payload_classes", lambda _request_type: [DeviceIPCWrapper]
    )
    monkeypatch.setattr(mq_mod, "msgspec_encode", lambda *_args, **_kwargs: b"encoded")

    future: mq_mod.MessagingFuture[Any] = client.submit_request(
        RequestType.NOOP, payloads
    )
    client.process_outbound_task()
    assert wrapper.state == "transferred"
    assert mq_mod._SENT_UNANSWERED_IPC_QUARANTINE == []
    ownership = client._inflight_ownership[40]

    client._fail_outstanding("transport reset")
    assert len(mq_mod._SENT_UNANSWERED_IPC_QUARANTINE) == 1
    assert mq_mod._SENT_UNANSWERED_IPC_QUARANTINE[0] is ownership
    mq_mod._quarantine_sent_unanswered_ipc_exports(ownership)
    assert len(mq_mod._SENT_UNANSWERED_IPC_QUARANTINE) == 1
    with pytest.raises(ConnectionError, match="transport reset"):
        future.result()

    # A second teardown sees no pending request and must not append a duplicate.
    client._fail_outstanding("transport reset again")
    assert len(mq_mod._SENT_UNANSWERED_IPC_QUARANTINE) == 1

    del wrapper, producer, payloads, future, ownership
    gc.collect()
    assert wrapper_ref() is not None
    assert producer_ref() is not None

    mq_mod._SENT_UNANSWERED_IPC_QUARANTINE.clear()
    gc.collect()
    assert wrapper_ref() is None
    assert producer_ref() is None


@pytest.mark.parametrize("teardown", ["reset", "close"])
def test_transport_teardown_releases_timed_out_inflight_cuda_exporter(
    teardown: str,
) -> None:
    """Reset/close quarantines unanswered CUDA resources instead of freeing."""

    class _FakeEvent:
        pass

    loop = ClientPollingLoop.__new__(ClientPollingLoop)
    loop._poller = MagicMock()
    old_socket = MagicMock(name="old_socket")
    new_socket = MagicMock(name="new_socket")
    client = MessageQueueClient.__new__(MessageQueueClient)
    client.ctx = MagicMock()
    client.ctx.socket.return_value = new_socket
    client.server_url = "tcp://127.0.0.1:16024"
    client.socket = old_socket
    client._socket_closed = False
    client._socket_close_lock = threading.Lock()
    client.pending_futures = {}
    client.input_queue = queue.Queue()
    loop._socket_to_client = {old_socket: client}

    raw_future = mq_mod.MessagingFuture[tuple[bytes, bool]]()
    event = _FakeEvent()
    event_ref = weakref.ref(event)
    cuda_future = raw_future.to_cuda_future(
        device="cuda:0",
        completion_event=event,
    )
    client.pending_futures[9] = raw_future
    del event
    with pytest.raises(TimeoutError):
        cuda_future.result(timeout=0)
    del cuda_future
    gc.collect()
    assert event_ref() is not None

    if teardown == "reset":
        loop._reset_client(client)
    else:
        loop._retire_client(client)
    gc.collect()

    assert raw_future.transport_complete
    assert event_ref() is not None
    with pytest.raises(TimeoutError):
        raw_future.result()


def test_connection_reset_discards_stale_pending_and_unsent_work() -> None:
    """Retiring an outage session fails old work before the fresh socket starts."""
    loop = ClientPollingLoop.__new__(ClientPollingLoop)
    loop._poller = MagicMock()
    old_socket = MagicMock(name="old_socket")
    new_socket = MagicMock(name="new_socket")

    client = MessageQueueClient.__new__(MessageQueueClient)
    client.ctx = MagicMock()
    client.ctx.socket.return_value = new_socket
    client.server_url = "tcp://127.0.0.1:16024"
    client.socket = old_socket
    client._socket_closed = False
    client._socket_close_lock = threading.Lock()
    client.pending_futures = {}
    client.input_queue = queue.Queue()

    pending: mq_mod.MessagingFuture[Any] = mq_mod.MessagingFuture()
    unsent: mq_mod.MessagingFuture[Any] = mq_mod.MessagingFuture()
    unsent_wrapper = _TrackedIPCWrapper()
    client.pending_futures[3] = pending
    client.input_queue.put(
        MessageQueueClient.WrappedRequest(4, unsent, RequestType.NOOP, [unsent_wrapper])
    )
    loop._socket_to_client = {old_socket: client}

    loop._reset_client(client)

    with pytest.raises(ConnectionError, match="became unhealthy"):
        pending.result()
    with pytest.raises(ConnectionError, match="became unhealthy"):
        unsent.result()
    assert unsent_wrapper.release_calls == 1
    assert unsent_wrapper.transfer_calls == 0
    old_socket.close.assert_called_once_with(linger=0)
    new_socket.setsockopt.assert_any_call(zmq.SNDHWM, mq_mod._CLIENT_SNDHWM)
    new_socket.setsockopt.assert_any_call(zmq.LINGER, 0)
    new_socket.connect.assert_called_once_with(client.server_url)
    loop._poller.unregister.assert_called_once_with(old_socket)
    loop._poller.register.assert_called_once_with(new_socket, zmq.POLLIN)
    assert loop._socket_to_client == {new_socket: client}


def test_timed_out_unregister_is_retired_when_loop_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deferred unregister keeps the socket alive until poller removal."""
    monkeypatch.setattr(mq_mod, "_CLIENT_CONTROL_TIMEOUT_S", 0.01)
    loop = ClientPollingLoop.__new__(ClientPollingLoop)
    loop._ops_queue = queue.Queue()
    loop._notifier = MagicMock()
    loop._poller = MagicMock()
    client = MagicMock()
    loop._socket_to_client = {client.socket: client}

    started = time.monotonic()
    assert loop.unregister(client) is False
    assert time.monotonic() - started < 0.5

    # When the loop recovers, it removes the socket before closing it.
    loop._process_ops()
    loop._poller.unregister.assert_called_once_with(client.socket)
    client._close_socket.assert_called_once_with()
    assert loop._socket_to_client == {}


def test_claimed_register_timeout_rolls_back_on_polling_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed REGISTER cannot outlive and re-register a failed client."""
    monkeypatch.setattr(mq_mod, "_CLIENT_CONTROL_TIMEOUT_S", 0.1)
    loop = ClientPollingLoop.__new__(ClientPollingLoop)
    loop._ops_queue = queue.Queue()
    loop._notifier = MagicMock()
    loop._poller = MagicMock()
    loop._socket_to_client = {}
    client = MagicMock()
    register_claimed = threading.Event()
    release_register = threading.Event()

    def _delayed_register(*_args) -> None:
        register_claimed.set()
        assert release_register.wait(timeout=1)

    loop._poller.register.side_effect = _delayed_register

    errors: list[BaseException] = []

    def _register() -> None:
        try:
            loop.register(client)
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target=_register, daemon=True)
    caller.start()
    deadline = time.monotonic() + 1
    while loop._ops_queue.empty():
        assert time.monotonic() < deadline
        time.sleep(0.001)

    worker = threading.Thread(target=loop._process_ops, daemon=True)
    worker.start()
    assert register_claimed.wait(timeout=1)
    caller.join(timeout=1)
    assert len(errors) == 1
    assert isinstance(errors[0], TimeoutError)
    client._close_socket.assert_not_called()

    release_register.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    loop._poller.unregister.assert_called_once_with(client.socket)
    client._close_socket.assert_called_once_with()
    assert loop._socket_to_client == {}


def test_release_instance_uses_bounded_join_and_preserves_live_notifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown returns even if the loop is stuck for an unrelated reason."""
    monkeypatch.setattr(mq_mod, "_CLIENT_THREAD_JOIN_TIMEOUT_S", 0.01)
    thread = MagicMock()
    thread.is_alive.return_value = True
    notifier = MagicMock()
    loop = SimpleNamespace(
        _ref_count=1,
        _is_finished=MagicMock(),
        _notifier=notifier,
        _thread=thread,
    )
    ClientPollingLoop._instance = loop
    try:
        ClientPollingLoop.release_instance()
    finally:
        ClientPollingLoop._instance = None

    thread.join.assert_called_once_with(timeout=0.01)
    notifier.close.assert_not_called()


def test_client_close_defers_socket_close_after_unregister_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MessageQueueClient.__new__(MessageQueueClient)
    client._closed = False
    client._polling_loop = MagicMock()
    client._polling_loop.unregister.return_value = False
    client.socket = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(ClientPollingLoop, "release_instance", release)

    client.close()

    release.assert_called_once_with()
    client.socket.close.assert_not_called()


def test_sync_handler_failure_completes_future_and_loop_recovers():
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:16021", context)
    add_handler_helper(
        server, RequestType.NOOP, test_mq_handler_helpers.failing_noop_handler
    )
    server.start()
    client = MessageQueueClient("tcp://127.0.0.1:16021", context)

    try:
        failed = client.submit_request(RequestType.NOOP, [])
        with pytest.raises(
            RemoteHandlerError, match="intentional sync handler failure"
        ) as exc_info:
            failed.result(timeout=2)
        assert exc_info.value.request_type is RequestType.NOOP
        assert exc_info.value.error_type == "ValueError"

        server.add_sync_handler(
            RequestType.NOOP, [], test_mq_handler_helpers.noop_handler
        )
        assert (
            client.submit_request(RequestType.NOOP, []).result(timeout=2) == "NOOP_OK"
        )
    finally:
        client.close()
        server.close()


def test_blocking_handler_failure_completes_future():
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:16022", context)
    add_handler_helper(
        server, RequestType.LOOKUP, test_mq_handler_helpers.failing_lookup_handler
    )
    server.add_normal_thread_pool([RequestType.LOOKUP], max_workers=1)
    server.start()
    client = MessageQueueClient("tcp://127.0.0.1:16022", context)

    try:
        future = client.submit_request(RequestType.LOOKUP, [create_cache_key(1), 1])
        with pytest.raises(
            RemoteHandlerError, match="intentional blocking handler failure"
        ) as exc_info:
            future.result(timeout=2)
        assert exc_info.value.request_type is RequestType.LOOKUP
        assert exc_info.value.error_type == "OSError"
    finally:
        client.close()
        server.close()


def test_shared_loop_recreate():
    """
    Test that closing all clients and creating new ones starts a fresh loop.
    """
    # First Party
    from lmcache.v1.multiprocess.mq import ClientPollingLoop

    context = zmq.Context.instance()

    client = MessageQueueClient("tcp://127.0.0.1:16010", context)
    first_loop = ClientPollingLoop._instance
    assert first_loop is not None
    client.close()
    assert ClientPollingLoop._instance is None

    # New client creates a brand-new loop
    client2 = MessageQueueClient("tcp://127.0.0.1:16011", context)
    second_loop = ClientPollingLoop._instance
    assert second_loop is not None
    assert second_loop is not first_loop
    client2.close()
    assert ClientPollingLoop._instance is None


# ==============================================================================
# Thread Pool Tests
# ==============================================================================


def test_add_normal_thread_pool():
    """
    Test that add_normal_thread_pool assigns handler executors.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15700", context)

    add_handler_helper(
        server, RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler
    )
    add_handler_helper(server, RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    lookup_handler = server.handlers[RequestType.LOOKUP]
    assert isinstance(lookup_handler, BlockingRequestHandler)
    assert lookup_handler.executor is None

    server.add_normal_thread_pool([RequestType.LOOKUP], max_workers=4)

    assert lookup_handler.executor is not None
    assert len(server.extra_pools) == 1

    server.close()


def test_add_affinity_thread_pool():
    """
    Test that add_affinity_thread_pool assigns AffinityThreadPool executors.
    """
    # First Party
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15700", context)

    add_handler_helper(server, RequestType.STORE, test_mq_handler_helpers.store_handler)
    add_handler_helper(
        server, RequestType.RETRIEVE, test_mq_handler_helpers.retrieve_handler
    )

    store_handler = server.handlers[RequestType.STORE]
    retrieve_handler = server.handlers[RequestType.RETRIEVE]
    assert isinstance(store_handler, BlockingRequestHandler)
    assert isinstance(retrieve_handler, BlockingRequestHandler)
    assert store_handler.executor is None

    server.add_affinity_thread_pool(
        [RequestType.STORE, RequestType.RETRIEVE], max_workers=2
    )

    assert isinstance(store_handler.executor, AffinityThreadPool)
    assert store_handler.executor is retrieve_handler.executor
    assert len(server.extra_pools) == 1

    server.close()


def test_normal_pool_error_on_sync_handler():
    """
    Test that add_normal_thread_pool raises TypeError for SYNC handlers.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15701", context)

    add_handler_helper(server, RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    with pytest.raises(TypeError, match="not BlockingRequestHandler"):
        server.add_normal_thread_pool([RequestType.NOOP], max_workers=1)

    server.close()


def test_affinity_pool_error_on_sync_handler():
    """
    Test that add_affinity_thread_pool raises TypeError for SYNC handlers.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15701", context)

    add_handler_helper(server, RequestType.NOOP, test_mq_handler_helpers.noop_handler)

    with pytest.raises(TypeError, match="not BlockingRequestHandler"):
        server.add_affinity_thread_pool([RequestType.NOOP], max_workers=1)

    server.close()


def test_pool_error_on_unregistered():
    """
    Test that pool methods raise ValueError for unregistered request types.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15702", context)

    with pytest.raises(ValueError, match="No handler registered"):
        server.add_normal_thread_pool([RequestType.STORE], max_workers=1)

    with pytest.raises(ValueError, match="No handler registered"):
        server.add_affinity_thread_pool([RequestType.STORE], max_workers=1)

    server.close()


def test_multiple_pools():
    """
    Test that normal and affinity pools can coexist.
    """
    # First Party
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15703", context)

    add_handler_helper(server, RequestType.STORE, test_mq_handler_helpers.store_handler)
    add_handler_helper(
        server, RequestType.RETRIEVE, test_mq_handler_helpers.retrieve_handler
    )
    add_handler_helper(
        server, RequestType.LOOKUP, test_mq_handler_helpers.lookup_handler
    )

    server.add_affinity_thread_pool(
        [RequestType.STORE, RequestType.RETRIEVE], max_workers=2
    )
    server.add_normal_thread_pool([RequestType.LOOKUP], max_workers=3)

    store_handler = server.handlers[RequestType.STORE]
    retrieve_handler = server.handlers[RequestType.RETRIEVE]
    lookup_handler = server.handlers[RequestType.LOOKUP]
    assert isinstance(store_handler, BlockingRequestHandler)
    assert isinstance(retrieve_handler, BlockingRequestHandler)
    assert isinstance(lookup_handler, BlockingRequestHandler)

    # STORE/RETRIEVE share affinity pool
    assert isinstance(store_handler.executor, AffinityThreadPool)
    assert store_handler.executor is retrieve_handler.executor
    # LOOKUP uses normal pool
    assert store_handler.executor is not lookup_handler.executor
    assert not isinstance(lookup_handler.executor, AffinityThreadPool)
    assert len(server.extra_pools) == 2

    server.close()


def test_start_fails_without_pool_assignment():
    """
    Test that start() raises RuntimeError if a blocking handler
    has no executor assigned.
    """
    context = zmq.Context.instance()
    server = MessageQueueServer("tcp://127.0.0.1:15704", context)

    add_handler_helper(server, RequestType.STORE, test_mq_handler_helpers.store_handler)
    # Don't assign any pool

    with pytest.raises(RuntimeError, match="no thread pool assigned"):
        server.start()

    server.close()
