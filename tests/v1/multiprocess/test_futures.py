# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock
import gc
import multiprocessing as mp
import threading
import time
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.mp_observability import errors as timeout_errors_mod
from lmcache.v1.multiprocess import futures as futures_mod
from lmcache.v1.multiprocess.futures import CUDAMessagingFuture, MessagingFuture

# ==============================================================================
# Helper Functions for CUDAMessagingFuture Tests
# ==============================================================================


def _create_cuda_event_in_process(event_queue: mp.Queue, delay: float = 0.0):
    """Helper process that creates a CUDA event and sends the IPC handle."""
    torch.cuda.init()
    if delay > 0:
        time.sleep(delay)

    # Create and record a CUDA event with interprocess flag
    event = torch.cuda.Event(interprocess=True)
    event.record()
    event_bytes = event.ipc_handle()

    # Send the event handle to the main process
    event_queue.put(event_bytes)


def test_messaging_future_basic_usage():
    """Test basic usage of MessagingFuture: set result and retrieve it."""
    future = MessagingFuture[int]()

    # Initially, future should not be done
    assert not future.query(), "Future should not be done initially"

    # Set result
    future.set_result(42)

    # Future should now be done
    assert future.query(), "Future should be done after setting result"

    # Get result (should be immediate)
    result = future.result(timeout=1)
    assert result == 42, f"Expected result 42, got {result}"


def test_messaging_future_with_thread():
    """Test MessagingFuture with result set from another thread."""
    future = MessagingFuture[str]()

    def set_future_result():
        time.sleep(0.5)
        future.set_result("Hello from thread")

    # Start thread that will set the result
    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Initially should not be done
    assert not future.query(), "Future should not be done before thread sets result"

    # Wait for result
    result = future.result(timeout=2)
    assert result == "Hello from thread", f"Expected 'Hello from thread', got {result}"

    # Should be done now
    assert future.query(), "Future should be done after getting result"

    thread.join()


def test_messaging_future_wait_success():
    """Test wait method when result becomes available."""
    future = MessagingFuture[int]()

    def set_future_result():
        time.sleep(0.3)
        future.set_result(100)

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait should return True when result is set
    success = future.wait(timeout=1)
    assert success, "Wait should return True when result is available"
    assert future.query(), "Future should be done after wait returns True"

    thread.join()


def test_messaging_future_wait_timeout():
    """Test wait method when timeout is reached."""
    future = MessagingFuture[int]()

    # Wait with short timeout (result never set)
    start_time = time.time()
    success = future.wait(timeout=0.2)
    elapsed = time.time() - start_time

    assert not success, "Wait should return False on timeout"
    assert not future.query(), "Future should not be done after timeout"
    assert 0.15 < elapsed < 0.3, f"Wait should respect timeout, elapsed: {elapsed}"


def test_messaging_future_result_timeout():
    """A result deadline terminally expires an unanswered future."""
    future = MessagingFuture[int]()

    with pytest.raises(
        TimeoutError, match="Future result not available within timeout"
    ):
        future.result(timeout=0.2)

    assert future.query(), "A result deadline should make the future terminal"
    with pytest.raises(
        TimeoutError, match="Future result not available within timeout"
    ):
        future.result()


def test_messaging_future_wait_no_timeout():
    """Test wait method without timeout (waits indefinitely until result is set)."""
    future = MessagingFuture[float]()

    def set_future_result():
        time.sleep(0.3)
        future.set_result(3.14)

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait without timeout should wait until result is available
    success = future.wait()  # No timeout parameter
    assert success, "Wait should return True when result is set"
    assert future.result() == 3.14, "Result should be accessible after wait"

    thread.join()


def test_messaging_future_multiple_result_calls():
    """Test that result can be retrieved multiple times after being set."""
    future = MessagingFuture[str]()
    future.set_result("persistent value")

    # Get result multiple times
    result1 = future.result(timeout=0.1)
    result2 = future.result(timeout=0.1)
    result3 = future.result(timeout=0.1)

    assert result1 == result2 == result3 == "persistent value", (
        "Result should be retrievable multiple times"
    )


def test_messaging_future_complex_type():
    """Test MessagingFuture with complex types like lists and dicts."""
    future = MessagingFuture[dict]()

    complex_data = {"key1": [1, 2, 3], "key2": {"nested": "value"}, "key3": 42}

    def set_future_result():
        time.sleep(0.2)
        future.set_result(complex_data)

    thread = threading.Thread(target=set_future_result)
    thread.start()

    result = future.result(timeout=1)
    assert result == complex_data, "Complex types should be preserved"

    thread.join()


def test_messaging_future_exception_is_re_raised():
    """A remote messaging failure completes the future instead of hanging it."""
    future = MessagingFuture[int]()
    error = RuntimeError("remote operation failed")

    future.set_exception(error)

    assert future.query()
    assert future.wait(timeout=0.1)
    with pytest.raises(RuntimeError, match="remote operation failed") as exc_info:
        future.result(timeout=0.1)
    assert exc_info.value is error


def test_messaging_future_rejects_non_exception():
    future = MessagingFuture[int]()

    with pytest.raises(TypeError, match="must derive from BaseException"):
        future.set_exception("not an exception")  # type: ignore[arg-type]


def test_cuda_future_retains_exporter_event_until_raw_response() -> None:
    """A local IPC exporter is not observed before the server reply arrives."""

    class _FakeEvent:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def query(self) -> bool:
            return True

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    event = _FakeEvent()
    raw_future = MessagingFuture[tuple[bytes, int]]()
    future = raw_future.to_cuda_future(
        device="cuda:0",
        completion_event=event,
    )

    # The event was initially recorded by the exporter, so it is already
    # queryable. The response gate must still keep the operation pending.
    assert not future.query()

    raw_future.set_result((b"worker-owned-event", 42))
    assert future.query()
    assert future.result() == 42
    assert future.event_ is event
    assert event.synchronize_calls == 1


def test_raw_future_retains_exporter_when_cuda_future_is_abandoned() -> None:
    """A timed-out caller cannot destroy an event still in use by the server."""

    class _FakeEvent:
        pass

    event = _FakeEvent()
    event_ref = weakref.ref(event)
    raw_future = MessagingFuture[tuple[bytes, bool]]()
    cuda_future = raw_future.to_cuda_future(
        device="cuda:0",
        completion_event=event,
    )

    del event, cuda_future
    gc.collect()
    assert event_ref() is not None

    raw_future.set_result((b"worker-owned-event", True))
    gc.collect()
    assert event_ref() is None


def test_timeout_releases_resources_once_and_rejects_late_reply() -> None:
    """A deadline owns the terminal state and releases its transport lease once."""

    class _Resource:
        pass

    timeout_notifications: list[str] = []
    releases: list[str] = []
    resource = _Resource()
    resource_ref = weakref.ref(resource)
    weakref.finalize(resource, releases.append, "released")
    future = MessagingFuture[int](
        on_timeout=lambda: timeout_notifications.append("timeout")
    )
    future.retain_until_complete(resource)
    del resource

    with pytest.raises(
        TimeoutError, match="Future result not available within timeout"
    ):
        future.result(timeout=0)

    gc.collect()
    assert resource_ref() is None
    assert releases == ["released"]
    assert timeout_notifications == ["timeout"]

    # A late response or transport failure cannot overwrite the timeout or
    # release the already-detached resource a second time.
    future.set_result(42)
    future.set_exception(ConnectionError("late transport failure"))
    gc.collect()
    assert releases == ["released"]
    assert timeout_notifications == ["timeout"]
    with pytest.raises(
        TimeoutError, match="Future result not available within timeout"
    ):
        future.result()


def test_first_success_releases_resources_once_and_rejects_late_failure() -> None:
    """A response wins once even when later terminal signals are delivered."""

    class _Resource:
        pass

    releases: list[str] = []
    resource = _Resource()
    weakref.finalize(resource, releases.append, "released")
    future = MessagingFuture[int]()
    future.retain_until_complete(resource)
    del resource

    future.set_result(7)
    future.set_exception(ConnectionError("late transport failure"))
    future.set_result(9)

    gc.collect()
    assert releases == ["released"]
    assert future.result() == 7


def test_completion_timeout_race_has_one_terminal_owner() -> None:
    """Concurrent completion and expiry never double-release or overwrite state."""

    class _Resource:
        pass

    def _run_race() -> None:
        timeout_notifications: list[str] = []
        releases: list[str] = []
        resource = _Resource()
        weakref.finalize(resource, releases.append, "released")
        future = MessagingFuture[int](
            on_timeout=lambda: timeout_notifications.append("timeout")
        )
        future.retain_until_complete(resource)
        del resource
        start = threading.Barrier(3)
        observed: list[int | BaseException] = []

        def _expire() -> None:
            start.wait()
            try:
                observed.append(future.result(timeout=0))
            except BaseException as exc:
                observed.append(exc)

        def _complete() -> None:
            start.wait()
            future.set_result(11)

        expiry_thread = threading.Thread(target=_expire)
        completion_thread = threading.Thread(target=_complete)
        expiry_thread.start()
        completion_thread.start()
        start.wait()
        expiry_thread.join(timeout=2)
        completion_thread.join(timeout=2)

        assert not expiry_thread.is_alive()
        assert not completion_thread.is_alive()
        assert len(observed) == 1
        if isinstance(observed[0], BaseException):
            assert isinstance(observed[0], TimeoutError)
            assert timeout_notifications == ["timeout"]
            with pytest.raises(TimeoutError):
                future.result()
        else:
            assert observed == [11]
            assert timeout_notifications == []
            assert future.result() == 11

        future.set_result(99)
        future.set_exception(ConnectionError("late"))
        gc.collect()
        assert releases == ["released"]

    for _attempt in range(100):
        _run_race()


def test_cuda_timeout_quarantines_exporter_until_late_transport_reply() -> None:
    """A sent request timeout cannot release an exporter still used remotely."""

    class _FakeEvent:
        pass

    event = _FakeEvent()
    event_ref = weakref.ref(event)
    raw_future = MessagingFuture[tuple[bytes, bool]]()
    cuda_future = raw_future.to_cuda_future(
        device="cuda:0",
        completion_event=event,
    )
    del event

    with pytest.raises(
        TimeoutError, match="CUDAMessagingFuture result not available within timeout"
    ):
        cuda_future.result(timeout=0)

    # Expiry detached the raw future's lease, but the caller-visible CUDA
    # future and the transport lease both still own the exporter.
    gc.collect()
    assert event_ref() is not None
    del cuda_future
    gc.collect()
    assert event_ref() is not None

    # The late transport response cannot replace the caller timeout, but it
    # does establish the safe point at which the exporter can be released.
    raw_future.set_result((b"late-worker-owned-event", True))
    gc.collect()
    assert event_ref() is None
    with pytest.raises(
        TimeoutError, match="CUDAMessagingFuture result not available within timeout"
    ):
        raw_future.result()


def test_timeout_callback_failure_does_not_replace_terminal_timeout() -> None:
    """A notifier failure is contained and the first deadline remains stable."""
    callback_calls = 0

    def failing_callback() -> None:
        nonlocal callback_calls
        callback_calls += 1
        raise RuntimeError("notifier failed")

    future = MessagingFuture[int](on_timeout=failing_callback)

    for timeout in (0, None, None):
        with pytest.raises(
            TimeoutError, match="Future result not available within timeout"
        ):
            future.result(timeout=timeout)

    assert callback_calls == 1


def test_repeated_timeout_consumption_publishes_one_observability_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh raised instances do not report the same deadline repeatedly."""
    published: list[str] = []
    monkeypatch.setattr(timeout_errors_mod, "is_observability_enabled", lambda: True)
    monkeypatch.setattr(
        timeout_errors_mod.LMCacheTimeoutError,
        "_publish_timeout_event",
        lambda self, message, stacktrace, session_id: published.append(message),
    )
    future = MessagingFuture[int]()

    for timeout in (0, None, None):
        with pytest.raises(TimeoutError):
            future.result(timeout=timeout)

    assert published == ["Future result not available within timeout"]


def test_cuda_future_materializes_ipc_event_once_for_concurrent_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent query calls cannot import the same IPC event twice."""

    class _FakeEvent:
        def query(self) -> bool:
            return True

        def synchronize(self) -> None:
            pass

    factory_calls = 0
    factory_lock = threading.Lock()
    second_factory_call = threading.Event()

    def from_ipc_handle(device: object, event_bytes: bytes) -> _FakeEvent:
        del device, event_bytes
        nonlocal factory_calls
        with factory_lock:
            factory_calls += 1
            call_number = factory_calls
        if call_number == 1:
            # Without the materialization lock, the second consumer enters the
            # factory and releases this wait. With the fix, this bounded wait
            # expires and the second consumer observes the materialized event.
            second_factory_call.wait(timeout=0.2)
        else:
            second_factory_call.set()
        return _FakeEvent()

    fake_backend = MagicMock()
    fake_backend.Event.from_ipc_handle.side_effect = from_ipc_handle
    monkeypatch.setattr(futures_mod, "torch_dev", fake_backend)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    raw_future.set_result((b"ipc-event", 7))
    cuda_future = raw_future.to_cuda_future(device="cuda:0")
    start = threading.Barrier(3)
    results: list[bool] = []

    def consume() -> None:
        start.wait()
        results.append(cuda_future.query())

    consumers = [threading.Thread(target=consume) for _ in range(2)]
    for consumer in consumers:
        consumer.start()
    start.wait()
    for consumer in consumers:
        consumer.join(timeout=2)

    assert all(not consumer.is_alive() for consumer in consumers)
    assert results == [True, True]
    assert factory_calls == 1


# ==============================================================================
# CUDAMessagingFuture Tests
# ==============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_basic_usage():
    """Test basic usage of CUDAMessagingFuture: create, wait, and get result."""
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    # Create the raw future that will return (event_bytes, result_value)
    raw_future = MessagingFuture[tuple[bytes, int]]()

    # Create CUDAMessagingFuture from raw future
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    # Initially, future should not be done
    assert not cuda_future.query(), "CUDAMessagingFuture should not be done initially"

    # Set result in raw future
    raw_future.set_result((event_bytes, 42))

    # Wait for CUDA future to complete
    success = cuda_future.wait()
    assert success, "Wait should return True when result is available"

    # Get result
    result = cuda_future.result()
    assert result == 42, f"Expected result 42, got {result}"

    # Query should return True after completion
    assert cuda_future.query(), "CUDAMessagingFuture should be done after wait"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_with_thread():
    """Test CUDAMessagingFuture with result set from another thread."""
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, str]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.5)
        raw_future.set_result((event_bytes, "Hello CUDA"))

    # Start thread that will set the result
    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Initially should not be done
    assert not cuda_future.query(), (
        "Future should not be done before thread sets result"
    )

    # Wait for result
    result = cuda_future.result()
    assert result == "Hello CUDA", f"Expected 'Hello CUDA', got {result}"

    # Should be done now
    assert cuda_future.query(), "Future should be done after getting result"

    thread.join()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_wait_no_timeout():
    """Test wait method without timeout (waits indefinitely
    until result is set).
    """
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, float]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.3)
        raw_future.set_result((event_bytes, 3.14))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait without timeout should wait until result is available
    success = cuda_future.wait()  # No timeout parameter
    assert success, "Wait should return True when result is set"
    assert cuda_future.result() == 3.14, "Result should be accessible after wait"

    thread.join()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_wait_with_timeout_success():
    """Test that wait method works correctly with timeout when result is available."""
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.3)
        raw_future.set_result((event_bytes, 123))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Wait with timeout should return True when result is available
    success = cuda_future.wait(timeout=2.0)
    assert success, "Wait with timeout should return True when result is available"
    assert cuda_future.result() == 123, "Result should be accessible after wait"

    thread.join()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_wait_timeout_reached():
    """Test that wait method returns False when timeout is reached."""
    torch.cuda.init()

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    # Wait with short timeout (result never set)
    start_time = time.time()
    success = cuda_future.wait(timeout=0.2)
    elapsed = time.time() - start_time

    assert not success, "Wait should return False on timeout"
    assert not cuda_future.query(), "Future should not be done after timeout"
    assert 0.15 < elapsed < 0.4, f"Wait should respect timeout, elapsed: {elapsed}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_result_with_timeout_success():
    """Test that result method works correctly with timeout when result is available."""
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.3)
        raw_future.set_result((event_bytes, 456))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    # Get result with timeout should succeed when result is available
    result = cuda_future.result(timeout=2.0)
    assert result == 456, f"Expected result 456, got {result}"

    thread.join()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_result_timeout_reached():
    """Test that result method raises TimeoutError when timeout is reached."""
    torch.cuda.init()

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    # Try to get result with timeout (result never set)
    with pytest.raises(
        TimeoutError, match="CUDAMessagingFuture result not available within timeout"
    ):
        cuda_future.result(timeout=0.2)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_multiple_result_calls():
    """Test that result can be retrieved multiple times after being set."""
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, str]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    raw_future.set_result((event_bytes, "persistent cuda value"))

    # Get result multiple times
    result1 = cuda_future.result()
    result2 = cuda_future.result()
    result3 = cuda_future.result()

    assert result1 == result2 == result3 == "persistent cuda value", (
        "Result should be retrievable multiple times"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_query_before_and_after():
    """Test query method returns False before completion and True after."""
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    # Query before setting result
    assert not cuda_future.query(), "Query should return False before result is set"

    # Set result
    raw_future.set_result((event_bytes, 100))

    # Wait for completion
    cuda_future.wait()

    # Query after setting result
    assert cuda_future.query(), (
        "Query should return True after result is set and waited"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_complex_type():
    """Test CUDAMessagingFuture with complex types like lists and dicts."""
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    complex_data = {"key1": [1, 2, 3], "key2": {"nested": "value"}, "key3": 42}

    raw_future = MessagingFuture[tuple[bytes, dict]]()
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future)

    def set_future_result():
        time.sleep(0.2)
        raw_future.set_result((event_bytes, complex_data))

    thread = threading.Thread(target=set_future_result)
    thread.start()

    result = cuda_future.result()
    assert result == complex_data, "Complex types should be preserved"

    thread.join()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_messaging_future_to_cuda_future():
    """Test converting MessagingFuture to CUDAMessagingFuture
    using to_cuda_future method.
    """
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    raw_future = MessagingFuture[tuple[bytes, int]]()

    # Convert to CUDA future
    cuda_future = raw_future.to_cuda_future()

    # Verify it's a CUDAMessagingFuture instance
    assert isinstance(cuda_future, CUDAMessagingFuture), (
        "to_cuda_future should return CUDAMessagingFuture instance"
    )

    # Set result and verify it works
    raw_future.set_result((event_bytes, 999))

    result = cuda_future.result()
    assert result == 999, f"Expected result 999, got {result}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CUDAMessagingFuture tests",
)
def test_cuda_messaging_future_with_explicit_device():
    """Test CUDAMessagingFuture with explicit device parameter."""
    torch.cuda.init()

    # Create CUDA event in a separate process
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    process = ctx.Process(target=_create_cuda_event_in_process, args=(event_queue,))
    process.start()

    # Get event bytes from the process
    event_bytes = event_queue.get(timeout=30)
    process.join(timeout=2)

    device = torch.cuda.current_device()
    raw_future = MessagingFuture[tuple[bytes, str]]()

    # Create CUDA future with explicit device
    cuda_future = CUDAMessagingFuture.FromMessagingFuture(raw_future, device=device)

    # Set result
    raw_future.set_result((event_bytes, "explicit device"))

    # Get result
    result = cuda_future.result()
    assert result == "explicit device", f"Expected 'explicit device', got {result}"
