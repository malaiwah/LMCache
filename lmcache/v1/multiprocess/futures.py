# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Callable, Generic, Optional, TypeVar
import threading

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

T = TypeVar("T")


class MessagingFuture(Generic[T]):
    def __init__(self, on_timeout: Callable[[], None] | None = None):
        self.is_done_ = threading.Event()
        self.result_ = None
        self.exception_: BaseException | None = None
        self._completion_lock = threading.Lock()
        self._retained_resources: list[Any] = []
        self._on_timeout = on_timeout

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        return self.is_done_.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            bool: True if the future is done, False if the timeout was reached.
        """
        return self.is_done_.wait(timeout)

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            timeout_error = LMCacheTimeoutError(
                "Future result not available within timeout"
            )
            if self._expire(timeout_error):
                # Keep the terminal sentinel traceback-free. Storing and
                # raising the same exception would make the future retain its
                # own result() frame and anything reachable from that frame.
                raise LMCacheTimeoutError(str(timeout_error))
            # Completion won the deadline race while wait() was returning.
            # Fall through and consume that terminal state.
        self._raise_if_failed()
        return self.result_

    def retain_until_complete(self, resource: Any) -> None:
        """Keep ``resource`` alive until this future first becomes terminal."""
        with self._completion_lock:
            if not self.is_done_.is_set():
                self._retained_resources.append(resource)

    def set_result(self, result: T) -> None:
        """
        Set the result of the future and mark it as done. This function is NOT
        SUPPOSED TO BE CALLED by users directly. It should be only called by
        the messaging system when the result is available.

        Args:
            result (T): The result to set.
        """
        self._complete(result=result)

    def set_exception(self, exception: BaseException) -> None:
        """Complete the future with an exception from the messaging system."""
        if not isinstance(exception, BaseException):
            raise TypeError("exception must derive from BaseException")
        self._complete(exception=exception)

    def to_cuda_future(
        self,
        device: Any | None = None,
        completion_event: Any | None = None,
    ) -> "CUDAMessagingFuture":
        # TODO: need extra type checking for the future type
        return CUDAMessagingFuture.FromMessagingFuture(  # type: ignore
            self, device, completion_event
        )

    def _complete(
        self,
        result: T | None = None,
        exception: BaseException | None = None,
    ) -> bool:
        """Commit the first terminal state and release retained resources once."""
        with self._completion_lock:
            if self.is_done_.is_set():
                return False
            self.result_ = result
            self.exception_ = exception
            retained_resources = self._retained_resources
            self._retained_resources = []
            self.is_done_.set()

        # Destructors may synchronize or invoke runtime cleanup. Run them
        # outside the state lock after the terminal state is visible.
        retained_resources.clear()
        return True

    def _expire(self, exception: BaseException) -> bool:
        """Atomically expire an unanswered future and notify its transport."""
        if not self._complete(exception=exception):
            return False
        if self._on_timeout is not None:
            self._on_timeout()
        return True

    def _raise_if_failed(self) -> None:
        """Raise the terminal failure without attaching state-owned timeouts."""
        exception = self.exception_
        if exception is None:
            return
        if isinstance(exception, LMCacheTimeoutError):
            raise LMCacheTimeoutError(str(exception))
        raise exception


class CUDAMessagingFuture(MessagingFuture[T]):
    """
    Wraps a result future and a CUDA IPC completion event. ``query``, ``wait``,
    and ``result`` first wait for the response and then for device completion.
    The original future returns ``tuple[bytes, T]``. When the exporter supplies
    ``completion_event``, this future retains and synchronizes that local event;
    otherwise it imports the serialized event for legacy callers.
    """

    def __init__(
        self,
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: Any | None = None,
        completion_event: Any | None = None,
    ) -> None:
        super().__init__()
        self.raw_future_ = raw_future
        self.event_: Any | None = None
        self.exported_event_: Any | None = completion_event
        self.result_: T | None = None
        self.device_ = device if device is not None else torch_dev.current_device()
        if completion_event is not None:
            # The caller-visible CUDA future may be abandoned after a timeout.
            # Keep an independent reference on the raw transport future until
            # its first terminal state so caller abandonment cannot release an
            # exporter while the request is still pending.
            raw_future.retain_until_complete(completion_event)

    def _on_raw_future_complete(self):
        """
        Update the CUDA event and result when the raw future is complete.
        """
        event_bytes, result = self.raw_future_.result()
        self.result_ = result

        if self.exported_event_ is not None:
            self.event_ = self.exported_event_
            self.exported_event_ = None
            return

        # Legacy callers do not retain an exporter-owned event, so import the
        # completion handle created by the server.
        if not hasattr(torch_dev, "Event") or not hasattr(
            torch_dev.Event, "from_ipc_handle"
        ):
            raise RuntimeError(
                f"Backend '{torch_device_type}' does not support interprocess "
                "Events (Event.from_ipc_handle not available). "
                "Multiprocess IPC requires CUDA."
            )
        self.event_ = torch_dev.Event.from_ipc_handle(self.device_, event_bytes)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done, with the CUDA stream.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the CUDA event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            bool: True if the future is done, False if the timeout was reached.

        Raises:
            ValueError: if the timeout is not None.

        Notes:
            This function does not support waiting for a specific time.
        """
        if self.event_:
            self.event_.synchronize()
            return True

        flag = self.raw_future_.wait(timeout)
        if not flag:
            return False

        self._on_raw_future_complete()

        assert self.event_ is not None
        self.event_.synchronize()

        return True

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the CUDA event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            timeout_error = LMCacheTimeoutError(
                "CUDAMessagingFuture result not available within timeout"
            )
            if self.raw_future_._expire(timeout_error):
                # The raw future owns the terminal sentinel. Raise a fresh
                # instance so its traceback cannot retain this CUDA wrapper
                # and the exporter event reachable from it.
                raise LMCacheTimeoutError(str(timeout_error))
            # The raw response won the timeout race; consume it normally.
            return self.result()

        assert self.result_ is not None
        return self.result_

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        if self.event_:
            return self.event_.query()

        if self.raw_future_.query():
            self._on_raw_future_complete()
            assert self.event_ is not None
            return self.event_.query()

        return False

    def set_result(self, result: T) -> None:
        raise NotImplementedError(
            "CUDAMessagingFuture does not support set_result directly"
        )

    @staticmethod
    def FromMessagingFuture(
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: Any | None = None,
        completion_event: Any | None = None,
    ) -> "CUDAMessagingFuture[T]":
        return CUDAMessagingFuture(raw_future, device, completion_event)
