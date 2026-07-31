# SPDX-License-Identifier: Apache-2.0
"""Native FS connector completion semantics.

These tests exercise the compiled extension. They are skipped when the
optional native FS module is not part of the test environment.
"""

# Standard
import select
import time

# Third Party
import pytest

lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")


def _wait_for_completion(client, timeout: float = 5.0):
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if poller.poll(50):
            completions = client.drain_completions()
            if completions:
                return completions
    raise AssertionError("native connector completion timed out")


def test_batch_set_reports_partial_results_and_continues(tmp_path):
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1)
    try:
        buffers = [bytearray(b"first"), bytearray(b"bad"), bytearray(b"last")]
        keys = [
            "model@00000000@01",
            "malformed-key",
            "model@00000000@03",
        ]

        future_id = client.submit_batch_set(
            keys,
            [memoryview(buffer) for buffer in buffers],
        )
        completions = _wait_for_completion(client)

        assert len(completions) == 1
        completed_id, ok, error, results = completions[0]
        assert completed_id == future_id
        assert ok is False
        assert "partially failed" in error
        assert results == [True, False, True]
        assert len(list(tmp_path.glob("*.data"))) == 2
    finally:
        client.close()
