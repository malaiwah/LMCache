# SPDX-License-Identifier: Apache-2.0
"""ObjectKey compatibility of the compiled native FS connector."""

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
                return completions[0]
    raise AssertionError("native connector completion timed out")


@pytest.mark.parametrize(
    ("wire_key", "filename"),
    [
        ("model@00000000@aabb", "model@0x00000000@aabb.data"),
        (
            "model@00000000@aabb@salt",
            "model@0x00000000@aabb@salt.data",
        ),
        (
            "model@00000000@2@aabb",
            "model@0x00000000@2@aabb.data",
        ),
        (
            "model@00000000@2@aabb@salt",
            "model@0x00000000@2@aabb@salt.data",
        ),
    ],
)
def test_legacy_and_current_key_shapes(wire_key, filename, tmp_path):
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1)
    try:
        payload = bytearray(b"payload")
        future_id = client.submit_batch_set([wire_key], [memoryview(payload)])
        completed_id, ok, error, _results = _wait_for_completion(client)

        assert completed_id == future_id
        assert ok, error
        assert (tmp_path / filename).read_bytes() == payload
    finally:
        client.close()
