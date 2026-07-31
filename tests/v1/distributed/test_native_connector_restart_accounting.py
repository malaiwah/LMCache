# SPDX-License-Identifier: Apache-2.0
"""Restart accounting behavior shared by native L2 connectors."""

# Third Party
import pytest

# First Party
from tests.v1.distributed.test_native_connector_l2_adapter import (
    adapter,
    create_object_key,
)


class _RecordingL2Listener:
    def __init__(self):
        self.stored = []

    def on_l2_keys_stored(self, keys, sizes) -> None:
        self.stored.append((list(keys), list(sizes)))


class TestPrimeExistingKeys:
    def test_priming_accounts_deduplicated_usage(self, adapter):
        key = create_object_key(1)

        adapter.prime_existing_keys([key, key], [100, 999])

        assert adapter.get_usage().total_bytes_used == 100
        assert adapter._key_sizes == {key: 100}

    def test_first_listener_receives_startup_snapshot_once(self, adapter):
        keys = [create_object_key(1), create_object_key(2)]
        adapter.prime_existing_keys(keys, [100, 200])
        first = _RecordingL2Listener()
        second = _RecordingL2Listener()

        adapter.register_listener(first)
        adapter.register_listener(second)

        assert first.stored == [(keys, [100, 200])]
        assert second.stored == []
        assert adapter.get_usage().total_bytes_used == 300

    def test_empty_snapshot_is_consumed_without_callback(self, adapter):
        adapter.prime_existing_keys([], [])
        listener = _RecordingL2Listener()

        adapter.register_listener(listener)

        assert listener.stored == []

    def test_rejects_invalid_or_repeated_priming(self, adapter):
        with pytest.raises(ValueError, match="length mismatch"):
            adapter.prime_existing_keys([create_object_key(1)], [1, 2])
        with pytest.raises(ValueError, match="must be positive"):
            adapter.prime_existing_keys([create_object_key(1)], [0])

        adapter.prime_existing_keys([], [])
        with pytest.raises(RuntimeError, match="already primed"):
            adapter.prime_existing_keys([], [])

    def test_rejects_priming_after_listener_registration(self, adapter):
        adapter.register_listener(_RecordingL2Listener())

        with pytest.raises(RuntimeError, match="before listeners"):
            adapter.prime_existing_keys([], [])
