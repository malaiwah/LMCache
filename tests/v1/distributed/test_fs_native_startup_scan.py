# SPDX-License-Identifier: Apache-2.0
"""Restart accounting for the native filesystem L2 adapter."""

# Standard
import os

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _object_key_to_filename,
)
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    _scan_existing_fs_native_files,
)


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test-model",
        kv_rank=0,
    )


def _write(root, key: ObjectKey, size: int, mtime_ns: int) -> None:
    path = root / _object_key_to_filename(key)
    path.write_bytes(b"x" * size)
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_scan_returns_oldest_to_newest_keys_and_sizes(tmp_path):
    newer = _key(2)
    older = _key(1)
    _write(tmp_path, newer, 200, 2_000_000_000)
    _write(tmp_path, older, 100, 1_000_000_000)

    keys, sizes = _scan_existing_fs_native_files(str(tmp_path))

    assert keys == [older, newer]
    assert sizes == [100, 200]


def test_scan_skips_foreign_empty_and_temporary_files(tmp_path):
    valid = _key(1)
    _write(tmp_path, valid, 100, 1_000_000_000)
    (tmp_path / "not-a-cache-file.data").write_bytes(b"x")
    (tmp_path / _object_key_to_filename(_key(2))).write_bytes(b"")
    (tmp_path / "pending.tmp").write_bytes(b"x")

    keys, sizes = _scan_existing_fs_native_files(str(tmp_path))

    assert keys == [valid]
    assert sizes == [100]


def test_scan_missing_directory_returns_empty(tmp_path):
    assert _scan_existing_fs_native_files(str(tmp_path / "missing")) == ([], [])


def test_scan_fails_closed_when_existing_file_cannot_be_inspected(
    tmp_path, monkeypatch
):
    path = tmp_path / "broken.data"
    path.write_bytes(b"x")
    original_stat = type(path).stat

    def fail_data_stat(self, *args, **kwargs):
        if self == path:
            raise OSError("injected stat failure")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(path), "stat", fail_data_stat)

    with pytest.raises(RuntimeError, match="Could not inspect cache file"):
        _scan_existing_fs_native_files(str(tmp_path))
