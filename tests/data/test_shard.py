import numpy as np
import pytest

from nanonemo.data.shard import HEADER_BYTES, file_sha256, open_shard, read_header, write_shard


def test_round_trip(tmp_path):
    tokens = np.random.default_rng(0).integers(0, 50257, size=10_001, dtype=np.uint16)
    path = tmp_path / "shard_000000.bin"
    info = write_shard(path, tokens, tokenizer="gpt2")

    assert read_header(path) == {"version": 1, "n_tokens": 10_001, "tokenizer": "gpt2"}
    np.testing.assert_array_equal(open_shard(path), tokens)
    assert info["sha256"] == file_sha256(path)
    assert not (tmp_path / "shard_000000.bin.tmp").exists()


def test_rejects_truncated_file(tmp_path):
    path = tmp_path / "s.bin"
    write_shard(path, np.arange(100, dtype=np.uint16), tokenizer="gpt2")
    with open(path, "r+b") as f:
        f.truncate(HEADER_BYTES + 150)  # cut mid-way through the tokens
    with pytest.raises(ValueError, match="truncated"):
        read_header(path)


def test_rejects_foreign_file(tmp_path):
    path = tmp_path / "not_a_shard.bin"
    path.write_bytes(b"\0" * 1024)
    with pytest.raises(ValueError, match="bad magic"):
        open_shard(path)
