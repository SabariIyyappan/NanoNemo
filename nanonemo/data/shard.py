"""Binary token shard format.

Layout: a 256-byte header, then n_tokens little-endian uint16 token ids.
Header: magic (8s) | version (u32) | n_tokens (u64) | tokenizer name (32s), zero-padded to 256 bytes.
"""

import hashlib
import os
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NNMOTOKS"
VERSION = 1
HEADER_BYTES = 256
_HEADER = struct.Struct("<8sIQ32s")
DTYPE = np.dtype("<u2")


def _pack_header(n_tokens: int, tokenizer: str) -> bytes:
    raw = _HEADER.pack(MAGIC, VERSION, n_tokens, tokenizer.encode())
    return raw.ljust(HEADER_BYTES, b"\0")


def read_header(path: str | Path) -> dict:
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path}: file too short for a shard header")
    magic, version, n_tokens, tokenizer = _HEADER.unpack_from(raw)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}, not a NanoNemo token shard")
    if version != VERSION:
        raise ValueError(f"{path}: shard version {version}, expected {VERSION}")
    expected_size = HEADER_BYTES + n_tokens * DTYPE.itemsize
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise ValueError(f"{path}: size {actual_size} != {expected_size} implied by header (truncated?)")
    return {"version": version, "n_tokens": n_tokens, "tokenizer": tokenizer.rstrip(b"\0").decode()}


def write_shard(path: str | Path, tokens: np.ndarray, tokenizer: str) -> dict:
    """Write tokens atomically; return {"n_tokens", "sha256"} of the finished file."""
    path = Path(path)
    tokens = np.ascontiguousarray(tokens, dtype=DTYPE)
    header = _pack_header(len(tokens), tokenizer)
    sha = hashlib.sha256(header)
    sha.update(tokens.data)

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(tokens.data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX: readers never see a half-written shard
    return {"n_tokens": len(tokens), "sha256": sha.hexdigest()}


def open_shard(path: str | Path) -> np.memmap:
    """Memory-map a shard's tokens read-only; pages are loaded lazily by the OS."""
    header = read_header(path)
    return np.memmap(path, dtype=DTYPE, mode="r", offset=HEADER_BYTES, shape=(header["n_tokens"],))


def file_sha256(path: str | Path, chunk_bytes: int = 1 << 24) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            sha.update(chunk)
    return sha.hexdigest()
