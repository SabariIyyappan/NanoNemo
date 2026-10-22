"""Turn raw documents into token shards.

Pipeline: read docs -> exact-dedup + length filter -> split train/val (by
document, before tokenizing) -> tokenize -> pack into shards -> write a manifest.

The pure, testable logic (dedup, filtering, splitting, packing) is kept
separate from the only two I/O operations: reading the source documents and
writing files to disk.
"""

import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nanonemo.data.shard import write_shard
from nanonemo.data.tokenizer import TOKENIZER_NAME, encode_batch


@dataclass
class Stats:
    docs_seen: int = 0
    docs_duplicate: int = 0
    docs_too_short: int = 0
    docs_kept: int = 0


def load_hf_sample_docs(repo_id: str, path: str) -> Iterator[dict]:
    """Read one parquet file of a Hugging Face *dataset* repo. Needs network access."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id, path, repo_type="dataset")
    df = pd.read_parquet(local_path)
    for row in df.itertuples(index=False):
        yield {"id": row.id, "text": row.text}


def dedup_and_filter(docs: Iterable[dict], min_chars: int, stats: Stats) -> Iterator[dict]:
    """Drop exact-duplicate text (by hash, so we never hold full text twice) and
    documents shorter than `min_chars`. Mutates `stats` as it goes."""
    seen_hashes: set[bytes] = set()
    for doc in docs:
        stats.docs_seen += 1
        text = doc["text"].strip()
        if len(text) < min_chars:
            stats.docs_too_short += 1
            continue
        h = hashlib.sha256(text.encode()).digest()
        if h in seen_hashes:
            stats.docs_duplicate += 1
            continue
        seen_hashes.add(h)
        stats.docs_kept += 1
        yield {**doc, "text": text}


def assign_split(doc_id: str, val_frac: float) -> str:
    """Deterministic train/val split from a hash of the document id.

    Splitting here, before tokenization, means a document's tokens never
    straddle the train/val boundary: the held-out set is held out by document,
    not by an arbitrary cut partway through the concatenated token stream.
    """
    digest = hashlib.sha256(doc_id.encode()).digest()
    frac = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    return "val" if frac < val_frac else "train"


class ShardWriter:
    """Buffers tokenized documents and flushes a shard whenever the buffer fills.

    Fed incrementally (one document's tokens at a time), so at most one
    shard's worth of tokens is ever held in memory for this split - not the
    whole split. Call `close()` once at the end to flush the remainder.
    """

    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.shards: list[dict] = []
        self._buffer: list[np.ndarray] = []
        self._buffered_len = 0
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, tokens: np.ndarray) -> None:
        if self._buffered_len + len(tokens) > self.shard_tokens and self._buffer:
            self._flush()
        self._buffer.append(tokens)
        self._buffered_len += len(tokens)

    def _flush(self) -> None:
        if not self._buffer:
            return
        tokens = np.concatenate(self._buffer)
        path = self.out_dir / f"{self.prefix}_{len(self.shards):06d}.bin"
        info = write_shard(path, tokens, tokenizer=TOKENIZER_NAME)
        self.shards.append({"path": path.name, **info})
        self._buffer.clear()
        self._buffered_len = 0

    def close(self) -> list[dict]:
        self._flush()
        return self.shards


def prepare(
    docs: Iterable[dict],
    out_dir: Path,
    *,
    min_chars: int = 200,
    val_frac: float = 0.02,
    shard_tokens: int = 100_000_000,
    source: dict,
    tokenize_batch_size: int = 256,
) -> dict:
    stats = Stats()
    filtered = dedup_and_filter(docs, min_chars=min_chars, stats=stats)

    writers = {split: ShardWriter(out_dir, split, shard_tokens) for split in ("train", "val")}
    # Small per-split text buffers: enough documents to tokenize efficiently
    # in parallel (encode_batch spreads them over threads), never a whole split.
    text_buffers: dict[str, list[str]] = {"train": [], "val": []}

    def drain(split: str) -> None:
        buf = text_buffers[split]
        if not buf:
            return
        for tokens in encode_batch(buf):
            writers[split].add(tokens)
        buf.clear()

    for doc in filtered:
        split = assign_split(doc["id"], val_frac)
        text_buffers[split].append(doc["text"])
        if len(text_buffers[split]) >= tokenize_batch_size:
            drain(split)
    for split in text_buffers:
        drain(split)  # each split's final, partial batch

    manifest = {
        "tokenizer": TOKENIZER_NAME,
        "min_chars": min_chars,
        "val_frac": val_frac,
        "source": source,
        "splits": {},
    }
    for split, writer in writers.items():
        shards = writer.close()
        manifest["splits"][split] = {"n_tokens": sum(s["n_tokens"] for s in shards), "shards": shards}
    manifest["stats"] = vars(stats)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default="nvidia/Nemotron-Pretraining-Dataset-sample")
    parser.add_argument("--hf-path", default="Nemotron-CC-High-Quality/part_000000.parquet")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--shard-tokens", type=int, default=2_000_000)
    args = parser.parse_args()

    docs = load_hf_sample_docs(args.repo_id, args.hf_path)
    source = {"kind": "hf_sample", "repo_id": args.repo_id, "path": args.hf_path}
    manifest = prepare(
        docs, args.out, min_chars=args.min_chars, val_frac=args.val_frac, shard_tokens=args.shard_tokens, source=source
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["stats"], indent=2))
    print(json.dumps({k: v["n_tokens"] for k, v in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
