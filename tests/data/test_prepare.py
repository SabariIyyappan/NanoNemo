import numpy as np
import pytest

from nanonemo.data.prepare import ShardWriter, Stats, assign_split, dedup_and_filter, prepare
from nanonemo.data.shard import open_shard, read_header
from nanonemo.data.tokenizer import decode


def make_docs(n: int, repeat_every: int = 0) -> list[dict]:
    docs = [{"id": f"doc-{i}", "text": f"This is document number {i}. " * 20} for i in range(n)]
    if repeat_every:
        for i in range(0, n, repeat_every):
            docs.append({"id": f"dup-of-{i}", "text": docs[i]["text"]})  # same text, different id
    return docs


def test_dedup_drops_exact_duplicates_not_distinct_docs():
    docs = make_docs(10, repeat_every=3)
    stats = Stats()
    kept = list(dedup_and_filter(docs, min_chars=10, stats=stats))
    assert stats.docs_seen == len(docs)
    assert stats.docs_duplicate == 4  # i = 0, 3, 6, 9
    assert stats.docs_kept == len(kept) == 10


def test_filters_short_documents():
    docs = [{"id": "short", "text": "hi"}, {"id": "long", "text": "x" * 500}]
    stats = Stats()
    kept = list(dedup_and_filter(docs, min_chars=100, stats=stats))
    assert [d["id"] for d in kept] == ["long"]
    assert stats.docs_too_short == 1


def test_assign_split_is_deterministic_and_roughly_proportioned():
    ids = [f"doc-{i}" for i in range(5000)]
    splits = [assign_split(i, val_frac=0.1) for i in ids]
    assert splits == [assign_split(i, val_frac=0.1) for i in ids]  # same input -> same output
    val_frac_observed = splits.count("val") / len(splits)
    assert 0.08 < val_frac_observed < 0.12


def test_shard_writer_flushes_without_splitting_a_document(tmp_path):
    writer = ShardWriter(tmp_path, "train", shard_tokens=10)
    for n in (4, 4, 4, 4):  # 2nd doc of 4 tokens would overflow a 10-token shard
        writer.add(np.arange(n, dtype=np.uint16))
    shards = writer.close()
    assert [s["n_tokens"] for s in shards] == [8, 8]  # never exceeds shard_tokens mid-document
    for shard in shards:
        assert read_header(tmp_path / shard["path"])["n_tokens"] == shard["n_tokens"]


def test_prepare_end_to_end_small(tmp_path):
    docs = make_docs(50, repeat_every=10)
    manifest = prepare(
        iter(docs), tmp_path, min_chars=10, val_frac=0.2, shard_tokens=64, source={"kind": "test"}
    )

    assert manifest["stats"]["docs_seen"] == len(docs)
    assert manifest["stats"]["docs_kept"] == 50  # the 5 duplicates were dropped before splitting

    total_tokens_in_files = 0
    for split in ("train", "val"):
        info = manifest["splits"][split]
        assert info["n_tokens"] == sum(s["n_tokens"] for s in info["shards"])
        for shard in info["shards"]:
            path = tmp_path / shard["path"]
            tokens = open_shard(path)
            assert len(tokens) == shard["n_tokens"]
            total_tokens_in_files += len(tokens)
    assert total_tokens_in_files == manifest["splits"]["train"]["n_tokens"] + manifest["splits"]["val"]["n_tokens"]

    # Spot check: decoding the first train shard's tokens gives back real text.
    first_shard = manifest["splits"]["train"]["shards"][0]
    text = decode(open_shard(tmp_path / first_shard["path"]))
    assert "This is document number" in text


def test_prepare_never_splits_one_document_across_splits(tmp_path):
    # Every doc's full token stream (ending in EOT) lands in exactly one split.
    docs = make_docs(30)
    manifest = prepare(iter(docs), tmp_path, min_chars=1, val_frac=0.3, shard_tokens=10_000, source={})
    seen_ids = set()
    for split in ("train", "val"):
        for shard in manifest["splits"][split]["shards"]:
            text = decode(open_shard(tmp_path / shard["path"]))
            for doc in docs:
                if doc["text"].strip() in text:  # prepare strips surrounding whitespace
                    assert doc["id"] not in seen_ids, f"{doc['id']} appears in more than one split"
                    seen_ids.add(doc["id"])
    assert len(seen_ids) == 30


def test_prepare_rejects_empty_val_split(tmp_path):
    # 3 docs with val_frac=0.001: (1 - 0.001)^3 ≈ 99.7% chance that no doc lands in val.
    docs = [{"id": f"d{i}", "text": "some text " * 30} for i in range(3)]
    with pytest.raises(ValueError, match="val split is empty"):
        prepare(iter(docs), tmp_path, min_chars=1, val_frac=0.001, shard_tokens=1000, source={})


def test_max_train_tokens_stops_early(tmp_path):
    docs = make_docs(2000)
    manifest = prepare(
        iter(docs), tmp_path, min_chars=1, val_frac=0.05, shard_tokens=10_000, source={},
        tokenize_batch_size=16, max_train_tokens=5_000,
    )
    train = manifest["splits"]["train"]["n_tokens"]
    per_doc = max(len(d["text"]) for d in docs)  # chars bound tokens from above
    assert 5_000 <= train <= 5_000 + 16 * per_doc  # overshoot at most one tokenize batch
    assert manifest["stats"]["docs_seen"] < 2000
