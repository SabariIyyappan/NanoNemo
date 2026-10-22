import json

import zstandard

from nanonemo.data.sources import iter_jsonl_zstd, select_evenly


def test_select_evenly_spans_the_whole_list():
    paths = [f"p{i:04d}" for i in range(100)]
    picked = select_evenly(paths, 4)
    assert picked == ["p0000", "p0025", "p0050", "p0075"]
    assert select_evenly(paths[:3], 10) == paths[:3]


def test_iter_jsonl_zstd_streams_records(tmp_path):
    records = [{"text": f"doc {i} " * 50, "warc_record_id": f"<urn:uuid:{i}>", "url": "u", "language": "en"} for i in range(500)]
    raw = "".join(json.dumps(r) + "\n" for r in records).encode()
    path = tmp_path / "part.jsonl.zstd"
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))

    docs = list(iter_jsonl_zstd(path))
    assert [d["id"] for d in docs] == [r["warc_record_id"] for r in records]
    assert docs[123]["text"] == records[123]["text"]
