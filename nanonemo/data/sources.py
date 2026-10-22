"""Document sources: Nemotron-CC v1 from Common Crawl's open host.

Files are downloaded once into a local raw cache (resumable, size-verified),
then streamed through zstd decompression one JSON line at a time, so neither
the download nor the decompressed text ever has to fit in memory.
"""

import gzip
import io
import json
import os
from collections.abc import Iterator
from pathlib import Path

import requests

CC_BASE = "https://data.commoncrawl.org/"
NEMOTRON_CC_PATHS = CC_BASE + "contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"


def nemotron_cc_paths(quality: str = "high", kind: str = "actual") -> list[str]:
    """All file paths of one quality bucket; kind="actual" is real web text (no synthetic rephrasing)."""
    resp = requests.get(NEMOTRON_CC_PATHS, timeout=60)
    resp.raise_for_status()
    paths = gzip.decompress(resp.content).decode().split()
    marker = f"/quality={quality}/kind={kind}/kind2={kind}/"
    return sorted(p for p in paths if marker in p)


def select_evenly(paths: list[str], n: int) -> list[str]:
    """n files evenly spaced through the sorted list, so every crawl era (2013-2024) is represented
    roughly in proportion to its share of the bucket."""
    if n >= len(paths):
        return list(paths)
    stride = len(paths) / n
    return [paths[int(i * stride)] for i in range(n)]


def download(path: str, raw_dir: Path, chunk_bytes: int = 1 << 22) -> Path:
    """Download CC_BASE + path to raw_dir, resuming a partial download; verified by size."""
    dest = raw_dir / Path(path).name
    if dest.exists():
        return dest
    raw_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = CC_BASE + path

    total = int(requests.head(url, timeout=60).headers["Content-Length"])
    while (have := part.stat().st_size if part.exists() else 0) < total:
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                if have and resp.status_code != 206:
                    raise RuntimeError(f"{url}: server ignored the Range request")
                with open(part, "ab") as f:
                    for chunk in resp.iter_content(chunk_bytes):
                        f.write(chunk)
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"retrying {dest.name} from byte {part.stat().st_size if part.exists() else 0}: {e}")
    if part.stat().st_size != total:
        raise RuntimeError(f"{dest.name}: got {part.stat().st_size} bytes, expected {total}")
    os.replace(part, dest)
    return dest


def iter_jsonl_zstd(path: Path) -> Iterator[dict]:
    import zstandard

    with open(path, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            record = json.loads(line)
            yield {"id": record["warc_record_id"], "text": record["text"]}
