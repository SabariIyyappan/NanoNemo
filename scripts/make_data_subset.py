"""Copy the first N train shards + the val split into a new data dir with its own manifest.

    python scripts/make_data_subset.py data/nemotron_cc data/nemotron_cc_1shard --train-shards 1

For short runs on a rented GPU (M8): one 100M-token shard is ~200 MB instead of ~4.8 GB.
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--train-shards", type=int, default=1)
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    m = json.loads((src / "manifest.json").read_text())
    train = m["splits"]["train"]
    train["shards"] = train["shards"][: args.train_shards]
    train["n_tokens"] = sum(s["n_tokens"] for s in train["shards"])
    m["subset_of"] = str(src)
    for split in m["splits"].values():
        for s in split["shards"]:
            shutil.copy2(src / s["path"], dst / s["path"])
    (dst / "manifest.json").write_text(json.dumps(m, indent=2))
    print(f"{dst}: {train['n_tokens']:,} train tokens, {m['splits']['val']['n_tokens']:,} val tokens")


if __name__ == "__main__":
    main()
