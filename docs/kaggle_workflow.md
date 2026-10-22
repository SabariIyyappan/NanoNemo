# Kaggle workflow

Kaggle provides 2× T4 notebooks with time-limited sessions. Data preparation happens locally, so no Kaggle GPU time is spent on it. Training resumes from checkpoints across sessions.

## One-time setup
1. Kaggle → Settings → API → **Create New Token**. This downloads `kaggle.json`.
2. Put it on the host machine:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
   `scripts/run_container.sh` mounts `~/.kaggle` read-only into the container.

## Uploading the token dataset
```bash
# 1. Download and tokenize locally (Nemotron-CC v1, quality=high, real text only)
scripts/run_container.sh python -m nanonemo.data.prepare --source nemotron_cc --cc-files 12 \
    --out data/nemotron_cc --max-train-tokens 2500000000 --val-frac 0.004 --shard-tokens 100000000
# 2. Upload the shards and manifest as a private Kaggle dataset
scripts/run_container.sh scripts/kaggle_upload_data.sh data/nemotron_cc nanonemo-nemotron-cc
```
The dataset is private. Only `*.bin` and `manifest.json` are uploaded; the raw `.jsonl.zstd` downloads stay local.

In a notebook, attach the dataset with "Add Input". It appears under `/kaggle/input/nanonemo-nemotron-cc/`. Then check the shards against the manifest before training:
```bash
python -c "from nanonemo.data.shard import read_header; import json, pathlib; d = pathlib.Path('/kaggle/input/nanonemo-nemotron-cc'); [read_header(d / s['path']) for sp in json.load(open(d / 'manifest.json'))['splits'].values() for s in sp['shards']]; print('shards OK')"
```
`read_header` checks magic, version and exact file size, so a truncated upload fails here.

## Checkpoint hand-off between sessions
_To be written in M7, together with the DCP checkpointing code._
