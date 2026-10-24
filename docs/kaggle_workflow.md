# Kaggle workflow

Kaggle provides 2× T4 notebooks with time-limited sessions. Data preparation happens locally, so no Kaggle GPU time is spent on it. Training resumes from checkpoints across sessions.

## One-time setup
Kaggle's CLI (2.x) authenticates with a single token rather than the older `kaggle.json` key pair:

1. Go to https://www.kaggle.com/settings/api and click **Generate New Token**. Copy the token string shown (it is not downloaded as a file this time).
2. Give it to the CLI in *one* of these ways:
   - **Environment variable** (simplest; `scripts/run_container.sh` forwards it into the container if it's set in your shell):
     ```bash
     export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx
     ```
   - **Token file** (persists across shells without re-exporting):
     ```bash
     mkdir -p ~/.kaggle && echo -n xxxxxxxxxxxxxx > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
     ```
   `scripts/run_container.sh` mounts `~/.kaggle` read-only into the container either way.

   (The legacy `~/.kaggle/kaggle.json` from "Create New Token" on older accounts still works too, if that's what your account shows instead.)

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

## Ladder benchmarks on a T4
One GPU of the 2× T4 notebook is enough (the harness uses `cuda:0`). In a notebook with the dataset attached and the GPU accelerator on:
```bash
!git clone https://github.com/SabariIyyappan/NanoNemo.git && cd NanoNemo && pip install -q -e .
!cd NanoNemo && python scripts/smoke_gpu.py   # torch must list sm_75; reports the Triton tl.dot status
!cd NanoNemo && for s in s0_fp32 s1_fp16 s2_mem_efficient s3_compile s4_cudagraphs; do \
    python scripts/bench_train.py --config configs/stages/$s.yaml --data-dir /kaggle/input/nanonemo-nemotron-cc; done
```
Results land in `NanoNemo/results/ladder/t4/`. Download that folder from the notebook's output and copy it into `results/ladder/t4/` locally. The whole ladder takes about 20 minutes on a T4.

## Checkpoint hand-off between sessions
_To be written in M7, together with the DCP checkpointing code._
