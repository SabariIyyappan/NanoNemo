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

In a notebook, attach the dataset with "Add Input". It appears under `/kaggle/input/datasets/sabariiyyappand/nanonemo-nemotron-cc/` (the `datasets/<owner>/` level is new; `find /kaggle/input -name manifest.json` shows the exact path). Then check the shards against the manifest before training:
```bash
python -c "from nanonemo.data.shard import read_header; import json, pathlib; d = pathlib.Path('/kaggle/input/datasets/sabariiyyappand/nanonemo-nemotron-cc'); [read_header(d / s['path']) for sp in json.load(open(d / 'manifest.json'))['splits'].values() for s in sp['shards']]; print('shards OK')"
```
`read_header` checks magic, version and exact file size, so a truncated upload fails here.

## Ladder benchmarks on a T4
One GPU of the 2× T4 notebook is enough (the harness uses `cuda:0`). In a notebook with the dataset attached and the GPU accelerator on:
```bash
!git clone https://github.com/SabariIyyappan/NanoNemo.git && cd NanoNemo && pip install -q -e .
!cd NanoNemo && python scripts/smoke_gpu.py   # torch must list sm_75; reports the Triton tl.dot status
!cd NanoNemo && for s in s0_fp32 s1_fp16 s2_mem_efficient s3_compile s4_cudagraphs; do \
    python scripts/bench_train.py --config configs/stages/$s.yaml --data-dir /kaggle/input/datasets/sabariiyyappand/nanonemo-nemotron-cc; done
```
Results land in `NanoNemo/results/ladder/t4/`, next to the committed laptop results that came with the clone. Zip and download that folder (`zip -r /kaggle/working/t4_results.zip results/ladder/t4`, then the Output panel) and unzip it at the repo root locally. The whole ladder takes about 20 minutes on a T4.

## Multi-GPU runs (M7) and the main run (M9)
`notebooks/kaggle_train.ipynb` is the launcher: upload it as a Kaggle notebook (File → Import Notebook), set **Accelerator: GPU T4 x2**, turn Internet on, and attach the token dataset. Its cells, in order:

1. Clone the repo, install `requirements-kaggle.txt` (Kaggle's own torch is kept), run `scripts/smoke_gpu.py`.
2. Find the dataset (`DATA`) and any attached previous checkpoint (`PREV`).
3. `pytest -m multigpu tests/distributed` with `NANONEMO_TEST_DEVICE=cuda`: DDP / FSDP2 / TP on 2 GPUs match 1 GPU, and resume reproduces an uninterrupted run (the same tests run on the laptop's CPU with gloo).
4. The multi-GPU table: `configs/dist/*.yaml` through `scripts/bench_train.py` (torchrun for the 2-GPU rows), then `scripts/dist_table.py`. Results in `results/dist/t4-x1/` and `t4-x2/`.
5. `scripts/nsys_dist.sh` per row: an Nsight Systems trace of one trial and the NCCL share of GPU time.
6. The LR sweep: `configs/lr_sweep/*.yaml`, 100M tokens each.
7. The main run, `configs/main.yaml`.
8. Zip `results/` into `/kaggle/working` for download.

Steps 3-6 fit in one session (roughly 3 h). Pick the main run's `parallel` and `peak_lr` from them and update `configs/main.yaml` before starting step 7.

## Checkpoint hand-off between sessions
A Kaggle session ends after 12 hours, and only `/kaggle/working` survives, as the notebook's output. The main run is built around that:

- `train.stop_after_min=690` makes the run stop itself after 11.5 h: every rank agrees on the step to stop at, writes a checkpoint, and exits cleanly (exit code 0, so the notebook "Save & Run All" finishes).
- Checkpoints go to `/kaggle/working/main/ckpt/step_NNNNNN/` every `ckpt_every` steps (250 steps, roughly 40-60 min) and at the stop. They are DCP checkpoints: each GPU writes its own shard in parallel, in a background thread (`ckpt_async`), so training continues while the files are written. `ckpt/latest` names the newest *complete* checkpoint; only the last two are kept.
- Each checkpoint holds the model, the AdamW moments, and `trainer.json`: the step (so the LR schedule continues where it was), the loader position (so no batch is seen twice or skipped), and the FP16 loss scale.

To continue in a new session:
1. In the finished session's page, **Output → New Dataset** (or "Add to dataset" with a new version), e.g. `nanonemo-main-ckpt`. Only `main/ckpt` matters (~1.4 GB: FP32 weights + 2 Adam moments).
2. In the next session, attach that dataset as an input next to the token data. The notebook finds `.../ckpt/latest` and passes `train.resume_from=` automatically.
3. `metrics.jsonl` in the new output continues the old one (the resumed run appends), so the loss curve is the concatenation of each session's file; the `start` record of each session marks the resume point.

DCP reshards on load, so a checkpoint is not tied to the layout that wrote it: an FSDP checkpoint from 2 GPUs loads on 1 GPU (tested), and the loader position is converted between data-parallel world sizes.

**GPU quota.** Kaggle gives ~30 GPU-hours per week. At ~29k tokens/s (2x the single-T4 S3 rate) the 2.4B-token main run takes ~23 h: two to three sessions, plus ~3 h for the M7 session, which can cross a weekly quota boundary.
