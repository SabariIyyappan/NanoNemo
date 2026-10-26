# AWS L4 runbook (M8: precision table)

Turing (laptop, T4) has no BF16 or FP8 tensor cores, so the precision rows run on one NVIDIA L4 (Ada, sm_89) rented on AWS. Everything is scripted; the instance should live for about 2–3 hours.

**Nothing here is launched without the project owner's explicit go-ahead** (it costs money).

## What runs

`scripts/run_precision.sh` runs each `configs/precision/*.yaml` twice: a throughput benchmark (`scripts/bench_train.py`, 20 timed steps × 3 trials) and a 20M-token short training run for the loss curve.

| Config | Precision | Linear layers |
|---|---|---|
| `fp16` | FP16 AMP + GradScaler (our Turing stack) | nn.Linear |
| `bf16` | BF16 AMP | nn.Linear |
| `bf16_te` | BF16 AMP | Transformer Engine `te.Linear` (isolates the layer swap) |
| `fp8_delayed` | FP8 GEMMs (HYBRID: E4M3 fwd, E5M2 grads), delayed scaling, BF16 elsewhere | `te.Linear` |
| `fp8_current` | same, per-step (current) scaling | `te.Linear` |

Expected time: ~5 min benchmark + ~10–15 min short run per row at L4 speeds, so ~1.5–2 h plus setup.

## Cost

Instance: **g6.xlarge** (1× L4 24 GB, 4 vCPU, 16 GB RAM). Check the current on-demand price for your region on the EC2 pricing page before launching (about $0.80/h in us-east-1 when this was written). Budget: ≤ 3 instance-hours plus a 100 GB gp3 volume for those hours, i.e. a few dollars. Spot instances are cheaper but can be reclaimed mid-run.

## Steps

1. **Prepare the data subset locally** (one 100M-token train shard + val, ~220 MB):
   ```bash
   scripts/run_container.sh python scripts/make_data_subset.py data/nemotron_cc data/nemotron_cc_1shard --train-shards 1
   tar czf /tmp/nemotron_cc_1shard.tgz -C data nemotron_cc_1shard
   ```
2. **Launch** a g6.xlarge with the AWS "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)" (driver, Docker and the NVIDIA container toolkit preinstalled), a 100 GB root volume, your SSH key, and a security group that allows SSH from your IP only. Note the launch time.
3. **Set up** (on the instance):
   ```bash
   git clone https://github.com/SabariIyyappan/NanoNemo.git && cd NanoNemo
   nvidia-smi                               # L4, driver >= 580 for CUDA 13.0 in NGC 25.11
   docker build -t nanonemo:dev docker      # NGC 25.11 base: ~20 GB pull, ~10 min
   ```
   From your machine: `scp /tmp/nemotron_cc_1shard.tgz ubuntu@<ip>:NanoNemo/data/` then on the instance `tar xzf data/nemotron_cc_1shard.tgz -C data`.
4. **Check** FP8 works before the long part:
   ```bash
   scripts/run_container.sh python -m pytest -q tests/model/test_te.py   # FP8 tests must run, not skip
   ```
5. **Run** (inside `tmux`, so an SSH drop doesn't kill it):
   ```bash
   scripts/run_container.sh scripts/run_precision.sh 2>&1 | tee runs/precision.log
   ```
6. **Copy results back** and **terminate immediately**:
   ```bash
   scp -r ubuntu@<ip>:NanoNemo/results/ladder/l4 ubuntu@<ip>:NanoNemo/results/precision ./results/
   ```
   Terminate the instance (not just stop: a stopped instance still bills its volume), and confirm in the console that its state is *terminated* and no volume is left.
7. **Record the spend** in `results/aws_cost.json`:
   ```json
   {"instance": "g6.xlarge", "region": "...", "pricing": "on-demand", "price_per_hour_usd": 0.0,
    "launched": "YYYY-MM-DDTHH:MM", "terminated": "YYYY-MM-DDTHH:MM", "hours": 0.0, "total_usd": 0.0,
    "source": "AWS Billing console / Cost Explorer"}
   ```
   Use the billed amount from the Billing console once it appears (it lags by up to a day), not an estimate.

Then `scripts/make_figures.py` fills the precision table.

## If something fails

- `is_fp8_available` false: the GPU is not an L4/Ada, or the driver is too old for the container.
- Out of memory: lower `micro_batch` to 4 in all five configs (keep them identical to each other).
- A TE row much slower than `bf16`: check the `bf16_te` row; if it is already slow, the cost is the TE layers under `torch.compile` (graph breaks), not FP8.
