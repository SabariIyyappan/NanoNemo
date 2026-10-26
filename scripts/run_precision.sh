#!/usr/bin/env bash
# M8 on an L4: for every configs/precision/*.yaml, a throughput benchmark and a 20M-token short run.
#   scripts/run_precision.sh            (inside the container, data in data/nemotron_cc_1shard)
# Results: results/ladder/l4/<name>.json (speed) and results/precision/l4/<name>.jsonl (loss curve).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/precision/l4
for cfg in configs/precision/*.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "=== $name"
  python scripts/bench_train.py --config "$cfg" --steps 20 | grep -E "tokens/s|MFU|peak memory|WARNING|saved"
  python scripts/train.py --config "$cfg" | grep -E "val_loss|step +3[0-9]{2} "
  cp "runs/precision/$name/metrics.jsonl" "results/precision/l4/$name.jsonl"
done
