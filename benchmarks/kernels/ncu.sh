#!/usr/bin/env bash
# Nsight Compute reports for the Triton kernels (run inside the container):
#   scripts/run_container.sh benchmarks/kernels/ncu.sh [rmsnorm swiglu rope]
# Writes results/ncu/<gpu>/<kernel>.ncu-rep (full report, open in the Nsight Compute GUI; not
# tracked in git) and <kernel>.csv (the key metrics per kernel launch, tracked).
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU=$(python -c "from nanonemo.utils.env import gpu_slug; print(gpu_slug())")
OUT=results/ncu/$GPU
mkdir -p "$OUT"
METRICS=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__grid_size,launch__block_size

KERNELS=("$@")
[ ${#KERNELS[@]} -eq 0 ] && KERNELS=(rmsnorm swiglu rope)
for k in "${KERNELS[@]}"; do
  # Each target call launches 2 of our kernels (fwd, bwd); skip the first call's (JIT compile).
  ncu --set full -k "regex:^_(fwd|bwd|rope)_kernel" --launch-skip 2 --launch-count 2 \
      -f -o "$OUT/$k" python benchmarks/kernels/ncu_target.py "$k" > /dev/null
  ncu --import "$OUT/$k.ncu-rep" --csv --page raw --metrics "$METRICS" > "$OUT/$k.csv"
  echo "== $k"; python benchmarks/kernels/ncu_table.py "$OUT/$k.csv"
done
