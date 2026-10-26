#!/usr/bin/env bash
# Nsight Systems trace of one multi-GPU benchmark config, and the share of GPU kernel time
# spent in NCCL (communication) kernels.
#   scripts/nsys_dist.sh configs/dist/ddp.yaml [nproc] [extra bench_train.py args, e.g. --data-dir ...]
# Writes results/nsys/<gpu>-x<nproc>/<name>.nsys-rep (open in the Nsight Systems GUI: the NCCL
# kernels on each GPU's stream show how much of the communication overlaps compute) and
# <name>_cuda_gpu_kern_sum.csv. Only the first timed trial is captured (NANONEMO_PROFILE=1).
set -euo pipefail
cd "$(dirname "$0")/.."
cfg=$1; n=${2:-2}; shift $(( $# < 2 ? $# : 2 ))
name=$(basename "$cfg" .yaml)
gpu=$(python -c "from nanonemo.utils.env import gpu_slug; print(gpu_slug())")
out=results/nsys/$gpu-x$n
mkdir -p "$out"
NANONEMO_PROFILE=1 nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
    -f true -o "$out/$name" \
    python -m torch.distributed.run --standalone --nproc_per_node="$n" scripts/bench_train.py \
    --config "$cfg" --steps 5 --trials 1 --out "$out/$name.bench.json" "$@"
nsys stats --report cuda_gpu_kern_sum --format csv --output "$out/$name" --force-export true "$out/$name.nsys-rep" > /dev/null
python scripts/nsys_comm.py "$out/${name}_cuda_gpu_kern_sum.csv"
