#!/usr/bin/env bash
# Start the NanoNemo dev container with the repo mounted at /workspace/nanonemo.
#   scripts/run_container.sh              -> interactive shell
#   scripts/run_container.sh python x.py  -> run one command
set -euo pipefail

IMAGE=nanonemo:dev
REPO=$(cd "$(dirname "$0")/.." && pwd)

# Persistent home inside the container: keeps Triton/torch.compile caches and
# HF/W&B credentials across runs instead of recompiling kernels every time.
CACHE_HOME="${NANONEMO_CACHE:-$HOME/.cache/nanonemo-container}"
mkdir -p "$CACHE_HOME"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" "$REPO/docker"
fi

TTY_FLAGS=()
[ -t 0 ] && TTY_FLAGS=(-it)

# --ipc=host and the ulimits are NGC's recommended flags for PyTorch
# (shared memory for data loader workers, pinned memory, deep stacks).
exec docker run --rm "${TTY_FLAGS[@]}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/dev \
    -e PYTHONPATH=/workspace/nanonemo \
    -v "$CACHE_HOME":/home/dev \
    -v "$REPO":/workspace/nanonemo \
    "$IMAGE" "$@"
